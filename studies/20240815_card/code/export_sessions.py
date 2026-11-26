# -*- coding: utf-8 -*-
"""Export Card 2024 speech sessions to the common session format.

The publisher stores one directory per recording day of participant T15 with
up to three files: ``data_train.hdf5``, ``data_val.hdf5``, and
``data_test.hdf5``. One session file holds every labeled trial of the day in
the order ``train`` then ``val``, each file in trial order. The 41
``data_test.hdf5`` files hold the second half of the "Val/Test" blocks with
the labels withheld; those trials are not exported, and the root attributes
``withheld_source_file`` and ``withheld_trials`` record what was left out.

Neural features are stored as released. The publisher already standardized
``input_features`` per channel and capped it at 10.0, so nothing is scaled
here. Phoneme targets are the publisher's ``seq_class_ids``, whose ids follow
the 41-class ``LOGIT_TO_PHONEME`` order the released decoder is trained on.
The corpus and the post-implant day come from the publisher's block
description CSV, joined on the session date and the block number.

Usage (from the repository root):

    python studies/20240815_card/code/export_sessions.py --list
    python studies/20240815_card/code/export_sessions.py --session t15.2023.08.11
"""

import csv
import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from session_format import Field, TrialSpec, export_cli, trial_id  # noqa: E402
from study_paths import study_checkout, study_path  # noqa: E402

STUDY = "20240815_card"
DATASET = "https://doi.org/10.5061/dryad.dncjsxm85"
BIN_MS = 20
# publisher partition, file name
PARTITIONS = (("train", "data_train.hdf5"), ("val", "data_val.hdf5"))
# second half of the "Val/Test" blocks; the publisher withheld its labels
WITHHELD_FILE = "data_test.hdf5"
DESCRIPTION = "data/extracted/t15_copyTaskData_description.csv"
FEATURE = {
    "description": "Neural feature vector combined by the publisher before release, "
                   "one row per 20 ms bin.",
    "column_layout": "Columns 0-255: threshold crossings; columns 256-511: spike-band power "
                     "of the same electrodes. Each half holds four 64-electrode arrays in "
                     "the order ventral 6v, area 4, 55b, dorsal 6v (upstream README).",
    "normalization": "Already standardized by the publisher: about zero mean and unit "
                     "variance per channel within a session, capped at 10.0. Stored as "
                     "released; this exporter applies no scaling.",
}
_CACHE = {"path": None, "file": None, "keys": None, "blocks": None}


def _data_dir():
    """Return the publisher's directory of per-day HDF5 files.

    Returns
    -------
    pathlib.Path
        ``data/extracted/dryad_dncjsxm85/hdf5_data_final`` of the study.
    """
    return Path(study_path(STUDY, "data/extracted/hdf5_data_final"))


def _partition_path(session, file_name):
    """Return the path of one partition file of a session.

    Parameters
    ----------
    session : str
        Native session identifier such as ``t15.2023.08.11``.
    file_name : str
        Publisher file name such as ``data_train.hdf5``.

    Returns
    -------
    pathlib.Path
        Path of the file, whether or not it exists.
    """
    return _data_dir() / session / file_name


def _relative(path):
    """Return a publisher file path relative to the study directory.

    Parameters
    ----------
    path : pathlib.Path
        Resolved publisher file.

    Returns
    -------
    str
        POSIX-style path such as
        ``data/extracted/dryad_dncjsxm85/hdf5_data_final/t15.2023.08.11/data_train.hdf5``.
    """
    return Path(path).absolute().relative_to(Path(study_checkout(STUDY)).absolute()).as_posix()


def _open(path):
    """Open one publisher file read-only and keep it as the cached file.

    Parameters
    ----------
    path : pathlib.Path
        Publisher file.

    Returns
    -------
    tuple of (h5py.File, list of str)
        The open file and its trial keys in name order.

    Notes
    -----
    The previous file is closed, so at most one publisher file stays open
    while a session is exported or its trials are re-read one by one.
    """
    if _CACHE["path"] != str(path):
        if _CACHE["file"] is not None:
            _CACHE["file"].close()
        source = h5py.File(path, "r")
        _CACHE.update(path=str(path), file=source, keys=sorted(source.keys()))
    return _CACHE["file"], _CACHE["keys"]


def _blocks():
    """Return the publisher's block descriptions keyed by date and block number.

    Returns
    -------
    dict of tuple to tuple
        ``(date, block number)`` to ``(corpus, post-implant day)``.

    Raises
    ------
    ValueError
        When the CSV describes one block twice.
    """
    if _CACHE["blocks"] is None:
        blocks = {}
        with open(study_path(STUDY, "data/extracted/t15_copyTaskData_description.csv"),
                  encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                key = (row["Date"], int(row["Block number"]))
                if key in blocks:
                    raise ValueError(f"{DESCRIPTION}: block {key} is described twice")
                blocks[key] = (row["Corpus"], int(row["Post-implant day"]))
        _CACHE["blocks"] = blocks
    return _CACHE["blocks"]


def label_sets():
    """Return the label vocabulary of the phoneme targets.

    Returns
    -------
    dict of str to list of str
        ``phonemes``: the 41 CTC classes of the released decoder, in class-id
        order (``BLANK``, 39 ARPAbet phonemes, then the word separator `` | ``).
        The publisher's ``seq_class_ids`` already use these ids.

    Notes
    -----
        The upstream package also defines ``LOGIT_PHONE_DEF`` in
        ``nejm_b2txt_utils/general_utils.py`` with ``SIL`` at index 1; the
        released labels do not follow that order. Every stored sequence ends
        with id 40, and re-deriving ids from the sentence text with a
        grapheme-to-phoneme model reproduces about 99.8% of the stored
        phoneme positions under ``LOGIT_TO_PHONEME`` and none under
        ``LOGIT_PHONE_DEF``.
    """
    sys.path.insert(0, study_path(
        "20240815_card", "open_source_code_for_reference/nejm-brain-to-text/model_training"))
    from evaluate_model_helpers import LOGIT_TO_PHONEME
    return {"phonemes": list(LOGIT_TO_PHONEME)}


def _spec(source, key, session, partition, index, relative):
    """Build the trial record of one group of a publisher file.

    Parameters
    ----------
    source : h5py.File
        Open publisher file.
    key : str
        Group name such as ``trial_0007``.
    session : str
        Native session identifier such as ``t15.2023.08.11``.
    partition : str
        Publisher partition label, ``train`` or ``val``.
    index : int
        Zero-based position of the group in the file.
    relative : str
        Publisher file path relative to the study directory.

    Returns
    -------
    TrialSpec
        The trial with released features and, when labeled, its targets.

    Raises
    ------
    KeyError
        When the block description CSV has no row for the trial's block.
    ValueError
        When the group's ``session`` attribute disagrees with its directory.
    """
    group, attrs = source[key], source[key].attrs
    if attrs["session"] != session:
        raise ValueError(f"{relative}: {key} belongs to session {attrs['session']!r}, "
                         f"not to directory {session!r}")
    block = int(attrs["block_num"])
    try:
        corpus, post_implant_day = _blocks()[("-".join(session.split(".")[1:]), block)]
    except KeyError:
        raise KeyError(f"{relative}: no description row for block {block} of {session}") from None
    neural = {"input_features": Field(np.ascontiguousarray(group["input_features"][:]),
                                      bin_ms=BIN_MS, source_name="input_features", **FEATURE)}
    targets = {}
    if "seq_class_ids" in group:
        # the publisher pads both sequences with zeros up to 500 elements
        codes = group["transcription"][:]
        zeros = np.flatnonzero(codes == 0)
        targets = {
            "sentence": Field(str(attrs["sentence_label"]), source_name="sentence_label"),
            "phonemes": Field(np.ascontiguousarray(group["seq_class_ids"][:int(attrs["seq_len"])]),
                              label_set="phonemes", source_name="seq_class_ids",
                              description="Phoneme sequence of the sentence without zero padding."),
            "transcription": Field(np.ascontiguousarray(codes[:int(zeros[0])] if zeros.size else codes),
                                   source_name="transcription",
                                   description="Character codes of the sentence, "
                                               "everything before the first zero."),
        }
    meta = {"trial_num": int(attrs["trial_num"]), "corpus": corpus,
            "post_implant_day": post_implant_day}
    return TrialSpec(trial_id=trial_id(session, partition, index), source_file=relative,
                     source_index=index, neural=neural, source_partition=partition,
                     source_block=block, targets=targets, meta=meta)


def list_sessions():
    """List the recording days that have a publisher partition file.

    Returns
    -------
    list of dict
        One item per session with ``session``, ``participant``, and
        ``sources``, in date order.
    """
    sessions = []
    for directory in sorted(path for path in _data_dir().iterdir() if path.is_dir()):
        files = [_partition_path(directory.name, file_name) for _, file_name in PARTITIONS]
        present = [path for path in files if path.is_file()]
        if present:
            sessions.append({"session": directory.name,
                             "participant": directory.name.split(".")[0],
                             "sources": [_relative(path) for path in present]})
    return sessions


def extra_attrs(session):
    """Return the root attributes that record the withheld-label trials of a session.

    Parameters
    ----------
    session : str
        Native session identifier.

    Returns
    -------
    dict
        ``withheld_source_file`` and ``withheld_trials`` when the session has
        a ``data_test.hdf5``, else an empty mapping.
    """
    path = _partition_path(session, WITHHELD_FILE)
    if not path.is_file():
        return {}
    with h5py.File(path, "r") as source:
        return {"withheld_source_file": _relative(path), "withheld_trials": len(source.keys())}


def iter_trials(session):
    """Yield every labeled trial of one recording day in partition order.

    Parameters
    ----------
    session : str
        Native session identifier such as ``t15.2023.08.11``.

    Yields
    ------
    TrialSpec
        Trials of the ``train`` file, then of the ``val`` file, each in file
        order.

    Notes
    -----
    Trial keys run ``trial_0000`` onward without gaps in all 127 released
    files, so name order is the index the upstream trainer reads
    (``dataset.py``) and the prefix ``sorted(src.keys())[:n]`` that
    ``prep_rapid_calibration.py`` takes as the first trials of a day.
    """
    for partition, file_name in PARTITIONS:
        path = _partition_path(session, file_name)
        if not path.is_file():
            continue
        relative, (source, keys) = _relative(path), _open(path)
        for index, key in enumerate(keys):
            yield _spec(source, key, session, partition, index, relative)


def read_source_trial(attrs):
    """Rebuild one trial from the publisher file named in its attributes.

    Parameters
    ----------
    attrs : dict
        Attributes stored on an exported trial.

    Returns
    -------
    TrialSpec
        The trial as the publisher's file holds it.
    """
    relative = attrs["source_file"]
    source, keys = _open(Path(study_path(STUDY, relative)))
    session, index = Path(relative).parent.name, int(attrs["source_index"])
    return _spec(source, keys[index], session, attrs.get("source_partition", ""), index, relative)


if __name__ == "__main__":
    raise SystemExit(export_cli(
        STUDY, list_sessions, iter_trials, read_source_trial, label_sets, __file__,
        dataset=DATASET, source_format="hdf5", extra_attrs=extra_attrs,
        description="Export Card 2024 speech sessions to the common session format."))
