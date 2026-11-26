# -*- coding: utf-8 -*-
"""Export Willett 2023 competition sessions to the common session format.

One file holds every trial of one recording day in the publisher's partition
order ``train``, ``test``, ``competitionHoldOut``. Features are stored as
released, without the block-wise z-scoring that ``convert_btot24.py`` applies
when it builds the trainer input. The release redacts the
``competitionHoldOut`` transcriptions, so those trials carry no sentence or
phoneme targets.

Usage (from the repository root):

    python studies/20230831_willett/code/export_sessions.py --list
    python studies/20230831_willett/code/export_sessions.py --session t12.2022.05.19
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from session_format import Field, TrialSpec, export_cli, trial_id  # noqa: E402
from study_paths import study_checkout, study_path  # noqa: E402

STUDY = "20230831_willett"
DATASET = "https://doi.org/10.5061/dryad.x69p8czpq"
PARTITIONS = ("train", "test", "competitionHoldOut")
BIN_MS = 20
# common name, publisher variable, description
FEATURES = (
    ("spike_pow", "spikePow", "Spike-band power per electrode."),
    ("tx", "tx1", "Threshold-crossing counts per electrode."),
    ("tx2", "tx2", "Auxiliary threshold-crossing feature released with the competition data."),
    ("tx3", "tx3", "Auxiliary threshold-crossing feature released with the competition data."),
    ("tx4", "tx4", "Auxiliary threshold-crossing feature released with the competition data."),
)
_CACHE = {"path": None, "mat": None}


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
        ``data/extracted/dryad_x69p8czpq/competitionData/test/t12.2022.05.19.mat``.
    """
    return Path(path).absolute().relative_to(Path(study_checkout(STUDY)).absolute()).as_posix()


def _load(path):
    """Load one publisher MAT file, keeping the last file in memory.

    Parameters
    ----------
    path : pathlib.Path
        Publisher file.

    Returns
    -------
    dict
        Variables of the MAT file.
    """
    if _CACHE["path"] != str(path):
        from scipy.io import loadmat
        _CACHE.update(path=str(path), mat=loadmat(path))
    return _CACHE["mat"]


def _partition_dir(partition):
    """Return the directory of one publisher partition.

    Parameters
    ----------
    partition : str
        ``train``, ``test``, or ``competitionHoldOut``.

    Returns
    -------
    pathlib.Path
        Directory holding the per-day MAT files of the partition.
    """
    return Path(study_path(STUDY, f"data/extracted/competitionData/{partition}"))


def label_sets():
    """Return the label vocabulary of the phoneme targets.

    Returns
    -------
    dict of str to list of str
        ``phonemes``: the 41 CTC classes of the released decoder, in class-id
        order (``BLANK``, 39 ARPAbet phonemes, then the word separator `` | ``).
    """
    sys.path.insert(0, study_path(
        "20240815_card", "open_source_code_for_reference/nejm-brain-to-text/model_training"))
    from evaluate_model_helpers import LOGIT_TO_PHONEME
    return {"phonemes": list(LOGIT_TO_PHONEME)}


def phoneme_ids(sentence):
    """Convert a sentence to the decoder's CTC class ids.

    Parameters
    ----------
    sentence : str
        Sentence text.

    Returns
    -------
    list of int
        Class ids computed by ``convert_btot24.phoneme_ids``, the function that
        also labels the trainer input.

    Notes
    -----
    The conversion module is imported on first use because loading its
    grapheme-to-phoneme model takes a few seconds.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import convert_btot24
    ids, _ = convert_btot24.phoneme_ids(sentence)
    return list(ids)


def _spec(mat, index, partition, relative):
    """Build the trial record of one sentence in a publisher file.

    Parameters
    ----------
    mat : dict
        Loaded publisher file.
    index : int
        Zero-based trial position in the file.
    partition : str
        Publisher partition label.
    relative : str
        Publisher file path relative to the study directory.

    Returns
    -------
    TrialSpec
        The trial with released features and, when labeled, its targets.
    """
    session = Path(relative).stem
    text = str(mat["sentenceText"][index]).strip()
    neural = {name: Field(np.ascontiguousarray(mat[source][0, index]), bin_ms=BIN_MS,
                          source_name=source, description=description)
              for name, source, description in FEATURES}
    targets, meta = {}, {}
    if partition == "competitionHoldOut":
        meta["source_text"] = text
    else:
        targets = {
            "sentence": Field(text, source_name="sentenceText"),
            "phonemes": Field(np.asarray(phoneme_ids(text), dtype=np.int32), label_set="phonemes",
                              source_name="sentenceText", description="CTC class ids from sentence_to_phonemes"),
            "transcription": Field(np.asarray([ord(char) for char in text], dtype=np.int32),
                                   source_name="sentenceText", description="Character codes of the sentence"),
        }
    return TrialSpec(trial_id=trial_id(session, partition, index), source_file=relative,
                     source_index=index, neural=neural, source_partition=partition,
                     source_block=int(np.asarray(mat["blockIdx"][index]).ravel()[0]),
                     targets=targets, meta=meta)


def list_sessions():
    """List the recording days that have a publisher file.

    Returns
    -------
    list of dict
        One item per session with ``session``, ``participant``, and
        ``sources``, in date order.
    """
    sessions = {}
    for partition in PARTITIONS:
        directory = _partition_dir(partition)
        if directory.is_dir():
            for path in sorted(directory.glob("*.mat")):
                item = sessions.setdefault(path.stem, {"session": path.stem,
                                                       "participant": path.stem.split(".")[0],
                                                       "sources": []})
                item["sources"].append(_relative(path))
    return [sessions[name] for name in sorted(sessions)]


def iter_trials(session):
    """Yield every trial of one recording day in partition order.

    Parameters
    ----------
    session : str
        Native session identifier such as ``t12.2022.05.19``.

    Yields
    ------
    TrialSpec
        Trials of the ``train``, then ``test``, then ``competitionHoldOut``
        file of the day, each in file order.
    """
    for partition in PARTITIONS:
        path = _partition_dir(partition) / f"{session}.mat"
        if not path.is_file():
            continue
        relative, mat = _relative(path), _load(path)
        for index in range(mat["sentenceText"].shape[0]):
            yield _spec(mat, index, partition, relative)


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
    path = Path(study_path(STUDY, attrs["source_file"]))
    return _spec(_load(path), int(attrs["source_index"]), attrs.get("source_partition", ""),
                 attrs["source_file"])


if __name__ == "__main__":
    raise SystemExit(export_cli(
        STUDY, list_sessions, iter_trials, read_source_trial, label_sets, __file__,
        dataset=DATASET, source_format="mat",
        description="Export Willett 2023 competition sessions to the common session format."))
