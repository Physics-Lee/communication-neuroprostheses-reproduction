# -*- coding: utf-8 -*-
"""Export Kunz 2025 inner-speech sessions to the common session format.

The publisher stores one MAT file per recording session, condition
(attempted or imagined), and vocabulary. A file holds the whole session as
continuous ``spikePow`` and ``binnedTX`` arrays plus one row per trial in
``goTrialEpochs``. One session file holds every trial of one MAT file, in
the order of ``goTrialEpochs``. A trial is the go period, cut from the
continuous arrays and stored as released: source dtype, source bin size
(10, 20, or 33 ms), no per-block z-scoring, and none of the resampling to
20 ms that ``convert_kunz.py`` applies when it builds the trainer input.

``goTrialEpochs`` rows are 1-based inclusive ``[start, end]``, so a trial
is ``[start - 1, end)`` in 0-based half-open form and spans
``sentenceDurations[i] + 1`` frames. That range reproduces the publisher's
``sentenceFeatures`` cell of the same trial exactly in every T15 and T16
file (all 1,170 trials). The three T12 files are the exception: their
continuous arrays hold 128 electrodes, while their ``sentenceFeatures``
cells are an 8-bit copy with 256 columns per half, the electrodes in
another order, and the spike power rounded and clipped at 255.
``convert_kunz.py`` trains on ``sentenceFeatures``, so its T12 input cannot
be rebuilt from the session file alone; the file keeps the continuous arrays.

Closed-loop decoding is a property of a block: the publisher fills
``decodedSentenceText`` for every trial of a block that was decoded in real
time and leaves it empty otherwise. Files without the ``_eval`` suffix hold
such blocks too, so ``loop`` follows the decoded text rather than the file
name, and a trial carries ``decoded_sentence`` only when the publisher
stored one entry per trial and the entry is not empty.

Usage (from the repository root):

    python studies/20250821_kunz/code/export_sessions.py --list
    python studies/20250821_kunz/code/export_sessions.py --session t12.2023.11.28_attempted_50words
"""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from session_format import Field, TrialSpec, export_cli, trial_id  # noqa: E402
from study_paths import study_checkout, study_path  # noqa: E402

STUDY = "20250821_kunz"
DATASET = "https://doi.org/10.5061/dryad.gf1vhhn1j"
SOURCE_DIR = "data/extracted/sentenceData"
SESSION_NAME = re.compile(r"(?P<participant>t\d+)\.(?P<date>\d{4}\.\d{2}\.\d{2})_"
                          r"(?P<condition>attempted|imagined)_(?P<vocabulary>50words|125kwords)"
                          r"(?P<eval>_eval)?$")
VOCABULARY = {"50words": "50-word", "125kwords": "125k-word"}
# sentenceFeatures is not read: it repeats the go-period features and is the
# largest variable of the T15 and T16 files.
VARIABLES = ("spikePow", "binnedTX", "blockNum", "goTrialEpochs", "sentenceText",
             "decodedSentenceText", "sentenceDurations", "binSize")
_CACHE = {"path": None, "mat": None}


def _convert():
    """Import the study's conversion module, which also labels the trainer input.

    Returns
    -------
    module
        ``convert_kunz``, imported on first use because loading its
        grapheme-to-phoneme model takes several seconds.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import convert_kunz
    return convert_kunz


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
        ``data/extracted/dryad_gf1vhhn1j/sentenceData/t12.2023.11.28_attempted_50words.mat``.
    """
    return Path(path).absolute().relative_to(Path(study_checkout(STUDY)).absolute()).as_posix()


def _source_path(session):
    """Return the publisher file of one session.

    Parameters
    ----------
    session : str
        Native session identifier such as ``t12.2023.11.28_attempted_50words``.

    Returns
    -------
    pathlib.Path
        MAT file of the session.
    """
    return Path(study_path(STUDY, f"{SOURCE_DIR}/{session}.mat"))


def _parts(session):
    """Split a session identifier into the facts its file name encodes.

    Parameters
    ----------
    session : str
        Native session identifier.

    Returns
    -------
    dict
        ``participant``, ``date``, ``condition``, ``vocabulary``, and
        ``eval`` (``"_eval"`` for a closed-loop evaluation file, else ``None``).
    """
    match = SESSION_NAME.match(session)
    if match is None:
        raise ValueError(f"not a Kunz session identifier: {session}")
    return match.groupdict()


def _closed_blocks(mat):
    """Return the numbers of the blocks that were decoded in real time.

    Parameters
    ----------
    mat : dict
        Loaded publisher file (see ``_load``).

    Returns
    -------
    set of int
        Blocks with a decoded sentence in at least one row of
        ``decodedSentenceText``.

    Notes
    -----
    Rows are matched to trials by position. That is enough to name the block
    even in a file whose decoded column holds fewer entries than it has
    trials, where the decoded text itself cannot be attached to a trial.
    """
    blocks = np.asarray(mat["blockNum"]).ravel()
    return {int(blocks[int(epoch[0]) - 1])
            for epoch, text in zip(mat["goTrialEpochs"], mat["_decoded"]) if text.strip()}


def _load(path):
    """Load one publisher MAT file, keeping the last file in memory.

    Parameters
    ----------
    path : pathlib.Path
        Publisher file.

    Returns
    -------
    dict
        The variables listed in ``VARIABLES``, plus ``_decoded`` (the decoded
        sentence of each row of ``decodedSentenceText``) and
        ``_closed_blocks`` (see the function of that name).

    Raises
    ------
    ValueError
        When the per-trial variables of the file disagree on the number of
        trials.
    """
    if _CACHE["path"] != str(path):
        from scipy.io import loadmat
        mat = loadmat(path, variable_names=VARIABLES)
        counts = {name: mat[name].shape[0]
                  for name in ("sentenceText", "goTrialEpochs", "sentenceDurations")}
        if len(set(counts.values())) != 1:
            raise ValueError(f"{path}: per-trial variables disagree on the trial count {counts}")
        mat["_decoded"] = [_convert().to_str(cell) for cell in mat["decodedSentenceText"]]
        mat["_closed_blocks"] = _closed_blocks(mat)
        _CACHE.update(path=str(path), mat=mat)
    return _CACHE["mat"]


def label_sets():
    """Return the label vocabulary of the phoneme targets.

    Returns
    -------
    dict of str to list of str
        ``phonemes``: the 41 CTC classes of the released decoder, in class-id
        order (``BLANK``, 39 ARPAbet phonemes, then the word separator `` | ``).
    """
    return {"phonemes": list(_convert().LOGIT_TO_PHONEME)}


def phoneme_ids(sentence):
    """Convert a sentence to the decoder's CTC class ids.

    Parameters
    ----------
    sentence : str
        Sentence text.

    Returns
    -------
    list of int
        Class ids computed by ``convert_kunz.phoneme_ids``, the function that
        also labels the trainer input.
    """
    ids, _ = _convert().phoneme_ids(sentence)
    return list(ids)


def _bin_ms(mat):
    """Return the bin size a publisher file records.

    Parameters
    ----------
    mat : dict
        Loaded publisher file (see ``_load``).

    Returns
    -------
    int
        ``binSize`` of the file in milliseconds.
    """
    return int(np.asarray(mat["binSize"]).ravel()[0])


def _spec(mat, index, relative):
    """Build the trial record of one row of ``goTrialEpochs``.

    Parameters
    ----------
    mat : dict
        Loaded publisher file (see ``_load``).
    index : int
        Zero-based row of the trial in ``goTrialEpochs``.
    relative : str
        Publisher file path relative to the study directory.

    Returns
    -------
    TrialSpec
        The trial with its go-period features and sentence targets.
    """
    session = Path(relative).stem
    info = _parts(session)
    start, end = (int(value) for value in mat["goTrialEpochs"][index])
    onset, offset = start - 1, end  # the row is 1-based and inclusive
    block = int(np.asarray(mat["blockNum"]).ravel()[onset])
    text = _convert().to_str(mat["sentenceText"][index]).strip()
    bin_ms = _bin_ms(mat)
    neural = {
        "spike_pow": Field(np.ascontiguousarray(mat["spikePow"][onset:offset]), bin_ms=bin_ms,
                           source_name="spikePow",
                           description="Spike-band power per electrode."),
        "tx": Field(np.ascontiguousarray(mat["binnedTX"][onset:offset]), bin_ms=bin_ms,
                    source_name="binnedTX",
                    description="Threshold-crossing counts per electrode."),
    }
    targets = {
        "sentence": Field(text, source_name="sentenceText"),
        "phonemes": Field(np.asarray(phoneme_ids(text), dtype=np.int32), label_set="phonemes",
                          source_name="sentenceText",
                          description="CTC class ids from convert_kunz.phoneme_ids"),
    }
    decoded = mat["_decoded"]
    meta = {"condition": info["condition"],
            "loop": "closed" if info["eval"] or block in mat["_closed_blocks"] else "open",
            "sentence_duration_frames": int(mat["sentenceDurations"][index, 0])}
    if len(decoded) == mat["sentenceText"].shape[0] and decoded[index].strip():
        meta["decoded_sentence"] = decoded[index]
    return TrialSpec(trial_id=trial_id(session, "data", index), source_file=relative,
                     source_index=index, neural=neural, source_block=block,
                     source_onset_frame=onset, source_offset_frame=offset,
                     targets=targets, meta=meta)


def list_sessions():
    """List the session, condition, and vocabulary files the publisher holds.

    Returns
    -------
    list of dict
        One item per session with ``session``, ``participant``, and
        ``sources``, in file-name order.
    """
    sessions = []
    for path in sorted(Path(study_path(STUDY, SOURCE_DIR)).glob("*.mat")):
        if SESSION_NAME.match(path.stem):
            sessions.append({"session": path.stem, "participant": path.stem.split(".")[0],
                             "sources": [_relative(path)]})
    return sessions


def iter_trials(session):
    """Yield every trial of one publisher file in the order of ``goTrialEpochs``.

    Parameters
    ----------
    session : str
        Native session identifier such as ``t12.2023.11.28_attempted_50words``.

    Yields
    ------
    TrialSpec
        Trials in the order of the publisher's ``goTrialEpochs`` rows.
    """
    path = _source_path(session)
    relative, mat = _relative(path), _load(path)
    for index in range(mat["sentenceText"].shape[0]):
        yield _spec(mat, index, relative)


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
    relative = str(attrs["source_file"])
    return _spec(_load(Path(study_path(STUDY, relative))), int(attrs["source_index"]), relative)


def extra_attrs(session):
    """Return the root attributes a session shares by its recording file.

    Parameters
    ----------
    session : str
        Native session identifier.

    Returns
    -------
    dict
        ``condition``, ``neural_bin_size_ms``, and ``vocabulary`` of the file.
        When the publisher's ``decodedSentenceText`` does not hold one entry
        per trial, the trials carry no ``decoded_sentence`` and the file
        records ``decoded_sentence_entries`` and ``decoded_sentence_aligned``.
    """
    info = _parts(session)
    mat = _load(_source_path(session))
    attrs = {"condition": info["condition"], "neural_bin_size_ms": _bin_ms(mat),
             "vocabulary": VOCABULARY[info["vocabulary"]]}
    if len(mat["_decoded"]) != mat["sentenceText"].shape[0]:
        attrs.update(decoded_sentence_entries=len(mat["_decoded"]),
                     decoded_sentence_aligned=False)
    return attrs


if __name__ == "__main__":
    raise SystemExit(export_cli(
        STUDY, list_sessions, iter_trials, read_source_trial, label_sets, __file__,
        dataset=DATASET, source_format="mat", extra_attrs=extra_attrs,
        description="Export Kunz 2025 inner-speech sessions to the common session format."))
