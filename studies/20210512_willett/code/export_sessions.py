# -*- coding: utf-8 -*-
"""Export Willett 2021 handwriting sessions to the common session format.

One file holds every sentence of one recording day, in the order of the
publisher's ``sentences.mat``. The neural feature is the released
``neuralActivityCube`` row cut to ``numTimeBinsPerSentence``; the frames after
that point belong to the following trials. Each row equals the continuous
``neuralActivityTimeSeries`` starting one frame before the released (1-based)
``goCueOnsetTimeBin``, which the exporter checks before recording the range.

The Step2 HMM labels depend on the train/test partition, so every label
dataset exists once per released partition: unqualified names hold the
``HeldOutTrials`` labels and the ``_heldoutblocks`` suffix the ``HeldOutBlocks``
ones. Sentences the publisher did not align (excluded sentences, and the
from-memory and free-write trials of the two free-answer days) carry text
targets only.

Usage (from the repository root):

    python studies/20210512_willett/code/export_sessions.py --list
    python studies/20210512_willett/code/export_sessions.py --session t5.2019.05.08
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from session_format import Field, TrialSpec, export_cli, trial_id  # noqa: E402
from study_paths import study_checkout, study_path  # noqa: E402

STUDY = "20210512_willett"
DATASET = "https://doi.org/10.5061/dryad.wh70rxwmv"
BIN_MS = 10
# Partition of the unqualified label datasets, and every partition with labels.
LABEL_PARTITION = "HeldOutTrials"
PARTITIONS = ("HeldOutTrials", "HeldOutBlocks")
# Frames for which charStartTarget stays 1 after each letter start.
START_FRAMES = 21
NEURAL_DESCRIPTION = ("Binned threshold-crossing spike counts per electrode while T5 attempted "
                      "to write the sentence, as released. Channels 0-95 are array 1 and "
                      "channels 96-191 array 2.")
FRAME_DESCRIPTION = ("Character class of each frame: argmax of the released one-hot charProbTarget, "
                     "cut to the sentence length. Frames before the first character start repeat "
                     "the first character and are excluded from the RNN loss (see timing/char).")
CHAR_DESCRIPTION = "Class ids of the characters of sentence, one per row of timing/char."
TIMING_DESCRIPTION = ("Half-open extent of each character of sentence, in frames from the go cue, "
                      "as defined by the HMM letter starts: char_class_frame holds that "
                      "character on the whole row. charStartTarget is 1 on the first 21 frames "
                      "of a row, and ignoreErrorHere is 1 on the frames before the first row.")
_CACHE = {"session": None, "data": None}


def _root():
    """Return the publisher's directory of unprocessed and Step2 data.

    Returns
    -------
    pathlib.Path
        ``data/extracted/dryad_wh70rxwmv/handwritingBCIData`` of the study.
    """
    return Path(study_path(STUDY, "data/extracted/handwritingBCIData"))


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
        ``data/extracted/dryad_wh70rxwmv/handwritingBCIData/Datasets/t5.2019.05.08/sentences.mat``.
    """
    return Path(path).absolute().relative_to(Path(study_checkout(STUDY)).absolute()).as_posix()


def _sentences_file(session):
    """Return the publisher file of one session.

    Parameters
    ----------
    session : str
        Native session identifier such as ``t5.2019.05.08``.

    Returns
    -------
    pathlib.Path
        ``Datasets/<session>/sentences.mat``.
    """
    return _root() / "Datasets" / session / "sentences.mat"


def _labels_file(session, partition):
    """Return the publisher's Step2 label file of one session and partition.

    Parameters
    ----------
    session : str
        Native session identifier.
    partition : str
        ``HeldOutTrials`` or ``HeldOutBlocks``.

    Returns
    -------
    pathlib.Path
        ``RNNTrainingSteps/Step2_HMMLabels/<partition>/<session>_timeSeriesLabels.mat``.
    """
    return _root() / "RNNTrainingSteps" / "Step2_HMMLabels" / partition / f"{session}_timeSeriesLabels.mat"


def _suffix(partition):
    """Return the dataset-name suffix of one label partition.

    Parameters
    ----------
    partition : str
        ``HeldOutTrials`` or ``HeldOutBlocks``.

    Returns
    -------
    str
        Empty for ``LABEL_PARTITION``, otherwise ``_<partition in lower case>``.
    """
    return "" if partition == LABEL_PARTITION else "_" + partition.lower()


def label_sets():
    """Return the label vocabulary of the character targets.

    Returns
    -------
    dict of str to list of str
        ``chars``: the 31 handwriting characters in the class order of the
        pipeline's ``getHandwritingCharacterDefinitions``, which equals the
        order of the publisher's ``rnn_charMapASCII``.
    """
    sys.path.insert(0, str(Path(study_path(
        STUDY, "open_source_code_for_reference/handwritingBCI/characterDefinitions.py")).parent))
    from characterDefinitions import getHandwritingCharacterDefinitions
    return {"chars": list(getHandwritingCharacterDefinitions()["charListAbbr"])}


def _class_ids(text, chars):
    """Convert a label text to class ids.

    Parameters
    ----------
    text : str
        Label text without pause markers.
    chars : list of str
        The ``chars`` label set.

    Returns
    -------
    numpy.ndarray or None
        Int32 class ids, or None when the text holds a character outside the
        label set.
    """
    if any(character not in chars for character in text):
        return None
    return np.asarray([chars.index(character) for character in text], dtype=np.int32)


def _frame_classes(onsets, ids, frames):
    """Rebuild the per-frame classes from the letter starts.

    Parameters
    ----------
    onsets : numpy.ndarray
        Letter starts of one sentence, in frames.
    ids : numpy.ndarray
        Class id of each character.
    frames : int
        Number of frames of the sentence.

    Returns
    -------
    numpy.ndarray
        The class of each frame under the rule of the pipeline's
        ``constructRNNTargets``: a character holds from its start to the next
        one, and the frames before the first start take the first character.
    """
    rows = np.searchsorted(onsets, np.arange(frames), side="right") - 1
    return ids[np.maximum(rows, 0)]


def _onsets(starts, count):
    """Return the letter starts of one sentence.

    Parameters
    ----------
    starts : numpy.ndarray
        One row of ``letterStarts``.
    count : int
        Number of characters of the label text.

    Returns
    -------
    numpy.ndarray or None
        Int64 starts, or None when the publisher left the sentence unaligned
        (all starts zero).

    Notes
    -----
    The letter starts are the only safe test: on zero starts the pipeline's
    ``constructRNNTargets`` fills the unaligned sentence's ``charProbTarget``
    with its last character and puts a start pulse on frames 0-20, so those
    targets must not be exported as labels.
    """
    values = starts[:count]
    if not values.any():
        return None
    if not np.array_equal(values, np.round(values)):
        raise ValueError("letter starts are not whole frames")
    return values.astype(np.int64)


def _check_labels(labels, loaded, partition, chars):
    """Check that the released targets follow from the letter starts.

    Parameters
    ----------
    labels : dict
        Variables of one Step2 label file.
    loaded : dict
        Per-sentence arrays of the session (see ``_load``).
    partition : str
        Label partition, for error messages.
    chars : list of str
        The ``chars`` label set.

    Raises
    ------
    ValueError
        When a sentence's frame classes, start signal, or ignore mask differ
        from what ``constructRNNTargets`` derives from its letter starts.

    Notes
    -----
    The dataset descriptions state these rules, so they must hold for every
    session that is written.
    """
    for index, text in enumerate(loaded["text"]):
        onsets = _onsets(labels["letterStarts"][index], len(text))
        if onsets is None:
            continue
        frames = int(loaded["bins"][index])
        ids = _class_ids(text, chars)
        where = f"{partition} sentence {index}"
        if ids is None or len(onsets) != len(ids) or onsets[-1] >= frames:
            raise ValueError(f"{where}: the alignment does not fit the label text")
        if not np.array_equal(labels["classes"][index, :frames], _frame_classes(onsets, ids, frames)):
            raise ValueError(f"{where}: charProbTarget differs from the letter starts")
        start = np.zeros(frames, dtype=np.float32)
        for onset in onsets:
            start[onset:onset + START_FRAMES] = 1
        ignore = np.zeros(frames, dtype=np.float32)
        ignore[:onsets[0]] = 1
        if not (np.array_equal(labels["charStartTarget"][index, :frames], start)
                and np.array_equal(labels["ignoreErrorHere"][index, :frames], ignore)):
            raise ValueError(f"{where}: charStartTarget or ignoreErrorHere differs from the letter starts")


def _load(session):
    """Load one session with its Step2 labels, keeping the last session in memory.

    Parameters
    ----------
    session : str
        Native session identifier.

    Returns
    -------
    dict
        Per-sentence arrays with keys ``source`` (relative path of
        ``sentences.mat``), ``cube``, ``series``, ``bins``, ``text``
        (prompts without pause markers), ``prompt``, ``intended``,
        ``condition``, ``block``, ``excluded``, ``go``, ``delay``, ``end``
        (the released 1-based time bins), ``char_map``, ``chars`` (the label
        set), and ``labels``. The last maps each partition to its label file
        path, the ``letterStarts`` matrix, and the frame classes.
    """
    if _CACHE["session"] != session:
        import scipy.io
        path = _sentences_file(session)
        sentences = scipy.io.loadmat(path, squeeze_me=True)
        chars = label_sets()["chars"]
        char_map = str(sentences["rnn_charMapASCII"])
        if char_map != "".join(chars):
            raise ValueError(f"{session}: rnn_charMapASCII differs from the pipeline's character list")
        prompt = [str(value) for value in np.atleast_1d(sentences["sentencePrompt"])]
        loaded = {
            "source": _relative(path),
            "cube": sentences["neuralActivityCube"],
            "series": sentences["neuralActivityTimeSeries"],
            "bins": np.asarray(sentences["numTimeBinsPerSentence"], dtype=np.int64).ravel(),
            "text": [value.replace("#", "") for value in prompt],
            "prompt": prompt,
            "intended": [str(value) for value in np.atleast_1d(sentences["intendedText"])],
            "condition": [str(value) for value in np.atleast_1d(sentences["sentenceCondition"])],
            "block": np.asarray(sentences["sentenceBlockNums"], dtype=np.int64).ravel(),
            "excluded": np.asarray(sentences["excludedSentences"], dtype=np.int64).ravel() == 1,
            "go": np.asarray(sentences["goCueOnsetTimeBin"], dtype=np.int64).ravel(),
            "delay": np.asarray(sentences["delayCueOnsetTimeBin"], dtype=np.int64).ravel(),
            "end": np.asarray(sentences["sentenceEndTimeBin"], dtype=np.int64).ravel(),
            "char_map": char_map,
            "chars": chars,
            "labels": {},
        }
        for partition in PARTITIONS:
            file = _labels_file(session, partition)
            if not file.is_file():
                continue
            labels = scipy.io.loadmat(file)
            if not np.array_equal(labels["timeBinsPerSentence"].ravel(), loaded["bins"]):
                raise ValueError(f"{session}: {partition} labels disagree with numTimeBinsPerSentence")
            labels["classes"] = labels["charProbTarget"].argmax(axis=2).astype(np.int32)
            _check_labels(labels, loaded, partition, chars)
            loaded["labels"][partition] = {"file": _relative(file), "starts": labels["letterStarts"],
                                           "classes": labels["classes"]}
        _CACHE.update(session=session, data=loaded)
    return _CACHE["data"]


def _spec(loaded, index, session):
    """Build the trial record of one sentence.

    Parameters
    ----------
    loaded : dict
        Per-sentence arrays of the session (see ``_load``).
    index : int
        Zero-based sentence position in ``sentences.mat``.
    session : str
        Native session identifier.

    Returns
    -------
    TrialSpec
        The sentence with its released neural activity, texts, and the HMM
        labels of every partition that aligned it.
    """
    source, text = loaded["source"], loaded["text"][index]
    frames = int(loaded["bins"][index])
    onset = int(loaded["go"][index]) - 1
    if not np.array_equal(loaded["cube"][index, :frames], loaded["series"][onset:onset + frames]):
        raise ValueError(f"{session} sentence {index}: neuralActivityCube is not neuralActivityTimeSeries "
                         "at goCueOnsetTimeBin - 1")
    ids = _class_ids(text, loaded["chars"])
    neural = {"neural_activity": Field(np.ascontiguousarray(loaded["cube"][index, :frames]), bin_ms=BIN_MS,
                                       source_name="neuralActivityCube", description=NEURAL_DESCRIPTION)}
    targets = {
        "sentence": Field(text, source_name="sentencePrompt",
                          description="Text the pipeline aligns, trains on, and scores: sentencePrompt "
                                      "without the # pause markers. > marks a space and ~ a period. "
                                      "A free-write trial holds its question here."),
        "prompt": Field(loaded["prompt"][index], source_name="sentencePrompt",
                        description="sentencePrompt as released, including the # pause markers."),
        "intended_text": Field(loaded["intended"][index], source_name="intendedText",
                               description="intendedText as released: the prompt without # markers, or "
                                           "T5's intended answer to a free-write question."),
    }
    if ids is not None:
        targets["char_class"] = Field(ids, label_set="chars", source_name="sentencePrompt",
                                      description=CHAR_DESCRIPTION)
    timing, labeled = {}, False
    for partition, labels in loaded["labels"].items():
        onsets = _onsets(labels["starts"][index], len(text))
        if onsets is None:
            continue
        attrs = {"bin_ms": BIN_MS, "source_file": labels["file"], "label_partition": partition}
        suffix = _suffix(partition)
        targets["char_class_frame" + suffix] = Field(labels["classes"][index, :frames], label_set="chars",
                                                     source_name="charProbTarget",
                                                     description=FRAME_DESCRIPTION, **attrs)
        timing["char" + suffix] = Field(np.stack([onsets, np.append(onsets[1:], frames)], axis=1).astype(np.int32),
                                        source_name="letterStarts", description=TIMING_DESCRIPTION, **attrs)
        labeled = labeled or partition == LABEL_PARTITION
    meta = {"sentence_condition": loaded["condition"][index], "excluded": bool(loaded["excluded"][index]),
            "go_cue_onset_time_bin": int(loaded["go"][index]),
            "delay_cue_onset_time_bin": int(loaded["delay"][index]),
            "sentence_end_time_bin": int(loaded["end"][index]), "labeled": labeled}
    return TrialSpec(trial_id=trial_id(session, "data", index), source_file=source, source_index=index,
                     neural=neural, source_block=int(loaded["block"][index]),
                     source_onset_frame=onset, source_offset_frame=onset + frames,
                     targets=targets, timing=timing, meta=meta)


def list_sessions():
    """List the recording days that have a publisher ``sentences.mat``.

    Returns
    -------
    list of dict
        One item per session with ``session``, ``participant``, and
        ``sources``, in date order.
    """
    sessions = []
    for directory in sorted((_root() / "Datasets").iterdir()):
        path = directory / "sentences.mat"
        if path.is_file():
            sources = [_relative(path)] + [_relative(_labels_file(directory.name, partition))
                                           for partition in PARTITIONS
                                           if _labels_file(directory.name, partition).is_file()]
            sessions.append({"session": directory.name, "participant": directory.name.split(".")[0],
                             "sources": sources})
    return sessions


def extra_attrs(session):
    """Return the root attributes a session shares by its recording day.

    Parameters
    ----------
    session : str
        Native session identifier.

    Returns
    -------
    dict
        ``neural_bin_size_ms``, ``label_partition`` (the partition behind the
        unqualified label datasets), and the released ``rnn_charMapASCII``.
    """
    return {"neural_bin_size_ms": BIN_MS, "label_partition": LABEL_PARTITION,
            "rnn_char_map_ascii": _load(session)["char_map"]}


def iter_trials(session):
    """Yield every sentence of one recording day in file order.

    Parameters
    ----------
    session : str
        Native session identifier such as ``t5.2019.05.08``.

    Yields
    ------
    TrialSpec
        Trials in the order of the publisher's per-sentence arrays.
    """
    loaded = _load(session)
    for index in range(len(loaded["text"])):
        yield _spec(loaded, index, session)


def read_source_trial(attrs):
    """Rebuild one trial from the publisher file named in its attributes.

    Parameters
    ----------
    attrs : dict
        Attributes stored on an exported trial.

    Returns
    -------
    TrialSpec
        The trial as the publisher's files hold it.
    """
    source = attrs["source_file"]
    session = Path(source).parent.name
    return _spec(_load(session), int(attrs["source_index"]), session)


if __name__ == "__main__":
    raise SystemExit(export_cli(
        STUDY, list_sessions, iter_trials, read_source_trial, label_sets, __file__,
        dataset=DATASET, source_format="mat", extra_attrs=extra_attrs,
        description="Export Willett 2021 handwriting sessions to the common session format."))
