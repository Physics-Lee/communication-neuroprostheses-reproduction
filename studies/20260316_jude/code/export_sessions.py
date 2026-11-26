# -*- coding: utf-8 -*-
"""Export Jude 2026a touch-typing sessions to the common session format.

The publisher stores each block of a session as one MAT file under
``Data/<participant>/<session>/Typing/RedisMat/``: the whole block as
continuous 10 ms arrays (threshold crossings, spike-band power, NSP
timestamps) plus one row of per-trial timing variables per cue. A trial is
the slice from the go cue to the trial end in the neural time base, bounded
by the bins nearest to ``go_cue_nsp_neural_time`` and
``trial_end_nsp_neural_time`` and open at the end, exactly as the official
``mat_to_tfrecord.formatSessionData`` bounds it. One session file holds every
cue of every block, in block order then cue order, including the cues the
official converter skips (paused, timed out, empty cue), which carry the
publisher's flags in their attributes.

Features are stored as released. The view ``window_W4_convert.py`` feeds the
trainer (two 10 ms bins summed or averaged into one 20 ms frame, spike power
capped at 50000, block mean subtracted, divided by the global standard
deviation) is not applied; the attributes state how to rebuild it.

Usage (from the repository root):

    python studies/20260316_jude/code/export_sessions.py --list
    python studies/20260316_jude/code/export_sessions.py --session t17.2024.04.25
"""

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from session_format import Field, TrialSpec, export_cli, trial_id  # noqa: E402
from study_paths import study_checkout, study_path  # noqa: E402

STUDY = "20260316_jude"
DATASET = "https://doi.org/10.5061/dryad.cz8w9gjjk"
PARTICIPANTS = ("t17", "t18")
BIN_MS = 10
# Trainer input of window_W4_convert.py: bin_compression_factor 2 makes one frame of two bins.
PIPELINE_BIN_MS = 20
# Character classes of the released decoder, in the order of CHAR_DEF in the
# upstream mat_to_tfrecord.py. ">" replaces the space between two words.
CHAR_DEF = [">", ",", "?", ".", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
            "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
# Seed of the shuffle that assigns the test trials (main of mat_to_tfrecord.py).
SPLIT_SEED = 9
BLOCK_FILE = re.compile(r"\((\d+)\)\.mat$")
# common name, publisher variable, description
FEATURES = (
    ("spike_pow", "binned_neural_spike_band_power", "Spike-band power per electrode."),
    ("tx", "binned_neural_threshold_crossings", "Threshold-crossing counts per electrode."),
)
# trial attribute, publisher variable. The NSP neural clock is released in
# nanoseconds and the redis clock in milliseconds; both are stored as released.
TIMES = (
    ("go_cue_nsp_ns", "go_cue_nsp_neural_time"),
    ("trial_end_nsp_ns", "trial_end_nsp_neural_time"),
    ("trial_start_nsp_ns", "trial_start_nsp_neural_time"),
    ("go_cue_redis_ms", "go_cue_redis_time"),
    ("trial_end_redis_ms", "trial_end_redis_time"),
    ("trial_start_redis_ms", "trial_start_redis_time"),
)
VARIABLES = ("participant", "session_name", "block_number", "block_description", "graph_name",
             "cue", "binned_neural_nsp_timestamp", "trial_paused_by_CNRA", "trial_timed_out",
             "delay_duration_ms") + tuple(source for _, source, _ in FEATURES) + tuple(
                 source for _, source in TIMES)
_CACHE = {"key": None, "block": None}


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
        ``data/extracted/dryad_cz8w9gjjk/Data/t17/t17.2024.04.25/Typing/RedisMat/20240425_135520_(1).mat``.
    """
    return Path(path).absolute().relative_to(Path(study_checkout(STUDY)).absolute()).as_posix()


def _load(path, variables=VARIABLES):
    """Load the needed variables of one block MAT file, keeping the last file in memory.

    Parameters
    ----------
    path : pathlib.Path
        Publisher file.
    variables : tuple of str, optional
        MAT variables to read; the others are skipped without loading them.

    Returns
    -------
    dict
        The requested variables of the MAT file.

    Raises
    ------
    KeyError
        When the file lacks one of the requested variables.
    """
    key = (str(path), variables)
    if _CACHE["key"] != key:
        from scipy.io import loadmat
        block = loadmat(path, variable_names=list(variables))
        missing = [name for name in variables if name not in block]
        if missing:
            raise KeyError(f"{path}: missing variables {missing}")
        _CACHE.update(key=key, block=block)
    return _CACHE["block"]


def _text(value):
    """Return the first element of a publisher char or cell variable as text.

    Parameters
    ----------
    value : numpy.ndarray
        Element of a char matrix or cell array.

    Returns
    -------
    str
        The text, with the padding a char matrix adds left in place.
    """
    if isinstance(value, np.ndarray):
        value = value.ravel()[0]
    return str(value)


def _value(block, name, index):
    """Return one per-trial value of a publisher variable.

    Parameters
    ----------
    block : dict
        Loaded variables of one block MAT file.
    name : str
        Publisher variable holding one value per cue.
    index : int
        Zero-based cue position.

    Returns
    -------
    numpy scalar
        The value of that cue.

    Raises
    ------
    ValueError
        When the variable holds fewer values than the cue position asked for.
    """
    values = np.asarray(block[name]).ravel()
    if index >= len(values):
        raise ValueError(f"{name} holds {len(values)} trials but cue {index} was requested")
    return values[index]


def trial_bounds(block, index):
    """Bound one cue in the block's binned arrays, as the official converter does.

    Parameters
    ----------
    block : dict
        Loaded variables of one block MAT file.
    index : int
        Zero-based cue position.

    Returns
    -------
    tuple of int
        ``(onset, offset)`` row limits, half-open, of the 10 ms bins from the
        go cue to the trial end. Each limit is the bin whose NSP timestamp is
        nearest to the publisher's neural time of that event.

    Notes
    -----
    Same arithmetic as ``formatSessionData``: ``argmin`` of the absolute
    distance between ``binned_neural_nsp_timestamp`` and
    ``go_cue_nsp_neural_time`` / ``trial_end_nsp_neural_time``. No padding is
    added and the end bin is excluded.
    """
    stamps = np.asarray(block["binned_neural_nsp_timestamp"]).ravel()
    onset = np.argmin(np.abs(stamps - _value(block, "go_cue_nsp_neural_time", index)))
    offset = np.argmin(np.abs(stamps - _value(block, "trial_end_nsp_neural_time", index)))
    return int(onset), int(offset)


def char_targets(cue):
    """Derive the trainer's text targets from one cue.

    Parameters
    ----------
    cue : str
        Cue text as released.

    Returns
    -------
    transcription : str
        Cleaned transcription: everything but letters, spaces, ``? . , - '``
        removed, ``--`` and the space before an apostrophe dropped, lower
        case, stripped. The trainer's TFRecord stores it as ASCII codes in
        its ``transcription`` feature and it is the reference of the CER and
        WER evaluation.
    characters : str
        Upper-case character string of ``transcription`` with ``>`` in place
        of each space; only the 30 symbols of ``CHAR_DEF`` survive, so an
        apostrophe or a lone hyphen is dropped. Its positions in ``CHAR_DEF``
        are the ``char_class_ids`` target.

    Notes
    -----
    Verbatim port of ``convertToTFRecord`` of the upstream
    ``mat_to_tfrecord.py``. The upstream step that removes digits from each
    character is left out because the first substitution already removed
    them.
    """
    transcription = re.sub(r"[^a-zA-Z?.,\- ']", "", cue)
    transcription = transcription.replace("--", "").lower()
    transcription = transcription.replace(" '", "'").lower().strip()
    characters = []
    for char in transcription.upper():
        if char == " ":
            characters.append(">")
        if char in CHAR_DEF:
            characters.append(char)
    return transcription, "".join(characters)


def pipeline_split(count, num_test):
    """Divide trial positions into train and test, as the official converter does.

    Parameters
    ----------
    count : int
        Number of trials the converter kept in a session.
    num_test : int
        ``num_test_trials`` of the conversion.

    Returns
    -------
    tuple of numpy.ndarray
        ``(train, test)``: the positions in the order the converter writes
        them, the last ``num_test`` positions of a seeded shuffle being the
        test set.

    Notes
    -----
    ``main`` of the upstream ``mat_to_tfrecord.py`` seeds the legacy NumPy
    generator with 9 and shuffles ``arange(count)``; ``RandomState(9)`` draws
    from the same stream without touching the global generator.
    """
    order = np.arange(count)
    np.random.RandomState(SPLIT_SEED).shuffle(order)
    return order[:-num_test], order[-num_test:]


def session_blocks(session):
    """Return the block MAT files of one session in block order.

    Parameters
    ----------
    session : str
        Native session identifier such as ``t17.2024.04.25``.

    Returns
    -------
    list of tuple
        ``(block number, path)`` pairs. The block number is read from the
        ``(N)`` tag of the file name. When several files carry the same tag,
        the last one in file-name order is taken, which is the file the
        official converter selects.

    Notes
    -----
    The session is looked up under ``Data/<participant>/`` with the
    participant code its name starts with; the other participant's directory
    is searched only when that folder does not hold block files.
    """
    data = Path(study_path(STUDY, "data/extracted/Data"))
    participant = re.split(r"[._]", session, maxsplit=1)[0]
    for other in dict.fromkeys((participant,) + PARTICIPANTS):
        files = [path for path in sorted((data / other / session / "Typing" / "RedisMat").glob("*.mat"))
                 if BLOCK_FILE.search(path.name)]
        if files:
            found = {}
            for path in files:
                found.setdefault(int(BLOCK_FILE.search(path.name).group(1)), []).append(path)
            return [(number, group[-1]) for number, group in sorted(found.items())]
    return []


def _unit(number):
    """Return the trial-id unit of one block.

    Parameters
    ----------
    number : int
        Block number.

    Returns
    -------
    str
        Unit such as ``block05``.
    """
    return f"block{number:02d}"


def _spec(session, block, index, relative):
    """Build the trial record of one cue in a block MAT file.

    Parameters
    ----------
    session : str
        Native session identifier such as ``t17.2024.04.25``.
    block : dict
        Loaded variables of the block MAT file.
    index : int
        Zero-based cue position in the file.
    relative : str
        Publisher file path relative to the study directory.

    Returns
    -------
    TrialSpec
        The trial with its raw 10 ms features, its text targets, and the
        publisher's per-trial timing and flags.

    Raises
    ------
    ValueError
        When the trial end falls at or before the go cue, or a cue that is
        not empty cleans to an empty transcription.
    """
    number = int(np.asarray(block["block_number"]).ravel()[0])
    cue = _text(block["cue"][index])
    onset, offset = trial_bounds(block, index)
    if offset <= onset:
        raise ValueError(f"{trial_id(session, _unit(number), index)}: the trial end falls at or "
                         f"before the go cue (bins {onset}, {offset})")
    frames = offset - onset
    neural = {name: Field(np.ascontiguousarray(block[source][onset:offset]), bin_ms=BIN_MS,
                          source_name=source, description=description)
              for name, source, description in FEATURES}
    sentence = cue.strip()
    targets = {"sentence": Field(sentence, source_name="cue")}
    # The official converter drops a paused, timed-out, or empty cue before it derives any target.
    if sentence.split() and "DO_NOTHING" not in cue:
        transcription, characters = char_targets(cue)
        if not transcription:
            raise ValueError(f"{trial_id(session, _unit(number), index)}: cue {cue.strip()!r} cleans "
                             "to an empty transcription")
        targets["transcription"] = Field(
            transcription, source_name="cue",
            description="Cleaned transcription; the trainer's TFRecord holds it as ASCII codes.")
        targets["char_class_ids"] = Field(
            np.asarray([CHAR_DEF.index(char) for char in characters], dtype=np.int32),
            label_set="chars", source_name="cue",
            description="CTC label sequence, one id per character of the cleaned transcription "
                        "with > for each space. The TFRecord seqClassIDs hold the same ids plus "
                        "one, with 0 as padding.")
    stamps = np.asarray(block["binned_neural_nsp_timestamp"]).ravel()[onset:offset]
    meta = {"block_description": _text(block["block_description"]),
            "graph_name": _text(block["graph_name"]),
            "paused": int(_value(block, "trial_paused_by_CNRA", index)),
            "timed_out": int(_value(block, "trial_timed_out", index)),
            "delay_ms": int(_value(block, "delay_duration_ms", index)),
            # Trainer view: frames 0..n of the 20 ms compression, which drops a trailing odd bin.
            "pipeline_frame_onset": 0,
            "pipeline_frame_offset": frames // 2,
            "pipeline_bin_ms": PIPELINE_BIN_MS,
            # Largest step between two consecutive bins of the trial; 10 when the recording is continuous.
            "max_bin_step_ms": float(np.max(np.diff(stamps)) * 1e-6) if frames > 1 else 0.0}
    meta.update({name: float(_value(block, source, index)) for name, source in TIMES})
    return TrialSpec(trial_id=trial_id(session, _unit(number), index), source_file=relative,
                     source_index=index, neural=neural, source_block=number,
                     source_onset_frame=onset, source_offset_frame=offset, targets=targets, meta=meta)


def _check_header(session, block, number):
    """Check that a block file carries the session and block it was listed under.

    Parameters
    ----------
    session : str
        Native session identifier.
    block : dict
        Loaded variables of the block MAT file.
    number : int
        Block number read from the file name.

    Raises
    ------
    ValueError
        When the file's ``participant``, ``session_name``, or ``block_number``
        disagrees with the session name or the file-name tag.
    """
    participant = re.split(r"[._]", session, maxsplit=1)[0]
    stored = {"participant": _text(block["participant"]), "session_name": _text(block["session_name"]),
              "block_number": int(np.asarray(block["block_number"]).ravel()[0])}
    expected = {"participant": participant, "session_name": session, "block_number": number}
    for key in expected:
        if stored[key] != expected[key]:
            raise ValueError(f"{session} block {number}: {key} is {stored[key]!r} in the file")


def list_sessions():
    """List the typing sessions that have publisher block files.

    Returns
    -------
    list of dict
        One item per session with ``session``, ``participant``, and
        ``sources``, in participant then date order. Sessions without a
        ``Typing/RedisMat`` block file, such as the handwriting and
        brain-to-text days, are left out, and so is a folder filed under the
        other participant's code.
    """
    sessions = []
    data = Path(study_path(STUDY, "data/extracted/Data"))
    for participant in PARTICIPANTS:
        directory = data / participant
        if not directory.is_dir():
            continue
        for path in sorted(directory.glob("*")):
            if path.is_dir() and path.name.startswith(participant + ".") and session_blocks(path.name):
                sessions.append({"session": path.name, "participant": participant,
                                 "sources": [_relative(file) for _, file in session_blocks(path.name)]})
    return sessions


def iter_trials(session):
    """Yield every cue of one session, block by block in cue order.

    Parameters
    ----------
    session : str
        Native session identifier such as ``t17.2024.04.25``.

    Yields
    ------
    TrialSpec
        Trials in block order, then cue order. Paused, timed-out, and empty
        cues are yielded with the publisher's flags; a cue whose trial ends
        at or before its go cue has no bins to store and is reported on
        standard error instead.
    """
    for number, path in session_blocks(session):
        relative, block = _relative(path), _load(path)
        _check_header(session, block, number)
        for index in range(len(block["cue"])):
            onset, offset = trial_bounds(block, index)
            if offset <= onset:
                print(f"skip      {trial_id(session, _unit(number), index)}: the trial end falls at "
                      f"or before the go cue (bins {onset}, {offset})", file=sys.stderr)
                continue
            yield _spec(session, block, index, relative)


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
    session = attrs["trial_id"].split("/")[0]
    path = Path(study_path(STUDY, attrs["source_file"]))
    return _spec(session, _load(path), int(attrs["source_index"]), attrs["source_file"])


def label_sets():
    """Return the label vocabulary of the character targets.

    Returns
    -------
    dict of str to list of str
        ``chars``: the 30 character classes of the released decoder in class
        order (``>`` word separator, ``,``, ``?``, ``.``, then ``A``-``Z``).
    """
    return {"chars": list(CHAR_DEF)}


def extra_attrs(session):
    """Return the root attributes a session shares by its task and grid.

    Parameters
    ----------
    session : str
        Native session identifier.

    Returns
    -------
    dict
        ``participant`` as the publisher records it, ``task``, and the neural
        bin size in milliseconds.
    """
    block = _load(session_blocks(session)[0][1], ("participant",))
    return {"participant": _text(block["participant"]), "task": "typing",
            "neural_bin_size_ms": BIN_MS}


if __name__ == "__main__":
    raise SystemExit(export_cli(
        STUDY, list_sessions, iter_trials, read_source_trial, label_sets, __file__,
        dataset=DATASET, source_format="mat", extra_attrs=extra_attrs,
        description="Export Jude 2026a touch-typing sessions to the common session format."))
