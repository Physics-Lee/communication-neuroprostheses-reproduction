# -*- coding: utf-8 -*-
"""Export Jude 2026b intended-speech sessions (T17) to the common session format.

The publisher stores each block of a session as one MAT file under
``Data/t17/<session>/brainToText/RedisMat/``: the whole block as continuous
10 ms arrays (threshold crossings, spike-band power, NSP timestamps) plus one
row of per-trial timing variables per cue. A trial is the slice from the go
cue to the trial end in the neural time base, carved as ``convert_jude.py``
carves it. It starts at the bin whose NSP timestamp is nearest to
``go_cue_nsp_neural_time`` and ends before the bin nearest to
``trial_end_nsp_neural_time``, one bin earlier when that window has an odd
length, because the pipeline compresses pairs of bins into one 20 ms frame
and drops the leftover. One session file holds every trial of one recording
day, in block order then cue order.

Only the sentence blocks ``convert_jude.py`` converts are exported. It picks
them by block number from its ``SENTENCE_BLOCKS`` table, which was copied from
the dataset README. ``block_description`` and ``graph_name`` do not tell a
sentence block from a single-word, phoneme, or orofacial block: every
brain-to-text block carries the same generic description and an open- or
closed-loop ``BrainToText`` graph. A block outside the table is not exported,
and neither is a cue the pipeline rejects (paused, timed out, shorter than two
bins, or with no more than four 20 ms frames per phoneme plus ten). Excluded
sessions, which ``SENTENCE_BLOCKS`` does not list, are named in
``EXCLUDED_SESSIONS`` with their reason: 03.25 (isolated phonemes and
orofacial movements) and 05.23 (reading task of the movement sweep).

Features are stored as released at 10 ms with the released channel count, 384
or 256 on 04.15 to 04.17. The trainer view of ``convert_jude.py`` (pairs of
bins summed or averaged into one 20 ms frame, per-block z-scoring, zero
padding to 768 features) is not applied.

Usage (from the repository root):

    python studies/20260428_jude/code/export_sessions.py --list
    python studies/20260428_jude/code/export_sessions.py --session t17.2024.03.26
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from session_format import Field, TrialSpec, export_cli, trial_id  # noqa: E402
from study_paths import study_checkout, study_path  # noqa: E402

STUDY = "20260428_jude"
DATASET = "https://doi.org/10.5061/dryad.vq83bk481"
BIN_MS = 10
# Trainer input of convert_jude.py: bin_compression_factor 2 makes one frame of two bins.
PIPELINE_BIN_MS = 20
# common name, publisher variable, description
FEATURES = (
    ("spike_pow", "binned_neural_spike_band_power", "Spike-band power per electrode."),
    ("tx", "binned_neural_threshold_crossings", "Threshold-crossing counts per electrode."),
)
# Variables that name a block, and every variable a trial is carved and labeled from.
HEADER = ("block_number", "graph_name")
VARIABLES = ("participant", "session_name", "block_number", "block_description", "graph_name",
             "cue", "binned_neural_nsp_timestamp", "go_cue_nsp_neural_time",
             "trial_end_nsp_neural_time", "trial_paused_by_CNRA", "trial_timed_out",
             "delay_duration_ms") + tuple(source for _, source, _ in FEATURES)
# Session folders under Data/t17 that convert_jude.py leaves out, with the reason.
EXCLUDED_SESSIONS = {
    "t17.2024.03.25": "convert_jude.py SENTENCE_BLOCKS has no entry for it: its 14 brain-to-text "
                      "blocks cue isolated phonemes and orofacial movements (80 single-word cues "
                      "in blocks 5-13, 34 orofacial instructions in blocks 14-18), no sentences",
    "t17.2024.05.23": "convert_jude.py SENTENCE_BLOCKS has no entry for it: reading task of the "
                      "movement sweep (T17_ReadingTask.yaml) between DO_NOTHING diagnostic blocks",
}
# Reason a block file is not converted. convert_jude.py selects blocks by number only;
# the graph stored in the file names the task the block ran.
GRAPH_SKIP_REASONS = {
    "Diagnostic.yaml": "diagnostic block (Diagnostic.yaml)",
    "T17_ReadingTask.yaml": "reading task of the movement sweep (T17_ReadingTask.yaml)",
}
UNLISTED_SKIP_REASON = "block whose number is not in SENTENCE_BLOCKS of convert_jude.py"
_CACHE = {"key": None, "block": None}


def _pipeline():
    """Import the conversion script that defines the trials of this study.

    Returns
    -------
    module
        ``convert_jude`` of this directory, source of the sentence-block table
        and of the phoneme ids of a sentence.

    Notes
    -----
    Imported on first use because loading its grapheme-to-phoneme model takes
    a few seconds.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import convert_jude
    return convert_jude


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
        ``data/extracted/dryad_vq83bk481/Data/t17/t17.2024.03.26/brainToText/RedisMat/20240326_162146_(23).mat``.
    """
    return Path(path).resolve().relative_to(Path(study_checkout(STUDY)).resolve()).as_posix()


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
        The stored text.
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


def trial_bounds(block, index):
    """Bound one cue in the block's binned arrays, as ``convert_jude.py`` does.

    Parameters
    ----------
    block : dict
        Loaded variables of one block MAT file.
    index : int
        Zero-based cue position.

    Returns
    -------
    tuple of int
        ``(onset, offset, nearest)`` row limits of the 10 ms bins from the go
        cue to the trial end, the first two half-open. Each limit is the bin
        whose NSP timestamp is nearest to the publisher's neural time of that
        event. ``offset`` leaves out the last bin when ``nearest - onset`` is
        odd; ``nearest`` is the limit before that trim.

    Notes
    -----
    Same arithmetic as ``convert_block``: ``argmin`` of the absolute distance
    between ``binned_neural_nsp_timestamp`` and ``go_cue_nsp_neural_time`` or
    ``trial_end_nsp_neural_time``, then ``((e_idx - s_idx) // 2) * 2`` bins.
    The window starts at the go cue, after the cued delay, and no padding is
    added. The indexes are zero-based positions in the binned arrays.
    """
    stamps = np.asarray(block["binned_neural_nsp_timestamp"]).ravel().astype(np.float64)
    onset = int(np.argmin(np.abs(stamps - _value(block, "go_cue_nsp_neural_time", index))))
    nearest = int(np.argmin(np.abs(stamps - _value(block, "trial_end_nsp_neural_time", index))))
    return onset, onset + (nearest - onset) // 2 * 2, nearest


def _block_files(session):
    """Return the block MAT files of one session keyed by block number.

    Parameters
    ----------
    session : str
        Native session identifier such as ``t17.2024.03.26``.

    Returns
    -------
    dict of int to pathlib.Path
        Block number and file of every block, in file-name order.

    Raises
    ------
    ValueError
        When two files of the session store the same block number.

    Notes
    -----
    Only ``block_number`` and ``graph_name`` are read. ``main`` of
    ``convert_jude.py`` maps block numbers to files the same way and drops a
    file it cannot read; such a file is reported on standard error and left
    out here as well.
    """
    participant = session.split(".")[0]
    directory = Path(study_path(STUDY, "data/extracted/Data")) / participant / session
    files = {}
    for path in sorted((directory / "brainToText" / "RedisMat").glob("*.mat")):
        try:
            number = int(np.asarray(_load(path, HEADER)["block_number"]).ravel()[0])
        except (OSError, ValueError) as exc:
            print(f"unreadable (skipped): {_relative(path)} ({exc})", file=sys.stderr)
            continue
        if number in files:
            raise ValueError(f"{session}: block {number} is stored by {files[number].name} and {path.name}")
        files[number] = path
    return files


def sentence_blocks(session):
    """Return the sentence blocks ``convert_jude.py`` converts, in block order.

    Parameters
    ----------
    session : str
        Native session identifier such as ``t17.2024.03.26``.

    Returns
    -------
    list of tuple
        ``(block number, path)`` of every block of ``SENTENCE_BLOCKS[session]``
        that has a block file, lowest block number first.

    Notes
    -----
    A block of the table without a file is reported on standard error, as
    ``main`` of ``convert_jude.py`` warns about it.
    """
    table = _pipeline().SENTENCE_BLOCKS[session]
    files = _block_files(session)
    missing = sorted(set(table) - set(files))
    if missing:
        print(f"{session}: WARNING missing blocks {missing}", file=sys.stderr)
    return [(number, files[number]) for number in sorted(files) if number in table]


def skipped_blocks(session):
    """Return the block files of one session that ``convert_jude.py`` does not convert.

    Parameters
    ----------
    session : str
        Native session identifier such as ``t17.2024.03.26``.

    Returns
    -------
    list of dict
        ``source_file``, ``block``, ``graph_name``, and ``reason`` of every
        block outside ``SENTENCE_BLOCKS``, in block order.
    """
    converted = _pipeline().SENTENCE_BLOCKS.get(session, {})
    skipped = []
    for number, path in sorted(_block_files(session).items()):
        if number in converted:
            continue
        graph = _text(_load(path, HEADER)["graph_name"])
        reason = GRAPH_SKIP_REASONS.get(graph, UNLISTED_SKIP_REASON)
        skipped.append({"source_file": _relative(path), "block": number,
                        "graph_name": graph, "reason": reason})
    return skipped


def _spec(session, block, index, relative, required=False):
    """Build the trial record of one cue in a sentence block.

    Parameters
    ----------
    session : str
        Native session identifier such as ``t17.2024.03.26``.
    block : dict
        Loaded variables of the block MAT file.
    index : int
        Zero-based cue position in the file.
    relative : str
        Publisher file path relative to the study directory.
    required : bool, optional
        Raise for a cue the pipeline rejects instead of returning None.

    Returns
    -------
    TrialSpec or None
        The trial with its raw 10 ms features, its phoneme targets, and the
        publisher's per-trial flags, or None for a rejected cue.

    Raises
    ------
    ValueError
        When the file is stored under another participant or session, its
        block is not a sentence block, or ``required`` is set for a cue the
        pipeline rejects.
    """
    pipeline = _pipeline()
    number = int(np.asarray(block["block_number"]).ravel()[0])
    where = trial_id(session, _unit(number), index)
    stored = (_text(block["participant"]), _text(block["session_name"]))
    if stored != (session.split(".")[0], session):
        raise ValueError(f"{where}: the file is stored as {stored[0]}/{stored[1]}")
    if number not in pipeline.SENTENCE_BLOCKS.get(session, {}):
        raise ValueError(f"{where}: block {number} is not a sentence block of convert_jude.py")
    vocab, loop = pipeline.SENTENCE_BLOCKS[session][number]
    onset, offset, nearest = trial_bounds(block, index)
    paused = int(_value(block, "trial_paused_by_CNRA", index))
    timed_out = int(_value(block, "trial_timed_out", index))
    reason = ""
    if paused or timed_out:
        reason = "paused by the clinical neural recorder assistant or timed out"
    elif nearest - onset < 2:
        reason = "shorter than two bins"
    sentence = str(block["cue"][index]).strip()
    neural, ids = {}, []
    if not reason:
        # convert_block labels the sentence before it applies its length filter.
        ids, _ = pipeline.phoneme_ids(sentence)
        neural = {name: Field(np.ascontiguousarray(block[source][onset:offset]), bin_ms=BIN_MS,
                              source_name=source, description=description)
                  for name, source, description in FEATURES}
        # Spike power is the only feature that can fail the pipeline's finite check.
        if ((offset - onset) // 2 <= 4 * len(ids) + 10
                or not np.isfinite(neural["spike_pow"].data).all()):
            reason = "no more than four 20 ms frames per phoneme plus ten, or not finite"
    if reason:
        if required:
            raise ValueError(f"{where}: convert_jude.py rejects the cue: {reason}")
        print(f"skip      {where}: {reason}", file=sys.stderr)
        return None
    targets = {
        "sentence": Field(sentence, source_name="cue"),
        "phonemes": Field(np.asarray(ids, dtype=np.int32), label_set="phonemes",
                          source_name="cue", description="CTC class ids from convert_jude.phoneme_ids"),
        "transcription": Field(np.asarray([ord(char) for char in sentence], dtype=np.int32),
                               source_name="cue", description="Character codes of the sentence"),
    }
    meta = {"block_description": _text(block["block_description"]),
            "graph_name": _text(block["graph_name"]),
            "paused": paused,
            "timed_out": timed_out,
            "delay_ms": int(_value(block, "delay_duration_ms", index)),
            # Vocabulary and loop of the dataset README session table, as convert_jude.py labels them.
            "vocab": vocab,
            "loop": loop,
            # Bin nearest to the trial end before the trim of a trailing odd bin.
            "trial_end_frame_nearest": nearest}
    return TrialSpec(trial_id=where, source_file=relative, source_index=index, neural=neural,
                     source_block=number, source_onset_frame=onset, source_offset_frame=offset,
                     targets=targets, meta=meta)


def list_sessions():
    """List the sessions that ``convert_jude.py`` converts and that have block files.

    Returns
    -------
    list of dict
        One item per session with ``session``, ``participant``, and
        ``sources``, the converted block files, in date order.

    Raises
    ------
    ValueError
        When a session folder is neither in ``SENTENCE_BLOCKS`` nor listed in
        ``EXCLUDED_SESSIONS``.
    """
    converted = _pipeline().SENTENCE_BLOCKS
    data = Path(study_path(STUDY, "data/extracted/Data"))
    sessions = []
    for directory in sorted(data.glob("*/*")) if data.is_dir() else []:
        name = directory.name
        if not (directory / "brainToText" / "RedisMat").is_dir():
            continue
        if name not in converted and name not in EXCLUDED_SESSIONS:
            raise ValueError(f"{name}: neither converted by convert_jude.py nor listed in EXCLUDED_SESSIONS")
        if name in converted:
            sessions.append({"session": name, "participant": name.split(".")[0],
                             "sources": [_relative(path) for _, path in sentence_blocks(name)]})
    return sessions


def iter_trials(session):
    """Yield every converted trial of one session, block by block in cue order.

    Parameters
    ----------
    session : str
        Native session identifier such as ``t17.2024.03.26``.

    Yields
    ------
    TrialSpec
        Trials in block order, then cue order. Blocks outside
        ``SENTENCE_BLOCKS`` and cues the pipeline rejects are left out and
        reported on standard error.

    Raises
    ------
    ValueError
        When no sentence block of the session has a block file.
    """
    files = _block_files(session)
    converted = _pipeline().SENTENCE_BLOCKS[session]
    numbers = sorted(number for number in files if number in converted)
    for number in sorted(files):
        if number not in converted:
            graph = _text(_load(files[number], HEADER)["graph_name"])
            print(f"skip      {session} block {number} ({graph}): {UNLISTED_SKIP_REASON}", file=sys.stderr)
    if not numbers:
        raise ValueError(f"{session}: no block of convert_jude.py SENTENCE_BLOCKS has a block file")
    for number in numbers:
        path = files[number]
        relative, block = _relative(path), _load(path)
        for index in range(len(block["cue"])):
            spec = _spec(session, block, index, relative)
            if spec is not None:
                yield spec


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
    session = str(attrs["trial_id"]).split("/")[0]
    path = Path(study_path(STUDY, str(attrs["source_file"])))
    return _spec(session, _load(path), int(attrs["source_index"]), str(attrs["source_file"]),
                 required=True)


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


def extra_attrs(session):
    """Return the root attributes a session shares by its task and grid.

    Parameters
    ----------
    session : str
        Native session identifier.

    Returns
    -------
    dict
        ``participant``, ``task``, and the neural bin size in milliseconds.
    """
    return {"participant": session.split(".")[0], "task": "intended speech",
            "neural_bin_size_ms": BIN_MS}


if __name__ == "__main__":
    raise SystemExit(export_cli(
        STUDY, list_sessions, iter_trials, read_source_trial, label_sets, __file__,
        dataset=DATASET, source_format="mat", extra_attrs=extra_attrs,
        description="Export Jude 2026b intended-speech sessions to the common session format."))
