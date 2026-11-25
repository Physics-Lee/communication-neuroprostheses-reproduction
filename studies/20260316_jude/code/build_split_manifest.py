# -*- coding: utf-8 -*-
"""Build the Jude 2026a split manifest from exported session files.

Writes ``data/splits/dryad_cz8w9gjjk__test_trials.csv`` with one row per
exported trial and the role the official ``mat_to_tfrecord`` conversion
gives it. ``window_W4_convert.py`` records the arguments of each conversion
in ``Typing/tfdata_20ms/mat_to_tfrecord_params.json``; the roles follow from
them and from the publisher's per-trial flags:

1. The cues of the blocks listed in ``block_nums`` (in that order, then cue
   order) are kept unless they are listed in ``trials_to_remove``, paused,
   timed out, or carry an empty or ``DO_NOTHING`` cue.
2. The kept positions are shuffled with NumPy seed 9; the last
   ``num_test_trials`` are the test set, the rest the training set.
3. A trial whose 20 ms frame count does not exceed four times its number of
   characters plus ten is not written to either TFRecord, although it keeps
   its position in the shuffle.

Every session is then compared with the TFRecords of its existing
conversion, which are only read: the record counts, their order, and their
frame counts and labels must agree with the manifest, or nothing is
written. The test references must also agree with the archived evaluation
list, which numbers the evaluated trials by session and row.

Usage (from the repository root):

    python studies/20260316_jude/code/build_split_manifest.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from session_format import read_session  # noqa: E402
from study_paths import study_checkout, study_path  # noqa: E402

import csv  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import re  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_sessions import SPLIT_SEED, pipeline_split  # noqa: E402

STUDY = "20260316_jude"
MANIFEST = "dryad_cz8w9gjjk__test_trials.csv"
ARCHIVE = "dryad_cz8w9gjjk__archived_evaluation.csv"
MAX_SEQ = 60
FIELDS = ["session", "trial_id", "block", "role", "split_rule", "split_index", "split_rank",
          "record_row", "reference", "exclusion_reason"]


def conversion(session):
    """Read the recorded arguments of a session's conversion.

    Parameters
    ----------
    session : str
        Native session identifier such as ``t17.2024.04.25``.

    Returns
    -------
    dict
        ``blocks`` (``block_nums``), ``num_test``, ``removed`` mapping a
        block number to the cue positions of ``trials_to_remove``, and
        ``directory``, the ``tfdata_20ms`` folder of the conversion.

    Raises
    ------
    SystemExit
        When the session has no conversion on disk.
    """
    participant = re.split(r"[._]", session, maxsplit=1)[0]
    directory = Path(study_path(
        STUDY, f"data/extracted/Data/{participant}/{session}/Typing/tfdata_20ms"))
    source = directory / "mat_to_tfrecord_params.json"
    if not source.is_file():
        raise SystemExit(f"{session}: no conversion under {directory}; the split rule needs its arguments")
    with source.open(encoding="utf-8") as stream:
        params = json.load(stream)
    return {"blocks": [int(block) for block in params["block_nums"]],
            "num_test": int(params["num_test_trials"]),
            "removed": {int(block): [int(cue) for cue in cues]
                        for block, cues in params["trials_to_remove"].items()},
            "directory": directory}


def exclusion_reason(trial, rule):
    """Return why the conversion drops a trial, or an empty string when it keeps it.

    Parameters
    ----------
    trial : session_format.Trial
        Exported trial.
    rule : dict
        Conversion arguments (see ``conversion``).

    Returns
    -------
    str
        ``block_not_converted``, ``removed_in_conversion``, ``paused``,
        ``timed_out``, ``empty_or_do_nothing_cue``, or ``''``.
    """
    block = int(trial.attrs["source_block"])
    if block not in rule["blocks"]:
        return "block_not_converted"
    if int(trial.attrs["source_index"]) in rule["removed"].get(block, []):
        return "removed_in_conversion"
    if trial.attrs["paused"] == 1:
        return "paused"
    if trial.attrs["timed_out"] == 1:
        return "timed_out"
    sentence = trial.sentence or ""
    if not sentence.split() or "DO_NOTHING" in sentence:
        return "empty_or_do_nothing_cue"
    return ""


def session_rows(session, rule):
    """Assign the split positions of one session.

    Parameters
    ----------
    session : session_format.Session
        Open exported session.
    rule : dict
        Conversion arguments (see ``conversion``).

    Returns
    -------
    list of dict
        One manifest row per exported trial, in trial order, plus the
        private keys ``_frames``, ``_ids``, and ``_text`` that hold the
        trial's 20 ms frame count and labels for ``compare_records``.
    """
    rows = []
    for trial in session.trials:
        names = trial.names("targets")
        rows.append({
            "session": session.attrs["session"], "trial_id": trial.trial_id,
            "block": int(trial.attrs["source_block"]), "role": "excluded", "split_rule": "",
            "split_index": -1, "split_rank": -1, "record_row": -1,
            "reference": trial.sentence or "", "exclusion_reason": exclusion_reason(trial, rule),
            "_frames": int(trial.attrs["pipeline_frame_offset"]),
            "_ids": trial["targets/char_class_ids"] if "char_class_ids" in names else np.zeros(0, np.int64),
            "_text": trial["targets/transcription"] if "transcription" in names else ""})
    kept = [row for row in rows if not row["exclusion_reason"]]
    train, test = pipeline_split(len(kept), rule["num_test"])
    name = f"shuffle_seed{SPLIT_SEED}_last{rule['num_test']}_of_{len(kept)}"
    for index, row in enumerate(kept):
        row["split_index"], row["split_rule"] = index, name
    for rank, index in enumerate(np.concatenate((train, test))):
        row = kept[index]
        row["split_rank"] = rank
        # convertToTFRecord writes a trial only when it has more 20 ms frames than four per character plus ten.
        if row["_frames"] > 4 * len(row["_ids"]) + 10:
            row["role"] = "train" if rank < len(train) else "test"
        else:
            row["exclusion_reason"] = "too_short"
    return rows


def tf_records(directory):
    """Read the labels of every record of a conversion's TFRecords.

    Parameters
    ----------
    directory : pathlib.Path
        ``tfdata_20ms`` directory of one conversion.

    Returns
    -------
    dict of str to list of dict or None
        ``train`` and ``test``: the records in file order, each with
        ``nTimeSteps``, ``nSeqElements``, ``seqClassIDs``, and
        ``transcription`` as loaded, or ``None`` when the split has no file.

    Notes
    -----
    Only the label features are parsed; the feature arrays are skipped.
    """
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf
    spec = {"seqClassIDs": tf.io.FixedLenFeature((MAX_SEQ,), tf.int64),
            "nTimeSteps": tf.io.FixedLenFeature((), tf.int64),
            "nSeqElements": tf.io.FixedLenFeature((), tf.int64),
            "transcription": tf.io.FixedLenFeature((MAX_SEQ,), tf.int64)}

    def parse(proto):
        return tf.io.parse_single_example(proto, spec)

    records = {}
    for split in ("train", "test"):
        files = sorted(str(path) for path in (directory / split).glob("*.tfrecord"))
        records[split] = None if not files else [
            {name: value.numpy() for name, value in parsed.items()}
            for parsed in tf.data.TFRecordDataset(files).map(parse)]
    return records


def compare_records(rows, records):
    """Check the manifest of one session against its TFRecords and number the records.

    Parameters
    ----------
    rows : list of dict
        Manifest rows of one session (see ``session_rows``).
    records : dict
        Loaded TFRecords of the session (see ``tf_records``).

    Returns
    -------
    list of str
        One message per difference; empty when the records agree. The
        ``record_row`` of every written trial is set to its position in the
        ``train`` or ``test`` TFRecord.
    """
    problems = []
    for split in ("train", "test"):
        expected = sorted((row for row in rows if row["role"] == split), key=lambda row: row["split_rank"])
        loaded = records.get(split)
        if loaded is None:
            continue
        if len(loaded) != len(expected):
            problems.append(f"{split}: {len(loaded)} TFRecords against {len(expected)} manifest rows")
        for position, (row, record) in enumerate(zip(expected, loaded)):
            row["record_row"] = position
            count = int(record["nSeqElements"])
            ids = record["seqClassIDs"][:count] - 1  # the TFRecord stores every class id plus one
            text = "".join(chr(int(code)) for code in record["transcription"] if code)
            if int(record["nTimeSteps"]) != row["_frames"] or not np.array_equal(ids, row["_ids"]) or text != row["_text"]:
                problems.append(f"{split} record {position} is {text!r} with {int(record['nTimeSteps'])} frames, "
                                f"the manifest has {row['trial_id']} {row['_text']!r} with {row['_frames']} frames")
    return problems


def archived_references(splits):
    """Collect the archived evaluation references of every session.

    Parameters
    ----------
    splits : pathlib.Path
        Study ``data/splits`` directory.

    Returns
    -------
    dict of str to list of str
        References of each session, in the order of their archive row.
    """
    archived = {}
    source = splits / ARCHIVE
    if source.is_file():
        with source.open(encoding="utf-8-sig", newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["reference"]]
        for name in sorted({row["session"] for row in rows}):
            archived[name] = [row["reference"] for row in sorted(
                (row for row in rows if row["session"] == name), key=lambda row: int(row["archive_row"]))]
    return archived


def build(splits, sessions):
    """Assign every exported trial its role, check it against the conversion, and write the manifest.

    Parameters
    ----------
    splits : pathlib.Path
        Study ``data/splits`` directory.
    sessions : list of pathlib.Path
        Exported session files in session order.

    Returns
    -------
    dict
        Rows written, counts by role, sessions compared with their
        TFRecords, sessions whose test references differ from the archive,
        and the number of archived sessions.

    Raises
    ------
    SystemExit
        When a session's manifest rows disagree with its TFRecords; the
        manifest is not written then.
    """
    rows, result = [], {"sessions": len(sessions), "compared_sessions": 0, "differing_sessions": [],
                        "archived_sessions": 0}
    archived = archived_references(splits)
    for path in sessions:
        with read_session(path) as session:
            name = session.attrs["session"]
            rule = conversion(name)
            mine = session_rows(session, rule)
        records = tf_records(rule["directory"])
        problems = compare_records(mine, records)
        if problems:
            raise SystemExit(f"{name}: the split rule does not reproduce the conversion:\n  " + "\n  ".join(problems))
        result["compared_sessions"] += int(any(part is not None for part in records.values()))
        test = sorted((row for row in mine if row["role"] == "test"), key=lambda row: row["record_row"])
        references = [re.sub(r"[.,?]", "", row["_text"]).strip() for row in test]
        if name in archived and references != archived[name]:
            result["differing_sessions"].append(name)
        rows.extend(mine)
    for role in ("train", "test", "excluded"):
        result[role] = sum(row["role"] == role for row in rows)
    result["rows"] = len(rows)
    result["archived_sessions"] = len(archived)
    splits.mkdir(parents=True, exist_ok=True)
    with (splits / MANIFEST).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return result


if __name__ == "__main__":
    directory = Path(study_checkout(STUDY)) / "data" / "processed"
    exported = sorted(directory.glob("*/*.h5")) if directory.is_dir() else []
    if not exported:
        raise SystemExit(f"No exported session files under {directory}")
    outcome = build(Path(study_checkout(STUDY)) / "data" / "splits", exported)
    for key, value in outcome.items():
        print(f"{key}: {value}")
    raise SystemExit(1 if outcome["differing_sessions"] else 0)
