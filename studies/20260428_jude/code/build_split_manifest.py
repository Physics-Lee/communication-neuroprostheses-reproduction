# -*- coding: utf-8 -*-
"""Build the day-level train/validation split manifest from the exported Jude 2026b sessions.

``convert_jude.py`` follows the BtoT24 rule: of the sentence blocks it
converts on one day, the block with the highest number that has a block file
becomes the validation split and every other sentence block is training. The
manifest gives every exported trial its role, ``validation`` or ``training``,
and repeats the vocabulary and loop of its block from the dataset README
table (``SENTENCE_BLOCKS``).

The validation rows are then compared with the archived evaluation list,
which records the evaluated trials by session and sentence.

Usage (from the repository root):

    python studies/20260428_jude/code/build_split_manifest.py [--require-all]
    python studies/20260428_jude/code/build_split_manifest.py SESSION_FILE ...

``--require-all`` fails when an archived evaluation session has no exported
session file.
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from session_format import read_session  # noqa: E402
from study_paths import release_prefix, study_checkout, study_path  # noqa: E402
import export_sessions  # noqa: E402

STUDY = "20260428_jude"
COLUMNS = ("session", "trial_id", "role", "block", "vocab", "loop", "reference")
SPLIT = "validation_blocks"
ARCHIVE = "archived_evaluation"


def read_rows(path):
    """Read a CSV file into a list of dicts.

    Parameters
    ----------
    path : pathlib.Path
        CSV file.

    Returns
    -------
    list of dict
        Rows in file order.
    """
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def exported_sessions():
    """List the common session files exported for this study.

    Returns
    -------
    list of pathlib.Path
        Session files under ``data/processed``, in file-name order.
    """
    directory = Path(study_checkout(STUDY)) / "data" / "processed"
    return sorted(directory.glob("*/*.h5")) if directory.is_dir() else []


def manifest_rows(path):
    """Return the manifest rows of one exported session.

    Parameters
    ----------
    path : pathlib.Path
        Common session file written by ``export_sessions.py``.

    Returns
    -------
    list of dict
        One row per trial in session-file order. ``role`` is ``validation``
        for the day's last sentence block, as ``main`` of ``convert_jude.py``
        picks it, and ``training`` for the other sentence blocks.
    """
    rows = []
    with read_session(path) as session:
        name = session.attrs["session"]
        last_block = max(number for number, _ in export_sessions.sentence_blocks(name))
        for trial in session.trials:
            block = trial.attrs["source_block"]
            rows.append({"session": name, "trial_id": trial.trial_id,
                         "role": "validation" if block == last_block else "training",
                         "block": block, "vocab": trial.attrs["vocab"], "loop": trial.attrs["loop"],
                         "reference": trial.sentence or ""})
    return rows


def compare_with_archive(rows, splits):
    """Compare the validation rows with the archived evaluation list.

    Parameters
    ----------
    rows : list of dict
        Manifest rows.
    splits : pathlib.Path
        Study ``data/splits`` directory.

    Returns
    -------
    dict
        ``archived_sessions``, ``matched_sessions``, ``differing_sessions``
        (archived sentence lists that differ from the exported validation
        sentences), and ``not_exported`` (archived sessions without a session
        file).
    """
    validation = {}
    for row in rows:
        if row["role"] == "validation":
            validation.setdefault(row["session"], []).append(row["reference"])
    archived = {}
    for row in read_rows(splits / f"{release_prefix(STUDY)}__{ARCHIVE}.csv"):
        archived.setdefault(row["session"], []).append(row["reference"])
    return {"archived_sessions": len(archived),
            "matched_sessions": [name for name in archived if validation.get(name) == archived[name]],
            "differing_sessions": [name for name in archived
                                   if name in validation and validation[name] != archived[name]],
            "not_exported": sorted(name for name in archived if name not in validation)}


def main():
    """Write the split manifest and print its comparison with the archive.

    Returns
    -------
    int
        Zero when every exported session matches the archive and, with
        ``--require-all``, every archived session is exported.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sessions", nargs="*", type=Path,
                        help="exported session files (default: every file of this study)")
    parser.add_argument("--require-all", action="store_true",
                        help="fail when an archived evaluation session is not exported")
    parser.add_argument("--output", type=Path, help="output CSV (default: data/splits)")
    args = parser.parse_args()
    paths = args.sessions or exported_sessions()
    if not paths:
        parser.error("no exported session file found; run export_sessions.py first")
    rows = [row for path in paths for row in manifest_rows(path)]
    output = args.output or Path(study_path(
        STUDY, f"data/splits/{release_prefix(STUDY)}__{SPLIT}.csv"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for name in dict.fromkeys(row["session"] for row in rows):
        counts = {}
        for row in rows:
            if row["session"] == name:
                counts[row["role"]] = counts.get(row["role"], 0) + 1
        blocks = sorted({row["block"] for row in rows if row["session"] == name})
        print(f"{name}: " + ", ".join(f"{counts[role]} {role}" for role in sorted(counts))
              + f" (blocks {blocks[0]}-{blocks[-1]})")
    print(f"wrote {len(rows)} rows -> {output}")
    failed = False
    if not args.output:
        result = compare_with_archive(rows, output.parent)
        for key, value in result.items():
            print(f"{key}: {value}")
        failed = bool(result["differing_sessions"]) or (args.require_all and result["not_exported"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
