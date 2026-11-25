# -*- coding: utf-8 -*-
"""Build the Willett 2023 split manifest from exported session files.

Writes ``data/splits/dryad_x69p8czpq__source_partitions.csv`` with one row per
exported trial: the publisher partition it belongs to and the role this
reproduction gives that partition (the ``test`` partition is the validation
set of the offline evaluation; ``competitionHoldOut`` is unlabeled). The rows
of the ``test`` partition are then checked against the archived evaluation
list, which records the evaluated trials by session and row.

Usage (from the repository root):

    python studies/20230831_willett/code/build_split_manifest.py [--require-all]

``--require-all`` fails when an archived evaluation session has no exported
session file.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from session_format import read_session  # noqa: E402
from study_paths import study_checkout  # noqa: E402

import argparse  # noqa: E402
import csv  # noqa: E402

STUDY = "20230831_willett"
ROLE = {"train": "training", "test": "validation", "competitionHoldOut": "unlabeled"}
MANIFEST = "dryad_x69p8czpq__source_partitions.csv"
ARCHIVE = "dryad_x69p8czpq__archived_evaluation.csv"


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


def build(splits, sessions):
    """Write the manifest and compare its validation rows with the archive.

    Parameters
    ----------
    splits : pathlib.Path
        Study ``data/splits`` directory.
    sessions : list of pathlib.Path
        Exported session files in session order.

    Returns
    -------
    dict
        Rows written, archived rows covered, sessions that matched, and
        sessions whose archived rows differ from the exported trials.
    """
    rows, by_session = [], {}
    for path in sessions:
        with read_session(path) as session:
            for trial in session.trials:
                partition = trial.attrs.get("source_partition", "")
                rows.append({"session": session.attrs["session"], "trial_id": trial.trial_id,
                             "partition": partition, "role": ROLE.get(partition, ""),
                             "block": trial.attrs["source_block"], "reference": trial.sentence or ""})
                if partition == "test":
                    by_session.setdefault(session.attrs["session"], []).append(trial.sentence)
    splits.mkdir(parents=True, exist_ok=True)
    with (splits / MANIFEST).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    archived = {}
    for row in read_rows(splits / ARCHIVE):
        archived.setdefault(row["session"], []).append(row["reference"])
    result = {"rows": len(rows), "validation_trials": sum(len(v) for v in by_session.values()),
              "archived_sessions": len(archived), "matched_sessions": [], "differing_sessions": [],
              "not_exported": sorted(name for name in archived if name not in by_session)}
    for name, references in archived.items():
        if name in by_session:
            result["matched_sessions" if by_session[name] == references else "differing_sessions"].append(name)
    return result


def main():
    """Build the manifest and print the comparison with the archive.

    Returns
    -------
    int
        Zero when every exported session matches the archive and, with
        ``--require-all``, every archived session is exported.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--require-all", action="store_true",
                        help="fail when an archived evaluation session is not exported")
    args = parser.parse_args()
    directory = Path(study_checkout(STUDY)) / "data" / "processed"
    sessions = sorted(directory.glob("*/*.h5")) if directory.is_dir() else []
    if not sessions:
        raise SystemExit(f"No exported session files under {directory}")
    result = build(Path(study_checkout(STUDY)) / "data" / "splits", sessions)
    for key, value in result.items():
        print(f"{key}: {value}")
    failed = bool(result["differing_sessions"]) or (args.require_all and result["not_exported"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
