# -*- coding: utf-8 -*-
"""Build the CORP adaptation split manifest from the exported Fan 2023 sessions.

Tier 3 of this study (``tier3_corp_sim_v7.py``) replays the online
recalibration of the ten follow-up sessions. It reads the ``recalibration``
arm of each session in file order (``testOnTrain``), trains on the official
``lm_decoded`` pseudo-labels, and keeps the final six trials of the arm out of
the training pool as the leakage-free holdout (``N_HOLD`` of v7). The manifest
gives every exported trial its role:

- ``adaptation``: recalibration-arm trials that may enter the training pool.
- ``holdout``: the final six recalibration-arm trials. ``order`` is the
  zero-based position in the arm, the ``online_trial_index`` of the archived
  ``paired_v7_trial_transcripts.csv``.
- ``frozen``: trials of the ``no_recalibration`` arm, which the tier-3
  simulation does not read.
- ``seed_training`` and ``seed_validation``: the ``train`` and ``test``
  partitions of the eleven seed sessions the released decoder was trained on.

The holdout rows are then compared with the archived evaluation list, which
records the scored trials by session, online index, and sentence.

Usage (from the repository root):

    python studies/20231201_fan/code/build_split_manifest.py [--require-all]
    python studies/20231201_fan/code/build_split_manifest.py SESSION_FILE ...

``--require-all`` fails when an archived evaluation session has no exported
session file.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from session_format import read_session  # noqa: E402
from study_paths import release_prefix, study_checkout, study_path  # noqa: E402

STUDY = "20231201_fan"
COLUMNS = ("session", "trial_id", "role", "order", "source_partition", "source_index",
           "block", "reference")
SPLIT = "paired_v7_holdout"
ARCHIVE = "archived_evaluation"
SEED_ROLE = {"train": "seed_training", "test": "seed_validation"}
# N_HOLD of tier3_corp_sim_v7.py; the archive comparison below checks it.
N_HOLD = 6


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
        One row per trial in session-file order.
    """
    rows = []
    with read_session(path) as session:
        name = session.attrs["session"]
        arms = {}
        for trial in session.trials:
            arms.setdefault(trial.attrs.get("source_arm", ""), []).append(trial)
        for arm, trials in arms.items():
            for order, trial in enumerate(trials):
                partition = trial.attrs.get("source_partition", "")
                if arm == "recalibration":
                    role = "holdout" if order >= len(trials) - N_HOLD else "adaptation"
                elif arm == "no_recalibration":
                    role = "frozen"
                else:
                    role = SEED_ROLE.get(partition, "other")
                rows.append({"session": name, "trial_id": trial.trial_id, "role": role,
                             "order": order if arm else "",
                             "source_partition": partition,
                             "source_index": trial.attrs["source_index"],
                             "block": trial.attrs["source_block"],
                             "reference": trial.sentence or ""})
    return rows


def compare_with_archive(rows, splits):
    """Compare the holdout rows with the archived evaluation list.

    Parameters
    ----------
    rows : list of dict
        Manifest rows.
    splits : pathlib.Path
        Study ``data/splits`` directory.

    Returns
    -------
    dict
        ``archived_rows``, ``archived_rows_of_exported_sessions``,
        ``matched_rows``, ``mismatched`` (archived rows of an exported session
        with no identical holdout row), and ``not_exported`` (archived
        sessions without a session file).
    """
    holdout = {(row["session"], row["order"], row["reference"]) for row in rows if row["role"] == "holdout"}
    exported = {row["session"] for row in rows}
    archived = read_rows(splits / f"{release_prefix(STUDY)}__{ARCHIVE}.csv")
    scored = [row for row in archived if row["session"] in exported]
    mismatched = [f"{row['session']} trial {row['archive_trial_id']}: {row['reference']!r}"
                  for row in scored
                  if (row["session"], int(row["archive_trial_id"]), row["reference"]) not in holdout]
    return {"archived_rows": len(archived), "archived_rows_of_exported_sessions": len(scored),
            "matched_rows": len(scored) - len(mismatched), "mismatched": mismatched,
            "not_exported": sorted({row["session"] for row in archived} - exported)}


def main():
    """Write the adaptation split manifest and print its comparison with the archive.

    Returns
    -------
    int
        Zero when every archived row matches a holdout row and, with
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
        counts = Counter(row["role"] for row in rows if row["session"] == name)
        print(f"{name}: " + ", ".join(f"{counts[role]} {role}" for role in sorted(counts)))
    print(f"wrote {len(rows)} rows -> {output}")
    failed = False
    if not args.output:
        result = compare_with_archive(rows, output.parent)
        for key, value in result.items():
            print(f"{key}: {value}")
        failed = bool(result["mismatched"]) or (args.require_all and result["not_exported"])
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
