# -*- coding: utf-8 -*-
"""Build the rapid-calibration split manifest from the exported Card 2024 sessions.

The rapid-calibration stage of this study (``prep_rapid_calibration.py``
followed by ``shared/finetune_chain/train_finetune.py``) fine-tunes a 40-day
baseline on one target day, and the manifest encodes how it uses the trials:

- Calibration trials are the ``data_train.hdf5`` trials of the target day in
  file order. A budget of ``N`` sentences takes the first ``N`` of them
  (``prep_rapid_calibration.py``, ``sorted(src.keys())[:n]``), so ``order`` is
  the zero-based position in that file and a budget of ``N`` is ``order < N``.
- Validation trials are every ``data_val.hdf5`` trial of the target day, in
  file order. ``prep_rapid_calibration.py`` copies the whole file and sets
  ``dataset_probability_val`` to 1 for the target day only.
- Trials of the other days take no part: ``prep_rapid_calibration.py`` writes
  empty training files for the 40 baseline days, so they get the role
  ``other``.

A day can be the target only when ``prep_rapid_calibration.py`` lists it in
its ``EXCLUDE`` sessions. The rows come from the exported session files.

Usage (from the repository root):

    python studies/20240815_card/code/build_split_manifest.py
    python studies/20240815_card/code/build_split_manifest.py --session-file PATH
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from session_format import read_session  # noqa: E402
from study_paths import release_prefix, study_checkout, study_path  # noqa: E402

STUDY = "20240815_card"
COLUMNS = ("session", "trial_id", "role", "order", "source_partition", "source_index",
           "block_num", "trial_num", "corpus")
SPLIT = "calibration_order"


def calibration_days():
    """Return the days that the rapid-calibration stage accepts as its target.

    Returns
    -------
    list of str
        ``prep_rapid_calibration.EXCLUDE``, the five val days that the 40-day
        baseline left out for the N-trial fine-tune.

    Notes
    -----
    The module is imported for its constant; importing it loads PyTorch but
    runs none of its code.
    """
    from prep_rapid_calibration import EXCLUDE
    return list(EXCLUDE)


def exported_sessions():
    """List the common session files exported for this study.

    Returns
    -------
    list of pathlib.Path
        Session files under ``data/processed``, in file-name order.
    """
    directory = Path(study_checkout(STUDY)) / "data" / "processed"
    return sorted(directory.glob("*/*.h5")) if directory.is_dir() else []


def manifest_rows(path, targets):
    """Return the manifest rows of one exported session.

    Parameters
    ----------
    path : pathlib.Path
        Common session file written by ``export_sessions.py``.
    targets : list of str
        Sessions treated as the calibration target.

    Returns
    -------
    list of dict
        One row per trial in session-file order.
    """
    with read_session(path) as session:
        name = session.attrs["session"]
        rows = []
        for trial in session.trials:
            attrs = trial.attrs
            partition = attrs.get("source_partition", "")
            if name in targets and partition == "train":
                role, order = "calibration", attrs["source_index"]
            elif name in targets and partition == "val":
                role, order = "validation", ""
            else:
                role, order = "other", ""
            rows.append({"session": name, "trial_id": trial.trial_id, "role": role,
                         "order": order, "source_partition": partition,
                         "source_index": attrs["source_index"],
                         "block_num": attrs["source_block"],
                         "trial_num": attrs.get("trial_num", ""),
                         "corpus": attrs.get("corpus", "")})
    return rows


def main():
    """Write the calibration-order manifest and print the role counts per session.

    Returns
    -------
    int
        Zero when the manifest was written.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("sessions", nargs="*", type=Path,
                        help="exported session files (default: every file of this study)")
    parser.add_argument("--target", action="append", default=[],
                        help="calibration target day (default: every exported day that "
                             "prep_rapid_calibration.py accepts)")
    parser.add_argument("--output", type=Path, help="output CSV (default: data/splits)")
    args = parser.parse_args()
    days = calibration_days()
    targets = args.target or days
    unknown = [name for name in targets if name not in days]
    if unknown:
        parser.error(f"not a rapid-calibration day: {unknown}; choose from {days}")
    paths = args.sessions or exported_sessions()
    if not paths:
        parser.error("no exported session file found; run export_sessions.py first")
    rows = [row for path in paths for row in manifest_rows(path, targets)]
    output = args.output or Path(study_path(
        STUDY, f"data/splits/{release_prefix(STUDY)}__{SPLIT}.csv"))
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for name in dict.fromkeys(row["session"] for row in rows):
        counts = Counter(row["role"] for row in rows if row["session"] == name)
        print(f"{name}: " + ", ".join(f"{counts.get(role, 0)} {role}" for role in
                                      ("calibration", "validation", "other")))
    print(f"wrote {len(rows)} rows -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
