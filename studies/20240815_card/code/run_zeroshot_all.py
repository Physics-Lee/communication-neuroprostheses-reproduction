# -*- coding: utf-8 -*-
"""Run zero-shot eval for all 5 excluded days (driver over eval_val_per)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from study_paths import derived_data_path, study_path

import os

sys.path.insert(0, study_path("20240815_card", 'code'))
import eval_val_per as base  # noqa: E402

RUNS = ["ft_t1520250112_N0", "ft_t1520250314_N0", "ft_t1520250316_N0",
        "ft_t1520250330_N0", "ft_t1520250413_N0"]

for run in RUNS:
    root = derived_data_path("20240815_card", f"hdf5/rapid_calibration/{run}",
                             legacy=f"data/20241009_card_BtoT25/{run}")
    base.MODEL_PATH = root
    base.DATA_DIR = root
    base.OUT_CSV = os.path.join(root, "zeroshot_per_trial.csv")
    base.OUT_SUMMARY = os.path.join(root, "zeroshot_summary.json")
    print(f"\n########## {run} ##########", flush=True)
    try:
        base.main()
    except SystemExit:
        pass
