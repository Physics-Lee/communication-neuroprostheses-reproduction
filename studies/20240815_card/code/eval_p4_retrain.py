# -*- coding: utf-8 -*-
"""Final eval of the P4 retrain best_checkpoint (run_p4_20261013).

Verbatim copy of eval_btot25_fromscratch.py semantics -- only the run dir and
logits path move. Fired automatically once training finishes.

Usage: python eval_p4_retrain.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from study_paths import study_path

import os

sys.path.insert(0, study_path("20240815_card", 'code'))

import eval_val_per as base  # noqa: E402

P4 = study_path("20240815_card", 'data/20241009_card_BtoT25/run_p4_20261013')
base.MODEL_PATH = P4
base.DATA_DIR = study_path("20240815_card", 'data/extracted/hdf5_data_final')
base.OUT_CSV = os.path.join(P4, "val_greedy_per_trial.csv")
base.OUT_SUMMARY = os.path.join(P4, "val_greedy_summary.json")

if __name__ == "__main__":
    base.main()
