# -*- coding: utf-8 -*-
"""Final eval of the FROM-SCRATCH BtoT25 run_120k best_checkpoint (Step-1 closing
of notes/20260903_rapid_calibration_plan.html).

Identical semantics to eval_val_per.py (the official-ckpt eval that produced
10.19%): same greedy CTC decode, same aggregation, same 41-day val set -- only
MODEL_PATH/outputs point at our run. Run concurrently with Step 2 training
(~2-3GB VRAM extra).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import eval_val_per as base  # noqa: E402  (reuse its functions/semantics)

base.MODEL_PATH = r".\20240815_card\data\20241009_card_BtoT25\run_120k"
base.DATA_DIR = r".\20240815_card\nejm-brain-to-text\data\hdf5_data_final"
base.OUT_CSV = r".\20240815_card\data\20241009_card_BtoT25\run_120k\val_greedy_per_trial.csv"
base.OUT_SUMMARY = r".\20240815_card\data\20241009_card_BtoT25\run_120k\val_greedy_summary.json"

if __name__ == "__main__":
    base.main()
