# -*- coding: utf-8 -*-
"""Rapid-calibration Step 3 prep (notes/20260903_rapid_calibration_plan.html).

Builds, for one target (excluded) day and an N-trial budget:
  D:\\repositories\\20240815_card\\data\\20241009_card_BtoT25\\ft_<day>_N<N>\\
    t15.* session dirs:
      - 40 kept days: EMPTY data_train.hdf5 (0 trials; empty-day patch skips them)
      - target day:   data_train.hdf5 = first N trials, data_val.hdf5 = full val
    checkpoint/args.yaml  -> sessions = 40 kept + target (target = day index 40)
    checkpoint/best_checkpoint -> Step-2 ckpt EXPANDED with an identity-init
      day layer appended at the end (official new-day mechanism: eye(512)/zeros)

With N=0 the target day has no data_train at all -> pure ZERO-SHOT config
(eval the expanded ckpt on the target day's val trials, day layer untrained).

Usage: prep_rapid_calibration.py <target_session> <N>   (N=0 for zero-shot)
"""
import os
import shutil
import sys

import h5py
import torch
from omegaconf import OmegaConf

SRC_DATA = r".\20240815_card\nejm-brain-to-text\data\hdf5_data_final"
STEP2 = r".\20240815_card\data\20241009_card_BtoT25\run_excl5_40k"
FT_BASE = r".\20240815_card\data\20241009_card_BtoT25"
EXCLUDE = ["t15.2025.01.12", "t15.2025.03.14", "t15.2025.03.16",
           "t15.2025.03.30", "t15.2025.04.13"]


def main():
    """Build one rapid-calibration fine-tune config for a target day.

    Creates ``ft_<day>_N<N>`` under ``FT_BASE`` with empty train splits for
    the 40 kept days, the first N train trials plus full val for the target
    day, a rewritten ``args.yaml`` (41 sessions, val = target only), and the
    Step-2 checkpoint expanded with an identity-init day layer at index 40.

    Returns
    -------
    None
        A summary line is printed to stdout.

    Raises
    ------
    AssertionError
        If the target is not an excluded day or the Step-2 session list is
        not the expected 40 kept days.
    SystemExit
        If the output directory already exists (refuses to overwrite).
    """
    target, n = sys.argv[1], int(sys.argv[2])
    assert target in EXCLUDE, f"{target} not an excluded day"
    out_root = os.path.join(FT_BASE, f"ft_{target.replace('.', '')}_N{n}")
    if os.path.exists(out_root):
        raise SystemExit(f"refusing to overwrite {out_root}")
    os.makedirs(os.path.join(out_root, "checkpoint"))

    # session list: 40 kept (Step-2 order) + target appended last (day idx 40)
    step2_args = OmegaConf.load(os.path.join(STEP2, "checkpoint", "args.yaml"))
    kept = list(step2_args.dataset.sessions)
    assert len(kept) == 40 and target not in kept
    sessions = kept + [target]

    # dataset dirs
    for s in kept:
        d = os.path.join(out_root, s)
        os.makedirs(d, exist_ok=True)
        with h5py.File(os.path.join(d, "data_train.hdf5"), "w"):
            pass  # zero trials -> skipped by empty-day patch
    td = os.path.join(out_root, target)
    os.makedirs(td)
    with h5py.File(os.path.join(SRC_DATA, target, "data_train.hdf5"), "r") as src, \
         h5py.File(os.path.join(td, "data_train.hdf5"), "w") as dst:
        if n > 0:
            for key in sorted(src.keys())[:n]:
                src.copy(key, dst)
    shutil.copy(os.path.join(SRC_DATA, target, "data_val.hdf5"),
                os.path.join(td, "data_val.hdf5"))

    # args.yaml: same hyperparams as Step-2 run, 41 sessions, val = target only
    args = OmegaConf.load(os.path.join(STEP2, "checkpoint", "args.yaml"))
    args.output_dir = out_root
    args.checkpoint_dir = os.path.join(out_root, "checkpoint")
    args.dataset.dataset_dir = out_root
    args.dataset.sessions = sessions
    args.dataset.dataset_probability_val = [0] * 40 + [1]
    args.dataset.days_per_batch = 1
    OmegaConf.save(args, os.path.join(out_root, "checkpoint", "args.yaml"))

    # expanded checkpoint: append identity day layer at index 40
    ckpt = torch.load(os.path.join(STEP2, "checkpoint", "best_checkpoint"),
                      weights_only=False, map_location="cpu")
    sd = ckpt["model_state_dict"]
    sd["day_weights.40"] = torch.eye(512)
    sd["day_biases.40"] = torch.zeros(1, 512)
    # drop pretraining optimizer/scheduler state so fine-tune LR schedules
    # built from args are kept (requires the rnn_trainer local patch);
    # best_val reset so every fine-tune val can save a new best checkpoint
    for k in ("optimizer_state_dict", "scheduler_state_dict"):
        ckpt.pop(k, None)
    ckpt["val_PER"] = float("inf")
    torch.save(ckpt, os.path.join(out_root, "checkpoint", "best_checkpoint"))

    n_val = len(list(h5py.File(os.path.join(td, "data_val.hdf5"), "r").keys()))
    print(f"built {out_root}: train={n} trials (target day), val={n_val}, "
          f"sessions=41, expanded ckpt saved (day 40 = identity init)")


if __name__ == "__main__":
    main()
