# -*- coding: utf-8 -*-
"""Train one fold of the T16 attempted-speech holdout baseline (review A.4 task 3).

Same protocol as train_kunz.py (official rnn_args.yaml, best checkpoint by val
greedy PER) but on the isolated hdf5_t16holdout root: train = two blocks of the
attempted session (80 trials), val = the held-out block (40 trials).

Usage: train_t16_holdout.py <fold 1|2|3> <num_batches> <out_root> [val_every=100]
"""
import glob
import os
import sys

sys.path.insert(0, r".\20240815_card\nejm-brain-to-text\model_training")

import torch  # noqa: E402

torch.compile = lambda m, **k: m  # skip torch.compile on native Windows

from omegaconf import OmegaConf  # noqa: E402
from rnn_trainer import BrainToTextDecoder_Trainer  # noqa: E402

ROOT = r".\20250821_kunz\data\20250821_kunz\hdf5_t16holdout"


def main():
    fold, n_batches, out_root = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    val_every = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    data_dir = os.path.join(ROOT, f"fold{fold}")

    sessions = sorted(os.path.basename(p) for p in glob.glob(os.path.join(data_dir, "*")))
    val_dir = [s for s in sessions if s.endswith("val")]
    assert len(val_dir) == 1, f"expected exactly one val session for fold, got {sessions}"
    train_sessions = [s for s in sessions if s != val_dir[0]]
    days_per_batch = min(4, len(train_sessions))

    args = OmegaConf.load(r".\20240815_card\nejm-brain-to-text\model_training\rnn_args.yaml")
    args.gpu_number = "0"
    args.output_dir = out_root
    args.checkpoint_dir = os.path.join(out_root, "checkpoint")
    args.num_training_batches = n_batches
    args.lr_decay_steps = n_batches
    args.lr_decay_steps_day = n_batches
    args.batches_per_train_log = 50
    args.batches_per_val_step = val_every
    args.batches_per_save = 0
    args.save_val_logits = False
    args.dataset.dataset_dir = data_dir
    args.dataset.sessions = list(sessions)
    args.dataset.dataset_probability_val = [1] * len(sessions)
    args.dataset.num_dataloader_workers = 0
    args.dataset.days_per_batch = days_per_batch

    print(f"[t16-holdout/fold{fold}] train sessions: {train_sessions}")
    print(f"[t16-holdout/fold{fold}] all sessions: {sessions}")
    print(f"days_per_batch={days_per_batch} | batches={n_batches} | val every {val_every}")
    trainer = BrainToTextDecoder_Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
