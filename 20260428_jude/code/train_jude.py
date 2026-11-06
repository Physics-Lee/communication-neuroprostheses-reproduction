# -*- coding: utf-8 -*-
"""Train the nejm-brain-to-text GRU decoder on converted jude 20260428 (T17) data.

Differences from train_btot24.py:
- 12 t17 sessions; features are 768-dim (384 ch x [SBP,TX]; 256-ch days padded)
- neural_dim / input layer sizes set to 768 accordingly

Usage: train_jude.py <num_batches> <output_root> [batches_per_val]
"""
import glob
import os
import sys

sys.path.insert(0, r".\20240815_card\nejm-brain-to-text\model_training")

import torch  # noqa: E402

torch.compile = lambda m, **k: m  # skip torch.compile on native Windows

from omegaconf import OmegaConf  # noqa: E402
from rnn_trainer import BrainToTextDecoder_Trainer  # noqa: E402

DATA_DIR = r".\20260428_jude\data\20260428_jude\hdf5"


def main():
    n_batches, out_root = int(sys.argv[1]), sys.argv[2]
    val_every = int(sys.argv[3]) if len(sys.argv) > 3 else 1000

    sessions = sorted(os.path.basename(p)
                      for p in glob.glob(os.path.join(DATA_DIR, "t17.*")))
    assert len(sessions) == 12, f"expected 12 sessions, got {len(sessions)}"

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
    args.model.n_input_features = 768
    args.model.input_network.input_layer_sizes = [768]
    args.dataset.neural_dim = 768
    args.dataset.dataset_dir = DATA_DIR
    args.dataset.sessions = list(sessions)
    args.dataset.dataset_probability_val = [1] * len(sessions)
    args.dataset.num_dataloader_workers = 0
    args.dataset.days_per_batch = min(4, len(sessions))

    print(f"sessions: {len(sessions)} | batches: {n_batches} | val every {val_every}")
    trainer = BrainToTextDecoder_Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
