# -*- coding: utf-8 -*-
"""Scratch training with session subsets and/or the BiGRU architecture
(64h block A/B: T4 biGRU-45sess, T2a GRU-44sess excl 2025.01.10, T2b GRU-39sess).

Same official recipe/optimizer/accum semantics as train_btot25.py; the model
class is swapped by monkeypatching rnn_model.GRUDecoder BEFORE rnn_trainer is
imported (upstream stays untouched).

Usage:
  train_bigrun.py <model:gru|bigru> <sessions:all45|excl0110|first39> \
                  <n_opt_batches> <out_root> [val_every=2000] [accum=2]
"""
import os
import sys

sys.path.insert(0, r".\20240815_card\nejm-brain-to-text\model_training")
sys.path.insert(0, r".\20240815_card\code")

import torch  # noqa: E402

torch.compile = lambda m, **k: m  # skip torch.compile on native Windows

import rnn_model  # noqa: E402
from calib_models import BiGRUDecoder  # noqa: E402

MODEL, SESS_MODE = sys.argv[1], sys.argv[2]
if MODEL == "bigru":
    rnn_model.GRUDecoder = BiGRUDecoder  # patch before rnn_trainer import

from omegaconf import OmegaConf  # noqa: E402
from rnn_trainer import BrainToTextDecoder_Trainer  # noqa: E402

DATA_DIR = r".\20240815_card\nejm-brain-to-text\data\hdf5_data_final"
EXCLUDE = {"excl0110": ["t15.2025.01.10"]}


def main():
    """Train a GRU or BiGRU decoder from scratch on a chosen session subset.

    Loads the official rnn_args.yaml, applies the requested session mode
    (all45 / excl0110 / first39, keeping the val probability list in sync),
    overrides the training-horizon hyperparameters, and runs
    ``BrainToTextDecoder_Trainer.train()``. The decoder class itself was
    already swapped at import time via ``rnn_model.GRUDecoder``.

    Returns
    -------
    None
        Checkpoints and logs are written under ``out_root``; the run summary
        is printed to stdout.

    Raises
    ------
    SystemExit
        If ``SESS_MODE`` (second CLI argument) is not a known session mode.
    """
    n_opt_batches, out_root = int(sys.argv[3]), sys.argv[4]
    val_every = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
    accum = int(sys.argv[6]) if len(sys.argv) > 6 else 2

    args = OmegaConf.load(r".\20240815_card\nejm-brain-to-text\model_training\rnn_args.yaml")
    args.gpu_number = "0"
    args.output_dir = out_root
    args.checkpoint_dir = os.path.join(out_root, "checkpoint")
    args.num_training_batches = n_opt_batches * accum
    args.lr_decay_steps = n_opt_batches
    args.lr_decay_steps_day = n_opt_batches
    args.batches_per_train_log = 50 * accum
    args.batches_per_val_step = val_every * accum
    args.batches_per_save = 0
    args.save_val_logits = False
    args.gradient_accumulation_steps = accum
    args.dataset.dataset_dir = DATA_DIR
    args.dataset.num_dataloader_workers = 0
    args.dataset.batch_size = 64 // accum

    sessions = list(args.dataset.sessions)
    probs = list(args.dataset.dataset_probability_val)
    if SESS_MODE == "all45":
        pass
    elif SESS_MODE == "excl0110":
        keep = [i for i, s in enumerate(sessions) if s not in EXCLUDE["excl0110"]]
        sessions = [sessions[i] for i in keep]
        probs = [probs[i] for i in keep]
    elif SESS_MODE == "first39":
        sessions, probs = sessions[:39], probs[:39]
    else:
        raise SystemExit(f"unknown sessions mode {SESS_MODE}")
    args.dataset.sessions = sessions
    args.dataset.dataset_probability_val = probs

    n_val = sum(1 for p in probs if p == 1)
    print(f"model={MODEL} sessions={len(sessions)} (val days {n_val}) "
          f"opt_batches={n_opt_batches} micro={args.dataset.batch_size}x{accum}", flush=True)
    trainer = BrainToTextDecoder_Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
