# -*- coding: utf-8 -*-
"""Train the nejm-brain-to-text GRU decoder on BtoT25 from scratch (official recipe).

Unlike train_btot24.py this needs NO data conversion: hdf5_data_final is the
trainer's native dataset. The official rnn_args.yaml already lists all 45
sessions with the official dataset_probability_val split (4 train-only days),
so sessions/probabilities are kept EXACTLY as-is -- only paths, gpu, workers
and logging cadence are adapted for this Windows machine.

Memory note (measured 2026-09-02): official 64-trial batches of BtoT25's long
copy-task trials (median 845, p99 1806, max 2475 frames) overflow a 16GB GPU
into WDDM shared memory (~10 s/batch, 40W "99% util" thrash). The trainer's
gradient-accumulation patch reproduces the full-batch mean gradient exactly:
micro-batch 32 x accum 2 == official batch 64, optimizer/scheduler steps stay
official. num_training_batches (micro-batches) = optimizer_batches * accum.

Usage: train_btot25.py <optimizer_batches> <output_root> [val_every_opt_steps=2000] [accum=2]
"""
import os
import sys

sys.path.insert(0, r".\20240815_card\nejm-brain-to-text\model_training")

import torch  # noqa: E402

torch.compile = lambda m, **k: m  # skip torch.compile on native Windows

from omegaconf import OmegaConf  # noqa: E402
from rnn_trainer import BrainToTextDecoder_Trainer  # noqa: E402

# SSD copy of .\20240815_card\nejm-brain-to-text\data\hdf5_data_final (robocopy
# 2026-09-02, byte-identical): the D: original sits on a 2TB SATA HDD whose
# random 45-day-sampled reads bottleneck training; RAM (32GB) cannot cache
# the full 12GB dataset alongside the ~13GB training process.
DATA_DIR = r".\20240815_card\nejm-brain-to-text\data\hdf5_data_final"


def main():
    """Train the official-recipe GRU decoder on BtoT25 from scratch.

    Loads rnn_args.yaml, keeps the official 45-session split and
    dataset_probability_val exactly as-is, sets Windows-local paths/workers,
    and reproduces the official 64-trial batch via micro-batch x gradient
    accumulation before running ``BrainToTextDecoder_Trainer.train()``.

    Returns
    -------
    None
        Checkpoints and logs are written under ``out_root``; the run summary
        is printed to stdout.

    Raises
    ------
    AssertionError
        If the official 45-session / 41-val-day split is not found in the
        yaml, or if ``accum`` does not divide the official batch size 64.
    """
    n_opt_batches, out_root = int(sys.argv[1]), sys.argv[2]
    val_every = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    accum = int(sys.argv[4]) if len(sys.argv) > 4 else 2

    args = OmegaConf.load(r".\20240815_card\nejm-brain-to-text\model_training\rnn_args.yaml")
    args.gpu_number = "0"
    args.output_dir = out_root
    args.checkpoint_dir = os.path.join(out_root, "checkpoint")
    # optimizer-step horizon (LR schedule) vs micro-batch count (data loader)
    args.num_training_batches = n_opt_batches * accum
    args.lr_decay_steps = n_opt_batches
    args.lr_decay_steps_day = n_opt_batches
    args.batches_per_train_log = 50 * accum
    args.batches_per_val_step = val_every * accum  # must be a multiple of accum
    args.batches_per_save = 0
    args.save_val_logits = False
    args.gradient_accumulation_steps = accum
    args.dataset.dataset_dir = DATA_DIR
    # sessions / dataset_probability_val stay from the official yaml
    args.dataset.num_dataloader_workers = 0
    args.dataset.batch_size = 64 // accum

    n_sess = len(args.dataset.sessions)
    n_val = sum(1 for p in args.dataset.dataset_probability_val if p == 1)
    print(f"sessions: {n_sess} (val days: {n_val}) | optimizer batches: {n_opt_batches}"
          f" | micro-batch {args.dataset.batch_size} x accum {accum}"
          f" | val every {val_every} opt-steps")
    assert n_sess == 45 and n_val == 41, "official 45-session split not found"
    assert 64 % accum == 0, "accum must divide the official batch size 64"
    trainer = BrainToTextDecoder_Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
