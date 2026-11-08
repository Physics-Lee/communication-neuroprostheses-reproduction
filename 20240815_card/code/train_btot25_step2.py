# -*- coding: utf-8 -*-
"""Step 2 of the rapid-calibration plan: 40-day baseline excluding the last 5
val days (see notes/20260903_rapid_calibration_plan.html).

Same protocol as train_btot25.py (official rnn_args.yaml, micro-batch 32 x
accum 2) but sessions = 40 days (the 5 excluded days below are removed from
BOTH train and val sides). The 5 excluded days become the "new days" for
tomorrow's N-trial fine-tuning via init_from_checkpoint.

Usage: train_btot25_step2.py <optimizer_batches> <output_root> [val_every=2000] [accum=2]
"""
import os
import sys

sys.path.insert(0, r".\20240815_card\nejm-brain-to-text\model_training")

import torch  # noqa: E402

torch.compile = lambda m, **k: m  # skip torch.compile on native Windows

from omegaconf import OmegaConf  # noqa: E402
from rnn_trainer import BrainToTextDecoder_Trainer  # noqa: E402

DATA_DIR = r".\20240815_card\nejm-brain-to-text\data\hdf5_data_final"
EXCLUDE = [
    "t15.2025.01.12",
    "t15.2025.03.14",
    "t15.2025.03.16",
    "t15.2025.03.30",
    "t15.2025.04.13",
]


def main():
    """Train the 40-day baseline GRU decoder (last 5 val days excluded).

    Follows the train_btot25.py protocol (official rnn_args.yaml,
    micro-batch 32 x accum 2) but removes the five EXCLUDE sessions from
    both the train and val sides before running
    ``BrainToTextDecoder_Trainer.train()``; those days are reserved as
    "new days" for the subsequent N-trial fine-tuning.

    Returns
    -------
    None
        Checkpoints and logs are written under ``out_root``; the run summary
        is printed to stdout.

    Raises
    ------
    AssertionError
        If the filtered session list is not 40 sessions with 36 val days.
    """
    n_opt_batches, out_root = int(sys.argv[1]), sys.argv[2]
    val_every = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    accum = int(sys.argv[4]) if len(sys.argv) > 4 else 2

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

    all_sessions = list(args.dataset.sessions)
    all_prob = list(args.dataset.dataset_probability_val)
    kept = [(s, p) for s, p in zip(all_sessions, all_prob) if s not in EXCLUDE]
    args.dataset.sessions = [s for s, _ in kept]
    args.dataset.dataset_probability_val = [p for _, p in kept]

    n_val = sum(1 for p in args.dataset.dataset_probability_val if p == 1)
    print(f"sessions: {len(args.dataset.sessions)} (val days: {n_val}) "
          f"| excluded: {EXCLUDE} | optimizer batches: {n_opt_batches}"
          f" | micro-batch {args.dataset.batch_size} x accum {accum}")
    assert len(args.dataset.sessions) == 40 and n_val == 36, "expected 40 sessions / 36 val days"
    trainer = BrainToTextDecoder_Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
