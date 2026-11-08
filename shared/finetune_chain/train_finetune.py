# -*- coding: utf-8 -*-
"""Step-3 new-day fine-tune (rapid calibration): init from the EXPANDED
Step-2 checkpoint (41 day layers, target day = identity init at index 40),
train only on the target day's first N trials, val on the target day's full
val set every `val_every` optimizer steps.

Usage: train_finetune.py <ft_root> <num_batches> <val_every> [lr=5e-4]
"""
import os
import sys

sys.path.insert(0, r".\shared\nejm-brain-to-text\model_training")

import torch  # noqa: E402

torch.compile = lambda m, **k: m

from omegaconf import OmegaConf  # noqa: E402
from rnn_trainer import BrainToTextDecoder_Trainer  # noqa: E402


def main():
    """Run the rapid new-day calibration fine-tune via the NEJM trainer.

    Loads ``args.yaml`` from the Step-2 checkpoint directory, overrides
    the training hyperparameters for short-run calibration (step count,
    warmup, learning rate, validation cadence, no saving), initializes
    from the expanded Step-2 best checkpoint, and launches
    ``BrainToTextDecoder_Trainer``.

    Notes
    -----
    Arguments come from ``sys.argv``:
    ``train_finetune.py <ft_root> <num_batches> <val_every> [lr=5e-4]``.
    """
    ft_root, n_batches = sys.argv[1], int(sys.argv[2])
    val_every = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    lr = float(sys.argv[4]) if len(sys.argv) > 4 else 5e-4

    args = OmegaConf.load(os.path.join(ft_root, "checkpoint", "args.yaml"))
    args.gpu_number = "0"
    args.output_dir = ft_root + "_run"
    args.checkpoint_dir = args.output_dir + r"\checkpoint"
    args.num_training_batches = n_batches * 2
    args.lr_decay_steps = n_batches
    args.lr_decay_steps_day = n_batches
    args.lr_warmup_steps = 10
    args.lr_warmup_steps_day = 10
    args.lr_max = lr
    args.lr_max_day = lr
    args.batches_per_train_log = 20
    args.batches_per_val_step = val_every * 2
    args.batches_per_save = 0
    args.save_val_logits = False
    args.gradient_accumulation_steps = 2
    args.init_from_checkpoint = True
    args.init_checkpoint_path = os.path.join(ft_root, "checkpoint", "best_checkpoint")
    args.dataset.num_dataloader_workers = 0
    args.dataset.batch_size = 32

    print(f"finetune: {ft_root} | steps {n_batches} | lr {lr} | "
          f"sessions {len(args.dataset.sessions)}")
    trainer = BrainToTextDecoder_Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
