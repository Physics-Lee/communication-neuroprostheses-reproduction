# -*- coding: utf-8 -*-
"""Step-3 mechanism smoke (notes/20260903_rapid_calibration_plan.html §3.1):
verify the init_from_checkpoint path loads and continues training sensibly.

Uses the 45-day run_120k best_checkpoint on the SAME 45-session config (the
strict load_state_dict in load_model_checkpoint requires matching day count;
expanding days for the real new-day fine-tune is tomorrow's patch).

Success criterion: first logged train loss is LOW (model already converged,
~0-30, not 700+ from-scratch start) and still decreases.

Usage: smoke_init_checkpoint.py <num_batches> <out_root>
"""
import os
import sys

sys.path.insert(0, r".\20240815_card\nejm-brain-to-text\model_training")

import torch  # noqa: E402

torch.compile = lambda m, **k: m

from omegaconf import OmegaConf  # noqa: E402
from rnn_trainer import BrainToTextDecoder_Trainer  # noqa: E402


def main():
    n_batches, out_root = int(sys.argv[1]), sys.argv[2]

    args = OmegaConf.load(r".\20240815_card\nejm-brain-to-text\model_training\rnn_args.yaml")
    args.gpu_number = "0"
    args.output_dir = out_root
    args.checkpoint_dir = os.path.join(out_root, "checkpoint")
    # short finetune-style horizon: small warmup, fast decay
    args.num_training_batches = n_batches * 2          # micro-batches (accum 2)
    args.lr_decay_steps = n_batches
    args.lr_decay_steps_day = n_batches
    args.lr_warmup_steps = 10
    args.lr_warmup_steps_day = 10
    args.lr_max = 0.0005                                # finetune-scale LR
    args.lr_max_day = 0.0005
    args.batches_per_train_log = 50
    args.batches_per_val_step = 1000 * 2
    args.batches_per_save = 0
    args.save_val_logits = False
    args.gradient_accumulation_steps = 2
    args.init_from_checkpoint = True
    args.init_checkpoint_path = r".\20240815_card\data\20241009_card_BtoT25\run_120k\checkpoint\best_checkpoint"
    args.dataset.dataset_dir = r".\20240815_card\nejm-brain-to-text\data\hdf5_data_final"
    args.dataset.num_dataloader_workers = 0
    args.dataset.batch_size = 32

    print(f"smoke: init_from {args.init_checkpoint_path} | batches {n_batches}")
    trainer = BrainToTextDecoder_Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
