# -*- coding: utf-8 -*-
"""Train the nejm-brain-to-text GRU decoder on converted BtoT24 data.

Adapts the official rnn_args.yaml programmatically:
- dataset_dir/sessions -> .\20230831_willett\data\20230616_willett_BtoT24\hdf5 (24 t12 days)
- num_dataloader_workers 0 (Windows), gpu '0'
- lr decay / cosine horizon scaled to the actual run length
- torch.compile disabled (native-Windows compatibility)

Usage: train_btot24.py <num_batches> <output_root> [batches_per_val]
                       [init_checkpoint] [lr_decay_steps]

init_checkpoint: full resume (model + optimizer + scheduler + best val PER,
see rnn_trainer.load_model_checkpoint). Pair with lr_decay_steps so the
restored scheduler.last_epoch lands where the intended schedule ends, e.g.
resume from a 60k-run best at scheduler epoch 56001 with lr_decay_steps=120000
and num_batches=64000 to finish the official 120k cosine exactly.
"""
import glob
import os
import sys

sys.path.insert(0, r".\20240815_card\nejm-brain-to-text\model_training")

import torch  # noqa: E402

torch.compile = lambda m, **k: m  # skip torch.compile on native Windows

from omegaconf import OmegaConf  # noqa: E402
from rnn_trainer import BrainToTextDecoder_Trainer  # noqa: E402


def main():
    n_batches = int(sys.argv[1])
    out_root = sys.argv[2]
    val_every = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    init_ckpt = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else None
    lr_decay = int(sys.argv[5]) if len(sys.argv) > 5 else None

    # glob matches session DIRECTORIES (names are already bare, no extension)
    sessions = sorted(os.path.basename(p)
                      for p in glob.glob(r".\20230831_willett\data\20230616_willett_BtoT24\hdf5\t12.*"))
    assert len(sessions) == 24 and len(set(sessions)) == 24, \
        f"expected 24 unique sessions, got {len(sessions)} ({len(set(sessions))} unique)"

    args = OmegaConf.load(r".\20240815_card\nejm-brain-to-text\model_training\rnn_args.yaml")
    args.gpu_number = "0"
    args.output_dir = out_root
    args.checkpoint_dir = os.path.join(out_root, "checkpoint")
    args.num_training_batches = n_batches
    args.lr_decay_steps = lr_decay or n_batches
    args.lr_decay_steps_day = lr_decay or n_batches
    args.batches_per_train_log = 50
    args.batches_per_val_step = val_every
    args.batches_per_save = 0
    args.save_val_logits = False
    args.save_final_model = True
    args.dataset.dataset_dir = r".\20230831_willett\data\20230616_willett_BtoT24\hdf5"
    args.dataset.sessions = list(sessions)
    args.dataset.dataset_probability_val = [1] * len(sessions)
    args.dataset.num_dataloader_workers = 0
    if init_ckpt:
        args.init_from_checkpoint = True
        args.init_checkpoint_path = init_ckpt

    print(f"sessions: {len(sessions)} | batches: {n_batches} | val every {val_every} "
          f"| lr_decay_steps {args.lr_decay_steps} | init: {init_ckpt}")
    trainer = BrainToTextDecoder_Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
