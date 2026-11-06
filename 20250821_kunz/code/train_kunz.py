# -*- coding: utf-8 -*-
"""Train the nejm-brain-to-text GRU decoder on converted Kunz inner-speech data.

One model per participant. Training arms:
- mixed    : open-loop attempted + imagined files of that participant
- imagined : open-loop imagined files only
- attempted: open-loop attempted files only
Validation is always the closed-loop *_eval (imagined) sessions.

Usage: train_kunz.py <t12|t15|t16> <mixed|imagined|attempted> <num_batches> <out_root>
"""
import glob
import os
import sys

sys.path.insert(0, r".\20240815_card\nejm-brain-to-text\model_training")

import torch  # noqa: E402

torch.compile = lambda m, **k: m  # skip torch.compile on native Windows

from omegaconf import OmegaConf  # noqa: E402
from rnn_trainer import BrainToTextDecoder_Trainer  # noqa: E402

DATA_DIR = r".\20250821_kunz\data\20250821_kunz\hdf5"


def main():
    subj, arm, n_batches, out_root = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    val_every = int(sys.argv[5]) if len(sys.argv) > 5 else 500

    # all sessions of this participant (train files + *_eval val files)
    all_sessions = sorted(os.path.basename(p)
                          for p in glob.glob(os.path.join(DATA_DIR, f"{subj}.*")))
    train_sessions = [s for s in all_sessions
                      if not s.endswith("_eval")
                      and (arm == "mixed"
                           or (arm == "imagined") == ("_imagined_" in s))]
    assert train_sessions, f"no train sessions for {subj}/{arm}"
    # *_eval sessions stay in the list: they carry the (shared) validation data
    # and keep day indexing well-defined; the arm selection must actually reach
    # args.dataset.sessions or the trainer would load every session's
    # data_train.hdf5 and all arms would train on identical data
    sessions = sorted(train_sessions + [s for s in all_sessions
                                        if s.endswith("_eval")])
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
    args.dataset.dataset_dir = DATA_DIR
    args.dataset.sessions = list(sessions)
    args.dataset.dataset_probability_val = [1] * len(sessions)
    args.dataset.num_dataloader_workers = 0
    args.dataset.days_per_batch = days_per_batch

    print(f"[{subj}/{arm}] train sessions: {train_sessions}")
    print(f"[{subj}/{arm}] model sessions: {sessions}")
    print(f"days_per_batch={days_per_batch} | batches={n_batches} | val every {val_every}")
    trainer = BrainToTextDecoder_Trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
