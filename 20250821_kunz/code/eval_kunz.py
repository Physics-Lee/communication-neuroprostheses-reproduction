# -*- coding: utf-8 -*-
"""Evaluate a Kunz-trained GRUDecoder checkpoint on the closed-loop imagined *_eval sessions.

Same greedy-CTC/PER semantics as eval_btot24.py:
argmax -> unique_consecutive -> drop blank; aggregate PER = sum(ed)/sum(len).
Also reports per-session and per-vocab breakdowns.

Usage: eval_kunz.py <run_dir> <t12|t15|t16>   (expects <run_dir>/checkpoint/best_checkpoint)
"""
import os
import sys
import time

import editdistance
import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, r".\20240815_card\nejm-brain-to-text\model_training")
from rnn_model import GRUDecoder  # noqa: E402
from evaluate_model_helpers import LOGIT_TO_PHONEME, runSingleDecodingStep  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

DATA_DIR = r".\20250821_kunz\data\20250821_kunz\hdf5"
MODEL_ARGS = OmegaConf.create({
    "use_amp": True,
    "dataset": {"data_transforms": {
        "smooth_kernel_size": 100, "smooth_data": True, "smooth_kernel_std": 2}},
})


def main():
    run_dir, subj = sys.argv[1], sys.argv[2]
    ckpt_path = os.path.join(run_dir, "checkpoint", "best_checkpoint")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # session list / day indexing must match training exactly; the run's saved
    # args.yaml is the authoritative record (arm runs use a session subset)
    run_args = OmegaConf.load(os.path.join(run_dir, "checkpoint", "args.yaml"))
    all_sessions = list(run_args.dataset.sessions)
    assert run_args.dataset.dataset_dir == DATA_DIR, \
        f"run was trained on {run_args.dataset.dataset_dir}, expected {DATA_DIR}"
    sessions = [s for s in all_sessions if s.endswith("_eval")]
    n_days = len(all_sessions)
    model = GRUDecoder(
        neural_dim=512, n_units=768, n_days=n_days, n_classes=41,
        rnn_dropout=0.4, input_dropout=0.2, n_layers=5,
        patch_size=14, patch_stride=4,
    )
    ckpt = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    sd = ckpt["model_state_dict"]
    for k in list(sd.keys()):
        sd[k.replace("module.", "").replace("_orig_mod.", "")] = sd.pop(k)
    model.load_state_dict(sd)
    model.to(device).eval()
    print(f"loaded {ckpt_path} | ckpt val_PER = {ckpt.get('val_PER')}")

    rows = []
    total_ed = total_len = n_trials = 0
    t0 = time.time()
    for sess in sessions:
        day_idx = all_sessions.index(sess)  # must match train-time day indexing
        with h5py.File(os.path.join(DATA_DIR, sess, "data_val.hdf5"), "r") as f:
            for key in sorted(f.keys()):
                g = f[key]
                x = torch.tensor(g["input_features"][:][None], device=device,
                                 dtype=torch.bfloat16)
                logits = runSingleDecodingStep(x, day_idx, model, MODEL_ARGS, device)[0]
                pred = np.argmax(logits, axis=-1)
                pred = torch.unique_consecutive(torch.tensor(pred), dim=-1).numpy()
                pred = np.array([i for i in pred if i != 0])
                seq_len = int(g.attrs["seq_len"])
                true = np.asarray(g["seq_class_ids"][:][:seq_len])
                ed = editdistance.eval(pred.tolist(), true.tolist())
                total_ed += ed
                total_len += seq_len
                n_trials += 1
                rows.append({
                    "session": sess,
                    "vocab": "50words" if "50words" in sess else "125kwords",
                    "sentence_label": g.attrs["sentence_label"],
                    "pred_phonemes": " ".join(LOGIT_TO_PHONEME[p] for p in pred),
                    "edit_distance": ed, "true_len": seq_len,
                })
        print(f"[{time.strftime('%H:%M:%S')}] {sess} done ({n_trials} trials, "
              f"{time.time()-t0:.0f}s)", flush=True)

    per = total_ed / total_len
    print(f"\n=== Kunz {subj} closed-loop imagined greedy PER: {total_ed}/{total_len} "
          f"= {100*per:.2f}% over {n_trials} trials ===")
    df = pd.DataFrame(rows)
    out_csv = os.path.join(run_dir, "val_greedy_per_trial.csv")
    df.to_csv(out_csv, index=False)
    per_sess = df.groupby("session").apply(
        lambda g: g["edit_distance"].sum() / g["true_len"].sum(),
        include_groups=False).sort_values()
    print("\nper session:")
    print(per_sess.apply(lambda v: f"{100*v:.2f}%").to_string())
    per_vocab = df.groupby("vocab").apply(
        lambda g: g["edit_distance"].sum() / g["true_len"].sum(),
        include_groups=False)
    print("\nper vocab:")
    print(per_vocab.apply(lambda v: f"{100*v:.2f}%").to_string())
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    main()
