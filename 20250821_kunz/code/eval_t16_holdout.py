# -*- coding: utf-8 -*-
"""Evaluate one T16 attempted-holdout fold checkpoint on its held-out block.

Same greedy-CTC/PER semantics as eval_kunz.py:
argmax -> unique_consecutive -> drop blank; aggregate PER = sum(ed)/sum(len).

Usage: eval_t16_holdout.py <run_dir> <fold 1|2|3>   (expects <run_dir>/checkpoint/best_checkpoint)
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

ROOT = r".\20250821_kunz\data\20250821_kunz\hdf5_t16holdout"
FOLD_VAL_BLK = {1: 1, 2: 4, 3: 6}  # fold -> held-out (val) block
MODEL_ARGS = OmegaConf.create({
    "use_amp": True,
    "dataset": {"data_transforms": {
        "smooth_kernel_size": 100, "smooth_data": True, "smooth_kernel_std": 2}},
})


def main():
    run_dir, fold = sys.argv[1], int(sys.argv[2])
    ckpt_path = os.path.join(run_dir, "checkpoint", "best_checkpoint")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_dir = os.path.join(ROOT, f"fold{fold}")
    sessions = sorted(d for d in os.listdir(data_dir)
                      if d.startswith("t16.") and f"blk{FOLD_VAL_BLK[fold]}val" in d)
    assert len(sessions) == 1, f"fold {fold} val session not found: {sessions}"
    val_sess = sessions[0]
    all_sessions = sorted(d for d in os.listdir(data_dir) if d.startswith("t16."))
    day_idx = all_sessions.index(val_sess)  # must match train-time day indexing

    model = GRUDecoder(
        neural_dim=512, n_units=768, n_days=len(all_sessions), n_classes=41,
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
    with h5py.File(os.path.join(data_dir, val_sess, "data_val.hdf5"), "r") as f:
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
                "session": val_sess,
                "fold": fold,
                "block_num": int(g.attrs["block_num"]),
                "vocab": "50words",
                "sentence_label": g.attrs["sentence_label"],
                "pred_phonemes": " ".join(LOGIT_TO_PHONEME[p] for p in pred),
                "edit_distance": ed, "true_len": seq_len,
            })
    per = total_ed / total_len
    print(f"\n=== T16 attempted holdout fold {fold} ({val_sess}): greedy PER "
          f"{total_ed}/{total_len} = {100*per:.2f}% over {n_trials} trials === "
          f"({time.time()-t0:.0f}s)")
    df = pd.DataFrame(rows)
    out_csv = os.path.join(run_dir, "val_greedy_per_trial.csv")
    df.to_csv(out_csv, index=False)
    print(f"saved: {out_csv}")


if __name__ == "__main__":
    main()
