# -*- coding: utf-8 -*-
"""Evaluate a BtoT24-trained GRUDecoder checkpoint on the competitionData test partition.

Same greedy-CTC/PER semantics as the official trainer validation:
argmax -> unique_consecutive -> drop blank; aggregate PER = sum(ed)/sum(len).
Usage: eval_btot24.py <run_dir>   (expects <run_dir>/checkpoint/best_checkpoint)
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

DATA_DIR = r".\20230831_willett\data\20230616_willett_BtoT24\hdf5"
CSV_PATH = r".\20230831_willett\data\20230616_willett_BtoT24\btot24_description.csv"
# transforms must match training (gauss smoothing before the model)
MODEL_ARGS = OmegaConf.create({
    "use_amp": True,
    "dataset": {"data_transforms": {
        "smooth_kernel_size": 100, "smooth_data": True, "smooth_kernel_std": 2}},
})


def main():
    run_dir = sys.argv[1]
    ckpt_path = os.path.join(run_dir, "checkpoint", "best_checkpoint")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model config identical to train_btot24.py (official rnn_args.yaml model section)
    model = GRUDecoder(
        neural_dim=512, n_units=768, n_days=24, n_classes=41,
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

    csv_df = pd.read_csv(CSV_PATH)
    sessions = sorted(d for d in os.listdir(DATA_DIR) if d.startswith("t12."))
    rows = []
    total_ed = total_len = n_trials = 0
    t0 = time.time()
    for si, sess in enumerate(sessions):
        vf = os.path.join(DATA_DIR, sess, "data_val.hdf5")
        with h5py.File(vf, "r") as f:
            for key in sorted(f.keys()):
                g = f[key]
                x = torch.tensor(g["input_features"][:][None], device=device,
                                 dtype=torch.bfloat16)
                logits = runSingleDecodingStep(x, si, model, MODEL_ARGS, device)[0]
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
                    "block_num": int(g.attrs["block_num"]),
                    "sentence_label": g.attrs["sentence_label"],
                    "true_phonemes": " ".join(LOGIT_TO_PHONEME[p] for p in true),
                    "pred_phonemes": " ".join(LOGIT_TO_PHONEME[p] for p in pred),
                    "edit_distance": ed, "true_len": seq_len,
                })
        print(f"[{time.strftime('%H:%M:%S')}] {sess} done ({n_trials} trials, "
              f"{time.time()-t0:.0f}s)", flush=True)

    per = total_ed / total_len
    print(f"\n=== BtoT24 test-partition greedy PER: {total_ed}/{total_len} "
          f"= {100*per:.2f}% over {n_trials} trials ===")
    df = pd.DataFrame(rows)
    out_csv = os.path.join(run_dir, "test_partition_greedy_per_trial.csv")
    df.to_csv(out_csv, index=False)
    day = df.groupby("session").apply(
        lambda g: g["edit_distance"].sum() / g["true_len"].sum(),
        include_groups=False).sort_values()
    print("\nbest 5 days:"); print(day.head(5).apply(lambda v: f"{100*v:.2f}%").to_string())
    print("\nworst 5 days:"); print(day.tail(5).apply(lambda v: f"{100*v:.2f}%").to_string())
    print(f"\nsaved: {out_csv}")


if __name__ == "__main__":
    main()
