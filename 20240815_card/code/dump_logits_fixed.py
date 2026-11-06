# -*- coding: utf-8 -*-
"""Dump per-trial CTC log-probs with CORRECT day indexing (2026-10-16 fix).

BUG BEING FIXED: dump_t15_logits.py / dump_p4_logits.py enumerate the
val-having subset and pass that counter as the day index, but day layers are
positional over the FULL 45-session list whose first entry (t15.2023.08.11)
has no val split -- so every val trial was decoded through the PREVIOUS
day's input layer. The official evaluate_model.py uses
sessions.index(session) (full list); this script matches official semantics.
Impact measured 2026-10-16: logits-greedy PER 08.13 20.9% vs correct 6.4%,
2025.01.10 65.7% vs 24.4% -- all prior WER numbers were computed on the
wrong-layer logits.

Usage: dump_logits_fixed.py <run_dir> [model=gru|bigru]
"""
import os
import sys
import time

import h5py
import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf

sys.path.insert(0, r".\20240815_card\nejm-brain-to-text\model_training")
sys.path.insert(0, r".\20240815_card\code")

MODEL_KIND = sys.argv[2] if len(sys.argv) > 2 else "gru"
if MODEL_KIND == "bigru":
    from calib_models import BiGRUDecoder as Decoder
else:
    from rnn_model import GRUDecoder as Decoder
from evaluate_model_helpers import load_h5py_file, runSingleDecodingStep  # noqa: E402

RUN = sys.argv[1]
DATA_DIR = r".\20240815_card\nejm-brain-to-text\data\hdf5_data_final"
CSV_PATH = r".\20240815_card\nejm-brain-to-text\data\t15_copyTaskData_description.csv"
OUT = os.path.join(RUN, "t15_logits_fixed.hdf5")


def main():
    model_args = OmegaConf.load(os.path.join(RUN, "checkpoint", "args.yaml"))
    ds = model_args["dataset"]
    device = torch.device("cuda:0")
    model = Decoder(
        neural_dim=model_args["model"]["n_input_features"],
        n_units=model_args["model"]["n_units"],
        n_days=len(ds["sessions"]),
        n_classes=ds["n_classes"],
        rnn_dropout=model_args["model"]["rnn_dropout"],
        input_dropout=model_args["model"]["input_network"]["input_layer_dropout"],
        n_layers=model_args["model"]["n_layers"],
        patch_size=model_args["model"]["patch_size"],
        patch_stride=model_args["model"]["patch_stride"],
    )
    ckpt = torch.load(os.path.join(RUN, "checkpoint", "best_checkpoint"),
                      weights_only=False, map_location="cpu")
    sd = ckpt["model_state_dict"]
    for key in list(sd.keys()):
        sd[key.replace("module.", "").replace("_orig_mod.", "")] = sd.pop(key)
    model.load_state_dict(sd)
    model.to(device).eval()
    print(f"checkpoint loaded | val_PER in ckpt: {ckpt.get('val_PER')}")

    b2txt_csv_df = pd.read_csv(CSV_PATH)
    n = 0
    t0 = time.time()
    with h5py.File(OUT, "w") as out:
        for session in ds["sessions"]:
            vp = os.path.join(DATA_DIR, session, "data_val.hdf5")
            if not os.path.exists(vp):
                continue
            day_idx = ds["sessions"].index(session)  # official semantics
            data = load_h5py_file(vp, b2txt_csv_df)
            for trial in range(len(data["neural_features"])):
                x = np.expand_dims(data["neural_features"][trial], axis=0)
                x = torch.tensor(x, device=device, dtype=torch.bfloat16)
                logits = runSingleDecodingStep(x, day_idx, model, model_args, device)[0]
                logp = torch.log_softmax(
                    torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
                g = out.create_group(f"{session}/tr{trial:04d}")
                g.create_dataset("log_probs", data=logp.astype(np.float32),
                                 compression="gzip")
                g.attrs["sentence_label"] = data["sentence_label"][trial]
                g.attrs["corpus"] = data["corpus"][trial]
                sl = int(data["seq_len"][trial])
                g.attrs["seq_len"] = sl
                g.create_dataset("seq_class_ids",
                                 data=np.asarray(data["seq_class_ids"][trial][:sl]))
                n += 1
            print(f"[{time.strftime('%H:%M:%S')}] {session} (day {day_idx}) done "
                  f"({n} trials, {time.time()-t0:.0f}s)", flush=True)
    print(f"saved {OUT} ({n} trials)")


if __name__ == "__main__":
    main()
