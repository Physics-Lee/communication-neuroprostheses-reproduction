# -*- coding: utf-8 -*-
"""Dump per-trial CTC log-probs for a kunz ablation arm's val (_eval) trials.

usage: dump_kunz_logits.py <run_dir_name>   (e.g. t15_mixed)
Writes <repo>/data/20250821_kunz/<run>/kunz_logits.hdf5
"""
import os
import sys
import time

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, r".\20240815_card\nejm-brain-to-text\model_training")
from rnn_model import GRUDecoder  # noqa: E402
from evaluate_model_helpers import runSingleDecodingStep  # noqa: E402

REPO = r".\20250821_kunz"
RUN = sys.argv[1]
MODEL_PATH = os.path.join(REPO, "data", "20250821_kunz", "run_" + RUN)
DATA_DIR = os.path.join(REPO, "data", "20250821_kunz", "hdf5")
OUT = os.path.join(MODEL_PATH, "kunz_logits.hdf5")


def main():
    model_args = OmegaConf.load(os.path.join(MODEL_PATH, "checkpoint/args.yaml"))
    ds = model_args["dataset"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = GRUDecoder(
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
    ckpt = torch.load(os.path.join(MODEL_PATH, "checkpoint/best_checkpoint"),
                      weights_only=False, map_location="cpu")
    sd = ckpt["model_state_dict"]
    for key in list(sd.keys()):
        sd[key.replace("module.", "").replace("_orig_mod.", "")] = sd.pop(key)
    model.load_state_dict(sd)
    model.to(device).eval()

    sessions = [s for s in ds["sessions"]
                if s.endswith("_eval")
                and os.path.exists(os.path.join(DATA_DIR, s, "data_val.hdf5"))]
    print(f"{RUN}: {len(sessions)} eval sessions")

    n = 0
    t0 = time.time()
    with h5py.File(OUT, "w") as out:
        for si, session in enumerate(sessions):
            with h5py.File(os.path.join(DATA_DIR, session, "data_val.hdf5"),
                           "r") as f:
                for key in sorted(f.keys()):
                    g = f[key]
                    x = np.expand_dims(g["input_features"][:], axis=0)
                    x = torch.tensor(x, device=device, dtype=torch.bfloat16)
                    logits = runSingleDecodingStep(x, si, model, model_args,
                                                   device)[0]
                    logp = torch.log_softmax(
                        torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
                    grp = out.create_group(f"{session}/{key}")
                    grp.create_dataset("log_probs", data=logp.astype(np.float32),
                                       compression="gzip")
                    grp.attrs["sentence_label"] = g.attrs["sentence_label"]
                    grp.attrs["seq_len"] = int(g.attrs["seq_len"])
                    n += 1
            print(f"[{time.strftime('%H:%M:%S')}] {session} done ({n} trials, "
                  f"{time.time()-t0:.0f}s)", flush=True)
    print(f"saved {OUT} ({n} trials)")


if __name__ == "__main__":
    main()
