# -*- coding: utf-8 -*-
"""Dump per-trial CTC log-probs for the BtoT25 (T15) 120k-run val set.

Mirrors eval_val_per.py / eval_btot25_fromscratch.py exactly: same args.yaml
model construction, same load_h5py_file iteration, same runSingleDecodingStep
AMP path. Saves log_softmax(logits) per trial + refs to run_120k/t15_logits.hdf5.
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
from rnn_model import GRUDecoder  # noqa: E402
from evaluate_model_helpers import load_h5py_file, runSingleDecodingStep  # noqa: E402

MODEL_PATH = r".\20240815_card\data\20241009_card_BtoT25\run_120k"
DATA_DIR = r".\20240815_card\nejm-brain-to-text\data\hdf5_data_final"
CSV_PATH = r".\20240815_card\nejm-brain-to-text\data\t15_copyTaskData_description.csv"
OUT = r".\20240815_card\data\20241009_card_BtoT25\run_120k\t15_logits.hdf5"


def main():
    """Decode the BtoT25 120k-run val set and dump per-trial CTC log-probs.

    Loads the run_120k best checkpoint, mirrors the eval_val_per.py /
    eval_btot25_fromscratch.py decoding path over every session with a
    data_val.hdf5 split, converts the logits to log-probs, and writes them to
    ``OUT`` grouped as ``<session>/tr<NNNN>``. Note: the day index passed to
    the decoder is the position within the val-having subset (legacy
    semantics; corrected later in dump_logits_fixed.py).

    Returns
    -------
    None
        Results are written to ``OUT``; progress is printed to stdout.
    """
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
    print(f"checkpoint loaded | val_PER in ckpt: {ckpt.get('val_PER')}")

    b2txt_csv_df = pd.read_csv(CSV_PATH)
    sessions = [s for s in ds["sessions"]
                if os.path.exists(os.path.join(DATA_DIR, s, "data_val.hdf5"))]
    print(f"{len(sessions)} sessions with val")

    n = 0
    t0 = time.time()
    with h5py.File(OUT, "w") as out:
        for si, session in enumerate(sessions):
            data = load_h5py_file(
                os.path.join(DATA_DIR, session, "data_val.hdf5"), b2txt_csv_df)
            for trial in range(len(data["neural_features"])):
                x = np.expand_dims(data["neural_features"][trial], axis=0)
                x = torch.tensor(x, device=device, dtype=torch.bfloat16)
                logits = runSingleDecodingStep(x, si, model, model_args, device)[0]
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
            print(f"[{time.strftime('%H:%M:%S')}] {session} done ({n} trials, "
                  f"{time.time()-t0:.0f}s)", flush=True)
    print(f"saved {OUT} ({n} trials)")


if __name__ == "__main__":
    main()
