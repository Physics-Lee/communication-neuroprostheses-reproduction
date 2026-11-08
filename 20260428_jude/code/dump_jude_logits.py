# -*- coding: utf-8 -*-
"""Dump per-trial CTC log-probs for the jude (T17) run_25k val partition.

Same model construction/AMP path as dump_t15_logits.py but reads jude's
hdf5 directly (keys: input_features / seq_class_ids / transcription;
attrs: sentence_label, seq_len, ...).

Run with the Brain2Qwerty venv python (torch+cuda).
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

MODEL_PATH = r".\20260428_jude\data\20260428_jude\run_25k"
DATA_DIR = r".\20260428_jude\data\20260428_jude\hdf5"
OUT = r".\20260428_jude\data\20260428_jude\run_25k\jude_logits.hdf5"


def main():
    """Dump per-trial CTC log-probs for every session that has a val split.

    Rebuilds the GRUDecoder from the run's ``checkpoint/args.yaml``, loads
    the best checkpoint, decodes each ``data_val.hdf5`` trial through
    ``runSingleDecodingStep`` (same AMP path as eval), and stores the
    log-softmax probabilities plus labels under ``<session>/<trial>`` groups
    in one output hdf5 (OUT).

    Returns
    -------
    None
        Writes the logits hdf5 file and prints progress per session.
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

    sessions = [s for s in ds["sessions"]
                if os.path.exists(os.path.join(DATA_DIR, s, "data_val.hdf5"))]
    print(f"{len(sessions)} sessions with val")

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
                    sl = int(g.attrs["seq_len"])
                    grp.attrs["seq_len"] = sl
                    grp.create_dataset("seq_class_ids",
                                       data=np.asarray(g["seq_class_ids"][:sl]))
                    n += 1
            print(f"[{time.strftime('%H:%M:%S')}] {session} done ({n} trials, "
                  f"{time.time()-t0:.0f}s)", flush=True)
    print(f"saved {OUT} ({n} trials)")


if __name__ == "__main__":
    main()
