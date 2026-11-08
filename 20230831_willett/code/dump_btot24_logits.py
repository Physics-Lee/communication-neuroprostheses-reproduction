# -*- coding: utf-8 -*-
"""Dump per-trial CTC log-probs for the BtoT24 test partition.

Mirrors eval_btot24.py exactly (same model config, same session/key iteration,
same runSingleDecodingStep AMP path) but saves log_softmax(logits) per trial to
an h5 for downstream word-level beam decoding (pyctcdecode / WFST).

Usage: dump_btot24_logits.py <run_dir> [out.h5]
"""
import os
import sys
import time

import h5py
import numpy as np
import torch

sys.path.insert(0, r".\20240815_card\nejm-brain-to-text\model_training")
from rnn_model import GRUDecoder  # noqa: E402
from evaluate_model_helpers import runSingleDecodingStep  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

DATA_DIR = r".\20230831_willett\data\20230616_willett_BtoT24\hdf5"
MODEL_ARGS = OmegaConf.create({
    "use_amp": True,
    "dataset": {"data_transforms": {
        "smooth_kernel_size": 100, "smooth_data": True, "smooth_kernel_std": 2}},
})


def main():
    """Run the trained GRUDecoder over the test partition and save log-probs.

    Rebuilds the 5-layer GRU exactly as eval_btot24.py, loads
    <run_dir>/checkpoint/best_checkpoint, iterates every session/trial of
    data_val.hdf5 in sorted order under the AMP decoding path, and stores
    log_softmax(logits) per trial (plus seq_len, block_num,
    sentence_label and seq_class_ids metadata) to an hdf5 file for
    downstream word-level beam decoding.
    """
    run_dir = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        run_dir, "test_logits.hdf5")
    ckpt_path = os.path.join(run_dir, "checkpoint", "best_checkpoint")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

    sessions = sorted(d for d in os.listdir(DATA_DIR) if d.startswith("t12."))
    n_trials = 0
    t0 = time.time()
    with h5py.File(out_path, "w") as out:
        for si, sess in enumerate(sessions):
            vf = os.path.join(DATA_DIR, sess, "data_val.hdf5")
            with h5py.File(vf, "r") as f:
                for key in sorted(f.keys()):
                    g = f[key]
                    x = torch.tensor(g["input_features"][:][None], device=device,
                                     dtype=torch.bfloat16)
                    logits = runSingleDecodingStep(x, si, model, MODEL_ARGS, device)[0]
                    logp = torch.log_softmax(
                        torch.tensor(logits, dtype=torch.float32), dim=-1).numpy()
                    grp = out.create_group(f"{sess}/{key}")
                    grp.create_dataset("log_probs", data=logp.astype(np.float32),
                                       compression="gzip")
                    for attr in ("seq_len", "block_num", "sentence_label"):
                        grp.attrs[attr] = g.attrs[attr]
                    grp.create_dataset("seq_class_ids",
                                       data=np.asarray(g["seq_class_ids"][:]
                                                       [:int(g.attrs["seq_len"])]))
                    n_trials += 1
            print(f"[{time.strftime('%H:%M:%S')}] {sess} done ({n_trials} trials, "
                  f"{time.time()-t0:.0f}s)", flush=True)
    print(f"saved {out_path} ({n_trials} trials)")


if __name__ == "__main__":
    main()
