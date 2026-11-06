# -*- coding: utf-8 -*-
"""Ensemble dump: average log-probs of P4 (45-sess) and T2a (44-sess remapped
into 45 slots, day 39 = copy-of-38), written as a logits hdf5 in the same
format as dump_logits_fixed.py for the decode/rescore chain.

Usage: python dump_logits_ensemble.py
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
from evaluate_model_helpers import load_h5py_file, runSingleDecodingStep  # noqa: E402
from calib_t2a_eval import load_45slot  # noqa: E402
from calib_online_sim import build_model, P4  # noqa: E402

DATA_DIR = r".\20240815_card\nejm-brain-to-text\data\hdf5_data_final"
CSV_PATH = r".\20240815_card\nejm-brain-to-text\data\t15_copyTaskData_description.csv"
OUT = os.path.join(os.path.dirname(P4), "run_ens_p4_t2a", "t15_logits_fixed.hdf5")


def main():
    device = torch.device("cuda:0")
    args = OmegaConf.load(os.path.join(P4, "checkpoint", "args.yaml"))
    sessions = list(args["dataset"]["sessions"])
    m1 = build_model(args, sessions, device)          # P4
    m2, _, _ = load_45slot(device)                     # T2a remapped, slot39=eye
    with torch.no_grad():                              # slot39 -> copy of 38
        m2.day_weights[39].copy_(m2.day_weights[38])
        m2.day_biases[39].copy_(m2.day_biases[38])
    for m in (m1, m2):
        m.eval()

    b2txt = pd.read_csv(CSV_PATH)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = 0
    t0 = time.time()
    with h5py.File(OUT, "w") as out:
        for session in sessions:
            vp = os.path.join(DATA_DIR, session, "data_val.hdf5")
            if not os.path.exists(vp):
                continue
            di = sessions.index(session)
            data = load_h5py_file(vp, b2txt)
            for trial in range(len(data["neural_features"])):
                x = np.expand_dims(data["neural_features"][trial], axis=0)
                x = torch.tensor(x, device=device, dtype=torch.bfloat16)
                lp = None
                for m in (m1, m2):
                    lg = runSingleDecodingStep(x, di, m, args, device)[0]
                    one = torch.log_softmax(torch.tensor(lg, dtype=torch.float32), dim=-1)
                    lp = one if lp is None else lp + one
                lp = (lp / 2).numpy()
                g = out.create_group(f"{session}/tr{trial:04d}")
                g.create_dataset("log_probs", data=lp.astype(np.float32), compression="gzip")
                g.attrs["sentence_label"] = data["sentence_label"][trial]
                g.attrs["corpus"] = data["corpus"][trial]
                sl = int(data["seq_len"][trial])
                g.attrs["seq_len"] = sl
                g.create_dataset("seq_class_ids",
                                 data=np.asarray(data["seq_class_ids"][trial][:sl]))
                n += 1
            print(f"[{time.strftime('%H:%M:%S')}] {session} done ({n}, {time.time()-t0:.0f}s)", flush=True)
    print(f"saved {OUT} ({n})")


if __name__ == "__main__":
    main()
