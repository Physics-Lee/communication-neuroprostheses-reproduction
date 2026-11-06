# -*- coding: utf-8 -*-
"""Export per-trial log-prob npy files + meta.tsv for lg_viterbi (WSL).

Reads <run_dir>/test_logits.hdf5, writes <out>/tr_*.npy
(float32 [T,40]: col0 blank, cols1-39 phonemes) and <out>/meta.tsv.
Run in WSL: python3 export_npy_trials.py ./20230831_willett/data/20230616_willett_BtoT24/run_60k
"""
import os
import sys

import h5py
import numpy as np

OUT = "/root/lm/trials"


DEFAULT_RUN_DIR = ("./20230831_willett"
                   "/data/20230616_willett_BtoT24/run_60k")


def main():
    run_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RUN_DIR
    os.makedirs(OUT, exist_ok=True)
    meta = []
    with h5py.File(os.path.join(run_dir, "test_logits.hdf5"), "r") as f:
        n = 0
        for sess in sorted(f.keys()):
            for key in sorted(f[sess].keys()):
                lp = f[sess][key]["log_probs"][:, :40].astype(np.float32)
                name = f"tr_{n:04d}.npy"
                np.save(os.path.join(OUT, name), lp)
                meta.append(f"{name}\t{sess}\t{key}")
                n += 1
    with open(os.path.join(OUT, "meta.tsv"), "w", encoding="utf-8") as fo:
        fo.write("\n".join(meta) + "\n")
    print(f"exported {n} trials -> {OUT}")


if __name__ == "__main__":
    main()
