"""Export per-trial CTC log-prob matrices from test_logits.hdf5 to .npy files.

Inputs: the dump_btot24_logits.py output hdf5 (default
./20230831_willett/data/20230616_willett_BtoT24/run_60k/test_logits.hdf5).
Outputs: one (T, 41) float32 .npy per trial under <out>/trials/ plus a
meta.tsv mapping each trial id to its reference sentence, feeding the C++
tlg_viterbi 5-gram decoder in shared/tools.

Usage: export_trials_npy.py <test_logits.hdf5> <output_dir>
"""
import os
import sys

import h5py
import numpy as np

H5 = sys.argv[1] if len(sys.argv) > 1 else (
    "./20230831_willett/"
    "data/20230616_willett_BtoT24/run_60k/test_logits.hdf5")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.expanduser("~/tlv")

os.makedirs(OUT + "/trials", exist_ok=True)
meta = open(OUT + "/meta.tsv", "w", encoding="utf-8")
n = 0
with h5py.File(H5, "r") as f:
    for sess in sorted(f.keys()):
        for key in sorted(f[sess].keys()):
            g = f[sess][key]
            logp = g["log_probs"][:].astype(np.float32)
            k = f"{n:04d}"
            np.save(f"{OUT}/trials/{k}.npy", logp)
            meta.write(f"{k}\t{g.attrs['sentence_label']}\n")
            n += 1
meta.close()
print("exported", n, "trials")
