# -*- coding: utf-8 -*-
"""Export per-trial npy log-probs + meta.tsv from run_120k/t15_logits.hdf5.

Self-contained regeneration of the wenet_lattice decode inputs for the Card
(T15) 120k val set: walks the hdf5 in insertion order (sessions x trials,
matching dump_t15_logits.py), writes <key>.npy float32 (T, 41) and meta.tsv
(key TAB sentence_label). Run inside WSL python3 so paths are /mnt/d style:

  python3 export_npy_t15.py <hdf5> <out_trials_dir> <out_meta.tsv>
"""
import os
import sys

import h5py
import numpy as np


def main():
    hdf5, out_dir, out_meta = sys.argv[1], sys.argv[2], sys.argv[3]
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    with h5py.File(hdf5, "r") as f, open(out_meta, "w", encoding="utf-8") as m:
        for session in f:  # insertion order == dump order
            for trial in f[session]:
                g = f[f"{session}/{trial}"]
                logp = g["log_probs"][...].astype(np.float32)
                key = f"{n:04d}"
                np.save(os.path.join(out_dir, key + ".npy"), logp)
                m.write(f"{key}\t{g.attrs['sentence_label']}\n")
                n += 1
    print(f"exported {n} trials -> {out_dir} + {out_meta}")


if __name__ == "__main__":
    main()
