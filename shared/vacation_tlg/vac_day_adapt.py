# -*- coding: utf-8 -*-
"""T6: label-free per-session logit mean-shift correction (day adaptation).

Attack: Card late sessions (2024H2-2025) show PER 24-30% but WER 67-91% --
acoustic drift + strong 5-gram flattening posteriors. Simulate the production
system's daily recalibration WITHOUT labels: estimate each session's mean
logit vector over frames (transductive, uses all of the session's trials),
subtract lam * (mu_session - mu_global) from every frame of that session.

Runs inside WSL (numpy only): reads ~/tlv_t15/trials/*.npy + row-ordered
session list from run_120k/val_greedy_per_trial.csv, writes
~/tlv_t15/trials_adapt_l{lam}/NNNN.npy for lam in LAMS.
"""
import csv
import os
import sys

import numpy as np

TRIALS = os.path.expanduser("~/tlv_t15/trials")
GREEDY = ("./shared/data/"
          "20241009_card_BtoT25/run_120k/val_greedy_per_trial.csv")
OUTBASE = os.path.expanduser("~/tlv_t15")
LAMS = [0.25, 0.5, 0.75, 1.0]

def main():
    """Write label-free day-adapted logit trials for each lambda.

    Loads every trial logit array, matches it to a session label by row
    order, computes the global and per-session frame-mean logit vectors,
    then for each adaptation strength saves each trial with
    ``lam * (mu_session - mu_global)`` subtracted.
    """
    sessions = []
    with open(GREEDY, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            sessions.append(r["session"])
    keys = sorted(f[:-4] for f in os.listdir(TRIALS) if f.endswith(".npy"))
    assert len(keys) == len(sessions), f"{len(keys)} trials vs {len(sessions)} meta"

    sum_g = None
    n_g = 0
    sums = {}
    counts = {}
    arrs = {}
    for k, sess in zip(keys, sessions):
        a = np.load(os.path.join(TRIALS, k + ".npy")).astype(np.float32)
        arrs[k] = (a, sess)
        s = a.sum(axis=0)
        sum_g = s if sum_g is None else sum_g + s
        n_g += a.shape[0]
        sums[sess] = sums.get(sess, 0) + s
        counts[sess] = counts.get(sess, 0) + a.shape[0]
    mu_g = sum_g / n_g
    deltas = {sess: sums[sess] / counts[sess] - mu_g for sess in sums}
    print(f"{len(keys)} trials, {len(sums)} sessions, {n_g} frames, "
          f"logit dim {mu_g.shape[0]}")
    for sess in sorted(deltas):
        d = deltas[sess]
        print(f"  {sess}: n={counts[sess]:6d}  |delta|_inf={np.abs(d).max():7.3f}  "
              f"|delta|_2={np.linalg.norm(d):7.3f}")

    for lam in LAMS:
        outdir = os.path.join(OUTBASE, f"trials_adapt_l{lam}")
        os.makedirs(outdir, exist_ok=True)
        for k, (a, sess) in arrs.items():
            np.save(os.path.join(outdir, k + ".npy"), a - lam * deltas[sess])
        print(f"wrote {outdir} ({len(arrs)} trials)")

if __name__ == "__main__":
    main()
