# -*- coding: utf-8 -*-
"""Scan all jude RedisMat .mat files: block number, trials, cue samples, feature shape."""
import glob
import os
import numpy as np
from scipy.io import loadmat

rows = []
for mp in sorted(glob.glob(r".\20260428_jude\data\20260428_jude\Data\t17\*\brainToText\RedisMat\*.mat")):
    m = loadmat(mp, variable_names=[
        "block_number", "cue", "binned_neural_threshold_crossings",
        "session_name", "trial_paused_by_CNRA", "trial_timed_out"])
    sess = str(np.ravel(m["session_name"])[0])
    blk = int(np.ravel(m["block_number"])[0])
    cue = np.ravel(m["cue"])
    cues = [str(c).strip() for c in cue]
    tx = m["binned_neural_threshold_crossings"]
    paused = int(np.ravel(m["trial_paused_by_CNRA"]).sum()) if "trial_paused_by_CNRA" in m else -1
    timed = int(np.ravel(m["trial_timed_out"]).sum()) if "trial_timed_out" in m else -1
    # heuristic: sentence block if any cue contains a space
    kind = "SENT" if any(" " in c for c in cues) else "single"
    rows.append((sess, os.path.basename(mp), blk, len(cues), tx.shape, kind,
                 paused, timed, cues[0][:48]))

rows.sort(key=lambda r: (r[0], r[2]))
print(f"{'session':18s} {'file':26s} {'blk':>3s} {'n':>3s} {'TXshape':>14s} {'kind':6s} {'paus':>4s} {'timo':>4s} cue0")
for r in rows:
    print(f"{r[0]:18s} {r[1]:26s} {r[2]:3d} {r[3]:3d} {str(r[4]):>14s} {r[5]:6s} {r[6]:4d} {r[7]:4d} {r[8]!r}")

sent = [r for r in rows if r[5] == "SENT"]
print(f"\ntotal mats={len(rows)} sentence-blocks={len(sent)} "
      f"sent-trials={sum(r[3] for r in sent)}")
