# -*- coding: utf-8 -*-
"""Inspect one jude RedisMat .mat: variables, shapes, cue structure."""
import numpy as np
from scipy.io import loadmat

mp = r".\20260428_jude\data\20260428_jude\Data\t17\t17.2024.04.02\brainToText\RedisMat\20240402_123849_(0).mat"
m = loadmat(mp)
for k, v in m.items():
    if k.startswith("__"):
        continue
    try:
        print(f"{k:38s} {str(np.shape(v)):18s} {str(v.dtype)[:12]:12s}", end=" ")
        if np.size(v) and np.ndim(v) <= 1 and np.size(v) < 8 and v.dtype.kind in "fiu":
            print(repr(np.ravel(v)[:4])[:60])
        elif v.dtype.kind in "US" or (np.size(v) and np.ndarray and v.dtype == object):
            a = np.ravel(v)
            print(repr(str(a[0])[:50]) if np.size(a) else "")
        else:
            print()
    except Exception as e:
        print("ERR", e)

cue = m["cue"]
print("\ncue entries:")
for i in range(min(5, cue.shape[0])):
    a = np.ravel(cue[i])
    s = "".join(chr(int(c)) for c in a if c > 0) if a.dtype.kind not in "US" else "".join(map(str, a))
    print(f"  [{i}] {s!r}")

tc = m["binned_neural_threshold_crossings"]
sb = m["binned_neural_spike_band_power"]
print(f"\nTX shape={tc.shape} SBP shape={sb.shape}")
print(f"TX[0,:6]={np.ravel(tc[0,:6])} SBP[0,:6]={np.ravel(sb[0,:6])}")
ts = np.ravel(m["binned_neural_nsp_timestamp"])
print(f"nsp ts: n={len(ts)} first={ts[0]:.3e} last={ts[-1]:.3e} step_ns~{(ts[-1]-ts[0])/(len(ts)-1):.3e}")
for fld in ["go_cue_nsp_neural_time", "trial_end_nsp_neural_time"]:
    v = np.ravel(m[fld])
    print(f"{fld}: {v[:3] if len(v)<4 else v[0]}")
for fld in ["trial_paused_by_CNRA", "trial_timed_out"]:
    if fld in m:
        print(f"{fld}: {np.ravel(m[fld])[:6]}")
