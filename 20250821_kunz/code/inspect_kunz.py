# -*- coding: utf-8 -*-
"""Inspect kunz sentenceData .mat files: variables, shapes, binSize, blocks, sentences."""
import glob
import os
import numpy as np
from scipy.io import loadmat

mats = sorted(glob.glob(r".\20250821_kunz\data\20250821_kunz\sentenceData\*.mat"))
for mp in mats:
    m = loadmat(mp, squeeze_me=False, struct_as_record=False)
    keys = [k for k in m if not k.startswith("__")]
    name = os.path.basename(mp)
    n = m["sentenceText"].shape[0]
    bs = m["binSize"].ravel()[0] if "binSize" in m else None
    sf = m["sentenceFeatures"]
    f0 = sf[0, 0]
    print(f"\n== {name}")
    print(f"   n_trials={n} binSize={bs} feat0_shape={np.shape(f0)}")
    if "blockNum" in m:
        bn = m["blockNum"].ravel()
        # per-trial block: from goTrialEpochs start rows
        ge = m["goTrialEpochs"]
        tblocks = [int(bn[int(ge[i,0])-1]) for i in range(min(n, ge.shape[0]))]
        ub = sorted(set(tblocks))
        print(f"   goTrialEpochs={ge.shape} blocks_seen={ub[:12]}{'...' if len(ub)>12 else ''} n_blocks={len(ub)}")
    if "chanSets" in m:
        cs = m["chanSets"]
        try:
            sizes = [len(x[0] if hasattr(x,'__len__') else []) for x in cs.ravel()]
        except Exception:
            sizes = "?"
        names = []
        cn = m.get("chanSetNames")
        if cn is not None:
            names = [str(x[0]) for x in cn.ravel()]
        print(f"   chanSets sizes={sizes} names={names}")
    st = m["sentenceText"]
    def sent(i):
        a = st[i]
        return "".join(chr(c) if isinstance(c, (int, np.integer)) else str(c) for c in np.ravel(a))
    print(f"   sent[0]={sent(0)!r}")
    print(f"   sent[1]={sent(1)!r}")
    # decoded text present?
    if "decodedSentenceText" in m:
        dt = m["decodedSentenceText"]
        nonempty = sum(1 for i in range(dt.shape[0]) if np.size(dt[i]) and (np.size(dt[i]) > 1))
        print(f"   decodedSentenceText nonempty≈{nonempty}/{dt.shape[0]}  ex={sent(0) if False else None}")
    # durations
    if "sentenceDurations" in m:
        sd = m["sentenceDurations"].ravel().astype(float)
        print(f"   durations(bins): min={sd.min():.0f} med={np.median(sd):.0f} max={sd.max():.0f}")
