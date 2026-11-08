# -*- coding: utf-8 -*-
"""Inspect kunz sentenceData .mat files: variables, shapes, binSize, blocks, sentences."""
import glob
import os
import numpy as np
from scipy.io import loadmat


def mat_str(x):
    """Decode one .mat cell into a Python string.

    Handles both storage conventions seen across the release: cells already
    stored as numpy unicode arrays, and cells stored as integer char codes
    (chars <= 0 are padding and are dropped).

    Parameters
    ----------
    x : numpy.ndarray
        Cell content loaded from the .mat file (unicode array or numeric
        char codes).

    Returns
    -------
    str
        The decoded string, with padding code points removed.
    """
    a = np.ravel(x)
    if a.dtype.kind in "US":  # already strings
        return "".join(str(v) for v in a)
    return "".join(chr(int(c)) for c in a if c > 0)

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
        """Decode the i-th sentenceText cell into a Python string.

        Parameters
        ----------
        i : int
            Trial index into the sentenceText array.

        Returns
        -------
        str
            The decoded sentence for trial ``i``.
        """
        a = st[i]
        return "".join(chr(c) if isinstance(c, (int, np.integer)) else str(c) for c in np.ravel(a))
    print(f"   sent[0]={sent(0)!r}")
    print(f"   sent[1]={sent(1)!r}")
    # decoded text present?
    if "decodedSentenceText" in m:
        dt = m["decodedSentenceText"]
        vals = []
        for i in range(dt.shape[0]):
            v = dt[i]
            try:
                s = mat_str(v[0] if isinstance(v, np.ndarray) and v.dtype == object else v)
            except Exception:
                s = repr(v)[:40]
            vals.append(s)
        nonempty = [s for s in vals if s.strip()]
        print(f"   decodedSentenceText nonempty={len(nonempty)}/{dt.shape[0]}; ex={nonempty[:2]}")
    # durations
    if "sentenceDurations" in m:
        sd = m["sentenceDurations"].ravel().astype(float)
        print(f"   durations(bins): min={sd.min():.0f} med={np.median(sd):.0f} max={sd.max():.0f}")
