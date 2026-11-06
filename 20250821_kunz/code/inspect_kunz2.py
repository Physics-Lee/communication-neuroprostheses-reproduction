# -*- coding: utf-8 -*-
import numpy as np
from scipy.io import loadmat

def sent_str(x):
    a = np.ravel(x)
    if a.dtype.kind in "US":  # already strings
        return "".join(str(v) for v in a)
    return "".join(chr(int(c)) for c in a if c > 0)

for mp in [r".\20250821_kunz\data\20250821_kunz\sentenceData\t16.2024.12.18_imagined_125kwords.mat",
           r".\20250821_kunz\data\20250821_kunz\sentenceData\t16.2024.12.18_imagined_125kwords_eval.mat",
           r".\20250821_kunz\data\20250821_kunz\sentenceData\t16.2024.04.29_imagined_50words_eval.mat"]:
    m = loadmat(mp)
    n = m["sentenceText"].shape[0]
    print(mp.split("\\")[-1], "n=", n, "binSize=", float(m["binSize"].ravel()[0]))
    ge = m["goTrialEpochs"]; bn = m["blockNum"].ravel()
    print("  blocks:", sorted(set(int(bn[int(ge[i,0])-1]) for i in range(n))))
    if "decodedSentenceText" in m:
        dt = m["decodedSentenceText"]
        vals = []
        for i in range(dt.shape[0]):
            v = dt[i]
            try:
                s = sent_str(v[0] if isinstance(v, np.ndarray) and v.dtype == object else v)
            except Exception:
                s = repr(v)[:40]
            vals.append(s)
        nonempty = [s for s in vals if s.strip()]
        print(f"  decoded nonempty={len(nonempty)}/{n}; ex={nonempty[:2]}")
