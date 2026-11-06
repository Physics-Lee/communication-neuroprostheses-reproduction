# -*- coding: utf-8 -*-
"""P0.1 of the calibration plan (20261014): per-day 5-gram and +OPT WER for the
six 2025 follow-up sessions, from the P4 endpoint products.

Row->session mapping: p4 tsvs are in logits-hdf5 order (yaml sessions minus
train-only days, trials sorted), which is also eval_val_per.py's CSV order.
Verified here by matching per-session trial counts against the logits hdf5.

Outputs (stdout + result/logs/calib_20261014_p0_wer.txt):
  - per-day WER for 5-gram (p4_final_full.tsv) and +OPT (p4_opt_fx.tsv)
  - pooled 6-day and "last-5" variants to pin the legacy "last 5=62.x" scope

Usage: python calib_p0_wer_baseline.py
"""
import os
import re
import sys

import numpy as np
import pandas as pd

REPO = r".\20240815_card"
P4 = os.path.join(REPO, r"data\20241009_card_BtoT25\run_p4_20261013")
LOGITS = os.path.join(P4, "t15_logits.hdf5")

SESSIONS_2025 = ["t15.2025.01.10", "t15.2025.01.12", "t15.2025.03.14",
                 "t15.2025.03.16", "t15.2025.03.30", "t15.2025.04.13"]


def word_ed(ref: str, hyp: str) -> tuple:
    """Word-level Levenshtein distance after the official remove_punctuation
    (evaluate_model_helpers.py): strip non-letters, collapse '- ', lowercase."""
    def clean(s):
        s = re.sub(r"[^a-zA-Z\- \']", "", s)
        return s.replace("- ", " ").lower().split()
    r, h = clean(ref), clean(hyp)
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int64)
    d[:, 0] = np.arange(len(r) + 1)
    d[0, :] = np.arange(len(h) + 1)
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            c = 0 if r[i - 1] == h[j - 1] else 1
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + c)
    return int(d[-1, -1]), len(r)


def read_tsv(path):
    df = pd.read_csv(path, sep="\t", encoding="utf-8")
    return df


def main():
    import h5py

    csv = pd.read_csv(os.path.join(P4, "val_greedy_per_trial.csv"))
    f = h5py.File(LOGITS, "r")
    hdf_counts = {s: len(f[s].keys()) for s in f.keys()}
    f.close()
    csv_counts = csv.groupby("session").size().to_dict()
    assert hdf_counts == csv_counts, "logits-hdf5 per-session counts != csv counts"
    # csv row order == tsv row order (both iterate the same session/trial order)
    sessions = csv["session"].tolist()

    fin = read_tsv(os.path.join(P4, "p4_final_full.tsv"))
    opt = read_tsv(os.path.join(P4, "p4_opt_fx.tsv"))
    assert len(fin) == len(opt) == len(csv), "row count mismatch"
    # sanity: opt tsv's own ed column vs recomputed (uppercase) -- must match
    for i in range(0, len(opt), 200):
        ed, rl = word_ed(opt["ref"][i], opt["hyp"][i])
        assert ed == int(opt["ed"][i]), f"opt ed mismatch at row {i}"

    rows = []
    for i, s in enumerate(sessions):
        if s not in SESSIONS_2025:
            continue
        ed5, rl5 = word_ed(fin["ref"][i], fin["hyp"][i])
        edo, _ = word_ed(opt["ref"][i], opt["hyp"][i])
        rows.append({"session": s, "ref_words": rl5, "ed_5gram": ed5, "ed_opt": edo})
    df = pd.DataFrame(rows)

    print(df.assign(
        wer_5gram=lambda d: 100 * d.ed_5gram / d.ref_words,
        wer_opt=lambda d: 100 * d.ed_opt / d.ref_words,
    ).to_string(index=False))

    def pooled(sub, col):
        m = df[df.session.isin(sub)]
        return 100 * m[col].sum() / m.ref_words.sum()

    all6 = SESSIONS_2025
    last5 = SESSIONS_2025[1:]  # date-sorted last five (excl 01.10)
    no0112 = [s for s in all6 if s != "t15.2025.01.12"]
    print("\npooled 6-day   : 5-gram %.2f | +OPT %.2f" % (pooled(all6, "ed_5gram"), pooled(all6, "ed_opt")))
    print("pooled last-5  : 5-gram %.2f | +OPT %.2f" % (pooled(last5, "ed_5gram"), pooled(last5, "ed_opt")))
    print("pooled no01.12 : 5-gram %.2f | +OPT %.2f" % (pooled(no0112, "ed_5gram"), pooled(no0112, "ed_opt")))
    print("\narchive anchors: run_120k 5-gram last5=62.21 / +OPT=62.01 ; p4 +OPT last5=62.91")
    df.to_csv(os.path.join(REPO, r"result\calib_20261014_p0_wer_byday.csv"), index=False)


if __name__ == "__main__":
    main()
