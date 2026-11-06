# -*- coding: utf-8 -*-
"""WER for T15 subset lg_viterbi csv (meta row order).

Run in WSL: python3 t15_wer_sweep.py <lg_csv>
Reads /root/lm/t15meta200.tsv for trial order, refs from t15_logits.hdf5.
"""
import csv
import h5py
import re
import sys

import editdistance


def norm(s):
    return re.sub(r"[^A-Za-z<\x27 ]", " ", s).upper().split()


def main():
    order = []
    with open("/root/lm/t15meta200.tsv", encoding="utf-8") as f:
        for line in f:
            rel, sess, key = line.strip().split("\t")
            order.append((sess, key))
    refs = {}
    with h5py.File("./20240815_card/data/20241009_card_BtoT25/run_120k/t15_logits.hdf5",
                   "r") as f:
        for sess in f.keys():
            for key in f[sess].keys():
                refs[(sess, key)] = norm(f[sess][key].attrs["sentence_label"])
    rlist = [refs[k] for k in order]
    hyps = []
    with open(sys.argv[1], encoding="utf-8", newline="") as fo:
        for r in csv.DictReader(fo):
            hyps.append(r["hyp_words"].split() if r["hyp_words"] else [])
    assert len(rlist) == len(hyps), (len(rlist), len(hyps))
    ed = sum(editdistance.eval(a, b) for a, b in zip(rlist, hyps))
    tot = sum(len(a) for a in rlist)
    print(f"{100*ed/tot:.2f}%  ({ed}/{tot}, {len(rlist)} sentences)")


if __name__ == "__main__":
    main()
