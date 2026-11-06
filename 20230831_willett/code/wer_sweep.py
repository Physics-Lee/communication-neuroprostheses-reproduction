# -*- coding: utf-8 -*-
"""WER for a subset lg_viterbi csv (first N rows aligned with greedy csv).

Run in WSL: python3 wer_sweep.py <lg_csv> <n_subset>
"""
import csv
import re
import sys

import editdistance

GREEDY = "./20230831_willett/data/20230616_willett_BtoT24/run_60k/test_partition_greedy_per_trial.csv"


def norm_ref(sent):
    return re.sub(r"[^A-Za-z'< ]", " ", sent).upper().split()


def main():
    n = int(sys.argv[2])
    refs = []
    with open(GREEDY, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            refs.append(norm_ref(r["sentence_label"]))
    hyps = []
    with open(sys.argv[1], encoding="utf-8", newline="") as f:
        for i, r in enumerate(csv.DictReader(f)):
            if i >= n:
                break
            hyps.append(r["hyp_words"].split() if r["hyp_words"] else [])
    ed = sum(editdistance.eval(a, b) for a, b in zip(refs, hyps))
    tot = sum(len(a) for a in refs[:n])
    print(f"{100*ed/tot:.2f}%  ({ed}/{tot}, {n} sentences)")


if __name__ == "__main__":
    main()
