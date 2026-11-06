# -*- coding: utf-8 -*-
"""Evaluate Card 2024 word decode TSVs in the paper's reporting style.

Joins trial keys (row order) with val_greedy_per_trial.csv to recover
session + corpus per trial, then reports:
  - pooled WER + bootstrap 95% CI over trials
  - per-session WER (paper Fig 2 style) + per-corpus breakdown
  - n-best pool oracle (WER if we always picked the best candidate)

Usage: wer_eval_card.py <final.tsv> [nbest.tsv] [--greedy-csv PATH]
       final.tsv: key TAB ref TAB hyp TAB score (from wenet_lattice sweep)
       nbest.tsv: key TAB rank TAB ac TAB lm TAB sentence
"""
import argparse
import collections
import random
import re

import editdistance

RUN = r".\20240815_card\data\20241009_card_BtoT25\run_120k"
GREEDY = RUN + r"\val_greedy_per_trial.csv"


def norm(s):
    return re.sub(r"[^A-Za-z'< ]", " ", s).upper().split()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("final_tsv")
    ap.add_argument("nbest_tsv", nargs="?", default="")
    ap.add_argument("--greedy-csv", default=GREEDY)
    args = ap.parse_args()

    meta = []
    import csv
    with open(args.greedy_csv, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            meta.append((r["session"], r["corpus"]))

    rows = []
    with open(args.final_tsv, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 4 or p[0] == "key":
                continue
            rows.append(p)
    assert len(rows) == len(meta), f"{len(rows)} rows vs {len(meta)} meta"

    ed_by_key = {}
    sess_ed = collections.Counter()
    sess_n = collections.Counter()
    corp_ed = collections.Counter()
    corp_n = collections.Counter()
    for (sess, corp), p in zip(meta, rows):
        key, ref, hyp = p[0], p[1], p[2]
        r, h = norm(ref), (hyp or "").split()
        ed = editdistance.eval(r, h)
        ed_by_key[key] = ed
        sess_ed[sess] += ed
        sess_n[sess] += len(r)
        corp_ed[corp] += ed
        corp_n[corp] += len(r)

    tot_ed, tot_n = sum(sess_ed.values()), sum(sess_n.values())
    print(f"POOLED WER: {100*tot_ed/tot_n:.2f}%  ({tot_ed}/{tot_n}, "
          f"{len(rows)} trials)")

    # bootstrap CI over trials
    eds, lens = [], []
    for (sess, corp), p in zip(meta, rows):
        r = norm(p[1])
        eds.append(ed_by_key[p[0]])
        lens.append(len(r))
    rng = random.Random(0)
    bs = []
    for _ in range(10000):
        e = n = 0
        for i in rng.choices(range(len(eds)), k=len(eds)):
            e += eds[i]
            n += lens[i]
        bs.append(e / n)
    bs.sort()
    print(f"bootstrap 95% CI: [{100*bs[499]:.2f}, {100*bs[9499]:.2f}]")

    print("\n--- per-session (paper Fig 2 style) ---")
    for s in sorted(sess_n):
        print(f"{s}  {100*sess_ed[s]/sess_n[s]:6.2f}%  ({sess_n[s]} ref words)")
    day_wers = sorted(100 * sess_ed[s] / sess_n[s] for s in sess_n)
    if day_wers:
        med = day_wers[len(day_wers) // 2]
        print(f"day-median WER: {med:.2f}%  range [{day_wers[0]:.2f}, "
              f"{day_wers[-1]:.2f}] over {len(day_wers)} sessions")

    print("\n--- per-corpus ---")
    for c in sorted(corp_n, key=lambda c: -corp_n[c]):
        print(f"{c:<28} {100*corp_ed[c]/corp_n[c]:6.2f}%  ({corp_n[c]} words)")

    if args.nbest_tsv:
        cands = collections.defaultdict(set)
        with open(args.nbest_tsv, encoding="utf-8") as f:
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) >= 5:
                    cands[p[0]].add(p[4])
        oed = on = unfixed = 0
        for (sess, corp), pr in zip(meta, rows):
            r = norm(pr[1])
            best = min(editdistance.eval(r, c.split())
                       for c in cands.get(pr[0], [""]))
            if best > 0:
                unfixed += 1
            oed += best
            on += len(r)
        print(f"\nPOOL ORACLE: {100*oed/on:.2f}%  ({oed}/{on}); "
              f"trials not fully fixed: {unfixed}/{len(rows)}")


if __name__ == "__main__":
    main()
