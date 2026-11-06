# -*- coding: utf-8 -*-
"""Oracle WER vs n-best depth, from an nbest tsv (key/rank/ac/graph/sentence).

Ref normalization identical to wer_from_tsv.py. Pure stdlib + editdistance,
so it runs under both the Windows venv and WSL python3.
Usage: vac_oracle_depth.py <nbest.tsv> <final.tsv> [d1,d2,...]
"""
import collections
import re
import sys

import editdistance


def norm_ref(sent):
    return re.sub(r"[^A-Za-z'< ]", " ", sent).upper().split()


def read_refs(path):
    refs = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3 or p[0] == "key":
                continue
            refs[p[0]] = norm_ref(p[1])
    return refs


def read_nbest(path):
    groups = collections.OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5 or p[0] == "key":
                continue
            groups.setdefault(p[0], []).append(p[4])
    return groups


def main():
    nbest_path, final_path = sys.argv[1], sys.argv[2]
    depths = [int(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3 \
        else [1, 5, 10, 25, 50, 100, 200, 400]
    refs = read_refs(final_path)
    groups = read_nbest(nbest_path)
    tot_ref = sum(len(r) for r in refs.values())
    print(f"{nbest_path}: {len(groups)} trials, {tot_ref} ref words, "
          f"pool depth median {sorted(len(v) for v in groups.values())[len(groups)//2]}")
    for k in depths:
        tot_ed = 0
        for key, ref in refs.items():
            cands = groups.get(key, [])[:k]
            if not cands:
                tot_ed += len(ref)  # no candidate at all -> all deletions
                continue
            tot_ed += min(editdistance.eval(ref, c.split()) for c in cands)
        print(f"  depth {k:>4}: oracle WER = {tot_ed}/{tot_ref} = "
              f"{100*tot_ed/max(1,tot_ref):.2f}%")


if __name__ == "__main__":
    main()
