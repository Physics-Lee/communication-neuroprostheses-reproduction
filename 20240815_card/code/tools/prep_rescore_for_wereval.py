# -*- coding: utf-8 -*-
"""Uppercase the hyp column of an opt_rescore output tsv in place-safe copy.

opt_rescore.py (fixed version) feeds the LM lowercase and writes the chosen
hyp in lowercase; wer_eval_card.py compares hyp tokens against uppercased refs
without normalizing hyp, so raw rescore tsvs would score ~100%. This writes
<in>.upperhyp.tsv with hyp uppercased, keeping key/ref/hyp/score/ed/ref_len.

Usage: prep_rescore_for_wereval.py <in.tsv> [out.tsv]
"""
import sys


def main():
    src = sys.argv[1]
    dst = sys.argv[2] if len(sys.argv) > 2 else src + ".upperhyp.tsv"
    n = 0
    with open(src, encoding="utf-8") as f, \
            open(dst, "w", encoding="utf-8", newline="") as o:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and p[0] != "key":
                p[2] = p[2].upper()
                o.write("\t".join(p) + "\n")
                n += 1
            else:
                o.write(line)
    print(f"wrote {n} rows -> {dst}")


if __name__ == "__main__":
    main()
