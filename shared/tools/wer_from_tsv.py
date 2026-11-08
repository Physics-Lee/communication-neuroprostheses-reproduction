# -*- coding: utf-8 -*-
"""WER for tlg_viterbi TSV output (key \t ref \t hyp \t score).

Run on Windows: python wer_from_tsv.py <out.tsv> [limit]
Ref normalization identical to decode_words_btot24_v2.py.
"""
import csv
import re
import sys

import editdistance


def norm_ref(sent):
    """Normalize a reference sentence for edit-distance scoring.

    Replaces every character other than ASCII letters, apostrophes,
    ``<``/``>`` and spaces with a space, uppercases, and tokenizes on
    whitespace (identical to decode_words_btot24_v2.py).

    Parameters
    ----------
    sent : str
        Raw reference sentence.

    Returns
    -------
    list of str
        Uppercase word tokens.
    """
    return re.sub(r"[^A-Za-z'< ]", " ", sent).upper().split()


def main():
    """Compute pooled WER from a decode TSV and save per-trial details.

    Reads key/ref/hyp/score rows, scores each with a token edit
    distance, prints the pooled WER, and writes a companion
    ``<path>.wer`` CSV with per-trial edit distances and reference
    lengths.

    Notes
    -----
    Arguments come from ``sys.argv``:
    ``wer_from_tsv.py <out.tsv> [limit]``.
    """
    path = sys.argv[1]
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    tot_ed = tot_ref = n = 0
    rows = []
    with open(path, encoding="utf-8", newline="") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or parts[0] == "key":
                continue
            ref = norm_ref(parts[1])
            hyp = parts[2].split() if parts[2] else []
            ed = editdistance.eval(ref, hyp)
            tot_ed += ed
            tot_ref += len(ref)
            n += 1
            rows.append((parts[0], ed, len(ref)))
            if limit and n >= limit:
                break
    print(f"WER = {tot_ed}/{tot_ref} = {100*tot_ed/max(1,tot_ref):.2f}% "
          f"over {n} sentences")
    with open(path + ".wer", "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["key", "edit_distance", "ref_len"])
        w.writerows(rows)
    print("saved", path + ".wer")


if __name__ == "__main__":
    main()
