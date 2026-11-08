# -*- coding: utf-8 -*-
"""T7: word-level INS/DEL/SUB error anatomy from final tsvs.

Card gets an early(2023)/late(2024+) split via run_120k/val_greedy_per_trial.csv
(row-ordered). The doc's qualitative claim "late sessions: fluent but
acoustically wrong English, LM overpowers flat posteriors" predicts a high
substitution share; insertion-heavy output would instead indicate blank/PIL
issues. Usage: vac_error_ops.py (run from repo root)
"""
import csv
import re

GREEDY = r"data\20241009_card_BtoT25\run_120k\val_greedy_per_trial.csv"
SETS = {
    "willett": r"data\20230616_willett_BtoT24\run_60k\wl_final_full.tsv",
    "card":    r"data\20241009_card_BtoT25\run_120k\final_full.tsv",
    "jude":    r"data\20260428_jude\run_25k\off_final_full.tsv",
    "kunz_t12": r"data\20250821_kunz\vac\t12_bp1_a0.325_final.tsv",
    "kunz_t15": r"data\20250821_kunz\vac\t15_bp1_a0.5_final.tsv",
    "kunz_t16": r"data\20250821_kunz\vac\t16_bp1_a0.325_final.tsv",
}


def norm(s):
    """Normalize a transcript for edit-distance scoring.

    Replaces every character other than ASCII letters, apostrophes,
    ``<``/``>`` and spaces with a space, uppercases, and tokenizes on
    whitespace.

    Parameters
    ----------
    s : str
        Raw transcript string (reference or hypothesis).

    Returns
    -------
    list of str
        Uppercase word tokens.
    """
    return re.sub(r"[^A-Za-z'< ]", " ", s).upper().split()


def ops(a, b):
    """edit ops turning a (ref) into b (hyp): D = ref word deleted,
    I = hyp word inserted, R = substitution.

    Parameters
    ----------
    a : list of str
        Reference word tokens.
    b : list of str
        Hypothesis word tokens.

    Returns
    -------
    list of str
        One opcode per aligned step in alignment order: ``"M"`` match,
        ``"R"`` substitution, ``"D"`` deletion from the reference, or
        ``"I"`` insertion in the hypothesis.
    """
    n, m = len(a), len(b)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i-1][j-1] + (a[i-1] != b[j-1]),
                           dp[i-1][j] + 1, dp[i][j-1] + 1)
    i, j, out = n, m, []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + (a[i-1] != b[j-1]):
            out.append("R" if a[i-1] != b[j-1] else "M"); i, j = i-1, j-1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            out.append("D"); i -= 1
        else:
            out.append("I"); j -= 1
    return out[::-1]


def tally(rows, label):
    """Print insertion/deletion/substitution error shares for a row set.

    Scores every (reference, hypothesis) pair with ``ops`` after
    normalization and prints the I/D/R counts and percentages plus the
    pooled WER.

    Parameters
    ----------
    rows : list of list of str
        Tab-split rows of at least 3 fields where index 1 is the
        reference and index 2 the hypothesis.
    label : str
        Row-set label used in the printed report.
    """
    c = {"I": 0, "D": 0, "R": 0}
    n_ref = 0
    for p in rows:
        ref, hyp = norm(p[1]), p[2].split()
        n_ref += len(ref)
        for o in ops(ref, hyp):
            if o != "M":
                c[o] += 1
    tot = sum(c.values())
    print(f"  {label:12s} ERR {tot:5d} | I {c['I']:5d} ({100*c['I']/max(1,tot):4.1f}%) "
          f"D {c['D']:5d} ({100*c['D']/max(1,tot):4.1f}%) "
          f"R {c['R']:5d} ({100*c['R']/max(1,tot):4.1f}%) | WER {100*tot/max(1,n_ref):.2f}%")


def main():
    """Print word-level error anatomy for all configured datasets.

    For every dataset TSV in ``SETS``, tallies I/D/R error shares over
    all rows; for Card additionally splits rows into early-2023 and
    late-2024+ sessions using the row-ordered session list.
    """
    sessions = [r["session"] for r in csv.DictReader(
        open(GREEDY, encoding="utf-8", newline=""))]
    for name, path in SETS.items():
        rows = []
        for line in open(path, encoding="utf-8"):
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and p[0] != "key":
                rows.append(p)
        print(name)
        tally(rows, "all")
        if name == "card":
            tally([p for p, s in zip(rows, sessions) if s < "t15.2024"], "early2023")
            tally([p for p, s in zip(rows, sessions) if s >= "t15.2024"], "late2024+")


if __name__ == "__main__":
    main()
