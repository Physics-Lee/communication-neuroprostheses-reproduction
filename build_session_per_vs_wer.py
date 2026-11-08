#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild `session_per_vs_wer.json` directly from the archived per-trial
decode outputs inside each `yyyymmdd_author/results/decode_outputs/` folder.

Usage:
    python build_session_per_vs_wer.py
"""
import csv
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TOK = re.compile(r"[^a-z0-9']+")


def norm_words(s):
    """Normalize a transcript string into a list of clean word tokens.

    Casefolds the input, splits on whitespace, and strips every character
    that is not a lowercase letter, digit, or apostrophe.

    Parameters
    ----------
    s : str
        Raw transcript string (reference or hypothesis sentence).

    Returns
    -------
    list of str
        Non-empty normalized tokens in original order.
    """
    out = []
    for w in s.casefold().split():
        w = TOK.sub("", w)
        if w:
            out.append(w)
    return out


def word_ed(ref, hyp):
    """Compute the word-level edit distance between two transcripts.

    Uses a rolling-row dynamic-programming Levenshtein computation over
    the word tokens produced by ``norm_words``.

    Parameters
    ----------
    ref : str
        Reference (ground-truth) transcript.
    hyp : str
        Hypothesis (decoded) transcript.

    Returns
    -------
    tuple of (int, int)
        ``(distance, n_ref_words)`` where ``distance`` is the minimum
        number of word insertions, deletions, and substitutions, and
        ``n_ref_words`` is the number of reference tokens.
    """
    a, b = norm_words(ref), norm_words(hyp)
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n], len(a)


def read_tsv(path):
    """Read (reference, hypothesis) sentence pairs from a decode TSV.

    Skips the header row (first field equal to ``"key"``) and any line
    with fewer than three tab-separated fields.

    Parameters
    ----------
    path : str
        Path to a tab-separated decode-output file whose columns are at
        least ``key``, reference sentence, hypothesis sentence.

    Returns
    -------
    list of tuple of (str, str)
        ``(reference, hypothesis)`` pairs, one per data row, in file
        order.
    """
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3 and p[0] != "key":
                rows.append((p[1], p[2]))
    return rows


def build(name, per_csv, tsv):
    """Aggregate per-trial PER errors and decode TSVs into session stats.

    Joins a per-trial CSV (edit distances grouped by session) with a
    decode TSV positionally, verifies the join by comparing sentence
    labels, accumulates phoneme- and word-level errors per session, and
    pools them across sessions.

    Parameters
    ----------
    name : str
        Human-readable dataset label used in log output.
    per_csv : str
        Path to a per-trial CSV with columns ``sentence_label``,
        ``session``, ``edit_distance``, and ``true_len``.
    tsv : str
        Path to the matching decode-output TSV (see ``read_tsv``).

    Returns
    -------
    dict
        ``{"sessions": [...], "pooled": {...}}`` where ``sessions`` lists
        ``{"session", "n", "per", "wer"}`` dicts for sessions with at
        least 3 trials (PER/WER in percent) and ``pooled`` holds the
        all-session pooled ``per`` and ``wer`` percentages.

    Raises
    ------
    AssertionError
        If the CSV and TSV row counts differ, or if too many sentence
        labels mismatch (positional join invalid).
    """
    with open(per_csv, encoding="utf-8") as f:
        per_rows = list(csv.DictReader(f))
    tsv_rows = read_tsv(tsv)
    assert len(per_rows) == len(tsv_rows), (
        f"{name}: {len(per_rows)} per-trial rows vs {len(tsv_rows)} tsv rows"
    )
    sessions = {}
    mismatch = 0
    for row, (ref, hyp) in zip(per_rows, tsv_rows):
        if row["sentence_label"].casefold().strip() != ref.casefold().strip():
            mismatch += 1
            continue
        wed, wref = word_ed(ref, hyp)
        d = sessions.setdefault(
            row["session"], {"pe": 0, "pl": 0, "we": 0, "wl": 0, "n": 0}
        )
        d["pe"] += int(row["edit_distance"])
        d["pl"] += int(row["true_len"])
        d["we"] += wed
        d["wl"] += wref
        d["n"] += 1
    assert mismatch <= max(2, int(0.01 * len(per_rows))), (
        f"{name}: {mismatch} ref mismatches - join is not positional"
    )
    out = [
        {
            "session": s,
            "n": v["n"],
            "per": round(100 * v["pe"] / v["pl"], 2),
            "wer": round(100 * v["we"] / v["wl"], 2),
        }
        for s, v in sorted(sessions.items())
        if v["n"] >= 3
    ]
    pooled = {
        "per": round(
            100 * sum(v["pe"] for v in sessions.values())
            / sum(v["pl"] for v in sessions.values()),
            2,
        ),
        "wer": round(
            100 * sum(v["we"] for v in sessions.values())
            / sum(v["wl"] for v in sessions.values()),
            2,
        ),
    }
    print(f"{name:18s}: {len(out):2d} sessions | pooled PER {pooled['per']:5.2f}% | WER {pooled['wer']:5.2f}%")
    return {"sessions": out, "pooled": pooled}


def main():
    """Rebuild ``session_per_vs_wer.json`` for all benchmark datasets.

    Runs ``build`` for Willett 2023, Card 2024, Jude 2026b, and the three
    Kunz 2025 inner-speech participants, then writes the combined result
    dictionary to ``session_per_vs_wer.json`` in the repository root.
    """
    w_dir = os.path.join(HERE, "20230831_willett", "results", "decode_outputs")
    c_dir = os.path.join(HERE, "20240815_card", "results", "decode_outputs")
    j_dir = os.path.join(HERE, "20260428_jude", "results", "decode_outputs")
    k_dir = os.path.join(HERE, "20250821_kunz", "results", "decode_outputs")

    data = {
        "Willett 2023 (T12)": build(
            "Willett 2023",
            os.path.join(w_dir, "test_partition_greedy_per_trial.csv"),
            os.path.join(w_dir, "wl_5g_opt_official.tsv"),
        ),
        "Card 2024 (T15)": build(
            "Card 2024",
            os.path.join(c_dir, "val_greedy_per_trial.csv"),
            os.path.join(c_dir, "p4_fixed_opt_fx.tsv"),
        ),
        "Jude 2026 (LIS, T17)": build(
            "Jude 2026b",
            os.path.join(j_dir, "val_greedy_per_trial.csv"),
            os.path.join(j_dir, "off_5g_opt_official.tsv"),
        ),
    }
    for tag, part in (("T12", "t12"), ("T15", "t15"), ("T16", "t16")):
        rd = os.path.join(k_dir, f"run_{part}_mixed")
        data[f"Kunz 2025 (inner, {tag})"] = build(
            f"Kunz 2025 ({tag})",
            os.path.join(rd, "val_greedy_per_trial.csv"),
            os.path.join(rd, "off_final_full.tsv"),
        )

    out_path = os.path.join(HERE, "session_per_vs_wer.json")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"[ok] Wrote {out_path}")


if __name__ == "__main__":
    main()
