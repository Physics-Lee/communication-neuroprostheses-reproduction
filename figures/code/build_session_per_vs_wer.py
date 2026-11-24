#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild `session_per_vs_wer.json` directly from the archived per-trial
decode outputs inside each `yyyymmdd_author/results/decode_outputs/` folder.

Usage:
    python build_session_per_vs_wer.py
"""
import argparse
import csv
import json
from pathlib import Path
import re

from figure_paths import PUBLISHED, ROOT, load_benchmark
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
        If the CSV and TSV row counts differ or a paired sentence label
        differs (positional join invalid).
    """
    with open(per_csv, encoding="utf-8") as f:
        per_rows = list(csv.DictReader(f))
    tsv_rows = read_tsv(tsv)
    assert len(per_rows) == len(tsv_rows), (
        f"{name}: {len(per_rows)} per-trial rows vs {len(tsv_rows)} tsv rows"
    )
    sessions = {}
    for i, (row, (ref, hyp)) in enumerate(zip(per_rows, tsv_rows), start=1):
        assert row["sentence_label"].casefold().strip() == ref.casefold().strip(), (
            f"{name}: reference mismatch at trial row {i}, session {row['session']}"
        )
        wed, wref = word_ed(ref, hyp)
        d = sessions.setdefault(
            row["session"], {"pe": 0, "pl": 0, "we": 0, "wl": 0, "n": 0}
        )
        d["pe"] += int(row["edit_distance"])
        d["pl"] += int(row["true_len"])
        d["we"] += wed
        d["wl"] += wref
        d["n"] += 1
    out = [
        {
            "session": s,
            "n": v["n"],
            "per": round(100 * v["pe"] / v["pl"], 2),
            "wer": round(100 * v["we"] / v["wl"], 2),
            "phoneme_errors": v["pe"],
            "phoneme_counts": v["pl"],
            "word_errors": v["we"],
            "word_counts": v["wl"],
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PUBLISHED)
    args = parser.parse_args()
    sources = load_benchmark()["figure3_per_sources"]
    data = {
        name: build(name, ROOT / source["greedy"], ROOT / source["decode"])
        for name, source in sources.items()
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "session_per_vs_wer.json"
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    print(f"[ok] Wrote {out_path}")


if __name__ == "__main__":
    main()
