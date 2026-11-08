# -*- coding: utf-8 -*-
"""Compare 5-gram / OPT-6.7B / Qwen3.5-4B rescoring WERs across the 4 papers.

Reads per-sentence decode TSVs (key TAB ref TAB hyp TAB ...), computes pooled
WER (+ optional per-session median via a meta csv join on row order).

Usage: compare_rescorers.py
"""
import csv
import re
from collections import defaultdict

import editdistance

ROOT = r".\shared\data"


def norm(s):
    """Normalize a transcript for WER scoring.

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


def wer_from_tsv(path, meta_csv=None, meta_session=False):
    """Compute pooled and per-session-median WER from a decode TSV.

    Scores each data row (key TAB ref TAB hyp ...) with a token edit
    distance against the normalized reference. Optionally joins a
    per-trial meta CSV by row order to also produce the median of
    per-session WERs.

    Parameters
    ----------
    path : str
        Path to the decode-output TSV to score.
    meta_csv : str, optional
        Path to a per-trial CSV with a ``session`` column, aligned with
        the TSV rows by position. If None, no per-session breakdown is
        computed.
    meta_session : bool, optional
        Accepted for interface compatibility; not used by the current
        implementation.

    Returns
    -------
    tuple of (float or None, float or None)
        ``(pooled_wer, day_median_wer)`` in percent. Both are None when
        the file has no scoreable tokens; ``day_median_wer`` is None when
        no session metadata was supplied.
    """
    te = tn = 0
    sess_ed, sess_n = defaultdict(int), defaultdict(int)
    sessions = None
    if meta_csv:
        with open(meta_csv, encoding="utf-8", newline="") as f:
            sessions = [r["session"] for r in csv.DictReader(f)]
    with open(path, encoding="utf-8") as f:
        i = 0
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] == "key":
                continue
            r = norm(p[1])
            h = (p[2] or "").split()
            ed = editdistance.eval(r, h)
            te += ed
            tn += len(r)
            if sessions and i < len(sessions):
                sess_ed[sessions[i]] += ed
                sess_n[sessions[i]] += len(r)
            i += 1
    if tn == 0:
        return None, None
    pooled = 100 * te / tn
    day_med = None
    if sess_ed:
        dw = sorted(100 * sess_ed[s] / sess_n[s] for s in sess_ed)
        day_med = dw[len(dw) // 2]
    return pooled, day_med


RUNS = [
    # (label, tsv, meta_csv or None)
    ("Willett 5g",      f"{ROOT}\\20230616_willett_BtoT24\\run_60k\\wl_final_full.tsv",            f"{ROOT}\\20230616_willett_BtoT24\\run_60k\\test_partition_greedy_per_trial.csv"),
    ("Willett +OPT",    f"{ROOT}\\20230616_willett_BtoT24\\run_60k\\wl_5g_opt_official.tsv",      f"{ROOT}\\20230616_willett_BtoT24\\run_60k\\test_partition_greedy_per_trial.csv"),
    ("Willett +Qwen",   f"{ROOT}\\20230616_willett_BtoT24\\run_60k\\wl_5g_qwen.tsv",              f"{ROOT}\\20230616_willett_BtoT24\\run_60k\\test_partition_greedy_per_trial.csv"),
    ("Willett +Qwen(parallel)", f"{ROOT}\\20230616_willett_BtoT24\\run_60k\\wl_5g_opt_qwen.tsv",  f"{ROOT}\\20230616_willett_BtoT24\\run_60k\\test_partition_greedy_per_trial.csv"),
    ("Willett +Qwen-base", f"{ROOT}\\20230616_willett_BtoT24\\run_60k\\wl_5g_opt_qwenbase.tsv",   f"{ROOT}\\20230616_willett_BtoT24\\run_60k\\test_partition_greedy_per_trial.csv"),
    ("Card 5g",         f"{ROOT}\\20241009_card_BtoT25\\run_120k\\final_full.tsv",                 f"{ROOT}\\20241009_card_BtoT25\\run_120k\\val_greedy_per_trial.csv"),
    ("Card +OPT",       f"{ROOT}\\20241009_card_BtoT25\\run_120k\\word_decode_5g_opt_official.tsv", f"{ROOT}\\20241009_card_BtoT25\\run_120k\\val_greedy_per_trial.csv"),
    ("Card +Qwen",      f"{ROOT}\\20241009_card_BtoT25\\run_120k\\word_decode_5g_qwen.tsv",        f"{ROOT}\\20241009_card_BtoT25\\run_120k\\val_greedy_per_trial.csv"),
    ("Card +Qwen-base", f"{ROOT}\\20241009_card_BtoT25\\run_120k\\word_decode_5g_opt_qwenbase.tsv", f"{ROOT}\\20241009_card_BtoT25\\run_120k\\val_greedy_per_trial.csv"),
    ("Jude 5g",         f"{ROOT}\\20260428_jude\\run_25k\\off_final_full.tsv",                     None),
    ("Jude +OPT",       f"{ROOT}\\20260428_jude\\run_25k\\off_5g_opt_official.tsv",               None),
    ("Jude +Qwen",      f"{ROOT}\\20260428_jude\\run_25k\\off_5g_qwen.tsv",                       None),
    ("Jude +Qwen(parallel)", f"{ROOT}\\20260428_jude\\run_25k\\off_5g_opt_qwen.tsv",              None),
    ("Jude +Qwen-base", f"{ROOT}\\20260428_jude\\run_25k\\off_5g_opt_qwenbase.tsv",              None),
    ("Kunz T12 +OPT",   f"{ROOT}\\20250821_kunz\\run_t12_mixed\\off_bp1_opt.tsv",                  None),
    ("Kunz T12 +Qwen",  f"{ROOT}\\20250821_kunz\\run_t12_mixed\\qwen_bp1_opt.tsv",                None),
    ("Kunz T15 +OPT",   f"{ROOT}\\20250821_kunz\\run_t15_mixed\\off_bp1_opt.tsv",                  None),
    ("Kunz T15 +Qwen",  f"{ROOT}\\20250821_kunz\\run_t15_mixed\\qwen_bp1_opt.tsv",                None),
    ("Kunz T16 +OPT",   f"{ROOT}\\20250821_kunz\\run_t16_mixed\\off_bp1_opt.tsv",                  None),
    ("Kunz T16 +Qwen",  f"{ROOT}\\20250821_kunz\\run_t16_mixed\\qwen_bp1_opt.tsv",                None),
]

if __name__ == "__main__":
    import os
    for label, tsv, meta in RUNS:
        if not os.path.exists(tsv):
            print(f"{label:26s} (missing)")
            continue
        pooled, day_med = wer_from_tsv(tsv, meta)
        if pooled is None:
            print(f"{label:26s} (empty)")
        elif day_med is not None:
            print(f"{label:26s} pooled {pooled:6.2f}%   day-median {day_med:6.2f}%")
        else:
            print(f"{label:26s} pooled {pooled:6.2f}%")
