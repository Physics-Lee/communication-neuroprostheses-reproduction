#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild `session_cer_vs_wer.json`: session-level raw CER (no LM) vs.
post-LM WER for the four character-level study arms we reproduced.

This is the character-level analog of `build_session_per_vs_wer.py`
and feeds Figure 3(b) in `render_phase_memory_figure.py`.

Study arms and sources
----------------------
Willett et al., 2021 (T5, handwriting, bigram LM), 10 sessions:
    Greedy CER counts extracted from the archived per-session errCounts.mat
    using HeldOutTrials, with bigram word errors from the matched trial CSV.
Fan 2023 (T5):
    Greedy CER and character-LM WER on the final six trials in each session.
Jude 2026a (T17 / T18):
    Multisession greedy CER and 5-gram LM WER on matched evaluation trials.

Conventions (mirror `build_session_per_vs_wer.py`)
--------------------------------------------------
- Sessions with fewer than ``MIN_TRIALS`` scored trials are dropped
  (removes ``t18.2024.12.04``, n=1).
- Pooled study values are micro-averages from raw counts where counts
  exist (both axes for Willett and Fan, WER for Jude). Jude CER is
  weighted by ``n_chars``. The pooling mode is recorded in ``_meta``.
- Reference characters-per-word (spaces included, matching the CER
  denominators) is accumulated from the same trials and stored in
  ``_meta.chars_per_word_pooled`` for the no-LM baseline curve
  ``1 - (1 - CER)^L`` of the phase diagram.

Usage:
    python build_session_cer_vs_wer.py
"""
import argparse
import csv
import json
from pathlib import Path

import editdistance

from figure_paths import PUBLISHED, ROOT, load_benchmark
MIN_TRIALS = 3

SOURCES = load_benchmark()["figure3_cer_sources"]
FAN_TRIALS = SOURCES["Fan 2023 (T5)"]["archive"]


def build_willett21():
    """Assemble the Willett 2021 (T5) session table.

    Returns
    -------
    dict
        ``{"sessions": [...], "pooled": {...}}`` with per-session
        ``(cer, wer)`` in percent and micro-averaged pooled values.
    """
    source = SOURCES["Willett 2021 (T5)"]
    with (ROOT / source["greedy_counts"]).open(encoding="utf-8", newline="") as stream:
        raw_counts = {
            row["session"]: {
                "n": int(row["trials"]),
                "char_errs": int(row["char_errors"]),
                "char_counts": int(row["reference_chars"]),
            }
            for row in csv.DictReader(stream)
        }
    path = ROOT / source["archive"]
    agg = {}
    with open(path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            a = agg.setdefault(r["session"], {"n": 0, "ce": 0, "cc": 0,
                                              "we": 0, "wc": 0})
            a["n"] += 1
            a["cc"] += int(r["charCounts"])
            a["wc"] += int(r["wordCounts"])
            # charErrs of the m3 CSV are post-bigram-decode characters and
            # are not used here; WER comes from wordErrs/wordCounts.
            a["we"] += int(r["wordErrs"])
    sessions = []
    tot_ce = tot_cc = tot_we = tot_wc = 0
    for s in sorted(agg):
        raw = raw_counts[s]
        if raw["n"] != agg[s]["n"]:
            raise ValueError(f"Willett 2021: mismatched holdout size in {s}")
        if raw["n"] < MIN_TRIALS:
            continue
        cer = 100.0 * raw["char_errs"] / raw["char_counts"]
        wer = 100.0 * agg[s]["we"] / agg[s]["wc"]
        sessions.append({"session": s, "n": raw["n"],
                         "cer": round(cer, 2), "wer": round(wer, 2),
                         "char_counts": int(raw["char_counts"]),
                         "word_counts": agg[s]["wc"],
                         "char_errors": raw["char_errs"], "word_errors": agg[s]["we"]})
        tot_ce += raw["char_errs"]; tot_cc += raw["char_counts"]
        tot_we += agg[s]["we"]; tot_wc += agg[s]["wc"]
    return {
        "sessions": sessions,
        "pooled": {"cer": round(100.0 * tot_ce / tot_cc, 2),
                   "wer": round(100.0 * tot_we / tot_wc, 2)},
        "pooling": "micro (errCounts char errors x m3 bigram word errors)",
    }


def build_fan23():
    """Score the matched Fan 2023 (T5) held-out CORP replay.

    Returns
    -------
    dict
        Session and pooled greedy CER and character-LM WER in percent.
        Both axes use the same last six trials per session.
    """
    counts = {}
    with open(ROOT / FAN_TRIALS, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            name = row["session"]
            score = counts.setdefault(name, {"trials": set(), "ce": 0, "cc": 0,
                                             "we": 0, "wc": 0})
            trial = int(row["heldout_trial_index"])
            if trial in score["trials"]:
                raise ValueError(f"Fan 2023: duplicate held-out trial {name}/{trial}")
            score["trials"].add(trial)
            score["ce"] += editdistance.eval(row["reference"], row["greedy_text"])
            score["cc"] += len(row["reference"])
            score["we"] += int(row["beam_word_errors"])
            score["wc"] += int(row["reference_words"])
    sessions = []
    for name, score in sorted(counts.items()):
        n = len(score["trials"])
        if n != 6:
            raise ValueError(f"Fan 2023: expected six held-out trials in {name}, got {n}")
        sessions.append({"session": name, "n": n,
                         "cer": round(100 * score["ce"] / score["cc"], 2),
                         "wer": round(100 * score["we"] / score["wc"], 2),
                         "char_counts": score["cc"], "word_counts": score["wc"],
                         "char_errors": score["ce"], "word_errors": score["we"]})
    totals = {key: sum(score[key] for score in counts.values())
              for key in ("ce", "cc", "we", "wc")}
    return {
        "sessions": sessions,
        "pooled": {"cer": round(100 * totals["ce"] / totals["cc"], 2),
                   "wer": round(100 * totals["we"] / totals["wc"], 2)},
        "pooling": "micro (greedy character errors and character-LM word errors)",
    }


def build_jude26a(part):
    """Assemble a Jude 2026a multisession arm (``'t17'`` or ``'t18'``).

    Parameters
    ----------
    part : str
        Participant key inside ``lm_decode_summary.json``.

    Returns
    -------
    dict
        ``{"sessions": [...], "pooled": {...}}``; pooled CER is
        ``n_chars``-weighted (raw_cer is a per-session rate), pooled WER
        is the micro average over word error counts.
    """
    with open(ROOT / SOURCES[f"Jude 2026a ({part.upper()})"]["archive"],
              encoding="utf-8") as f:
        d = json.load(f)[part]
    sessions = []
    sum_ce = sum_cc = sum_we = sum_wc = 0
    for name in sorted(d):
        s = d[name]
        if s["n_trials"] < MIN_TRIALS:
            continue
        sessions.append({"session": name, "n": s["n_trials"],
                         "cer": round(100.0 * s["raw_cer"], 2),
                         "wer": round(100.0 * s["post_lm_wer"], 2),
                         "char_counts": int(s["n_chars"]),
                         "word_counts": int(s["n_words"]),
                         "word_errors": int(s["word_errs"])})
        sum_ce += s["raw_cer"] * s["n_chars"]; sum_cc += s["n_chars"]
        sum_we += s["word_errs"]; sum_wc += s["n_words"]
    return {
        "sessions": sessions,
        "pooled": {"cer": round(100.0 * sum_ce / sum_cc, 2),
                   "wer": round(100.0 * sum_we / sum_wc, 2)},
        "pooling": "CER = n_chars-weighted raw_cer; WER = micro over "
                   "post-LM word error counts",
    }


def main():
    """Build all four arms, print a reconciliation table, write the JSON.

    Returns
    -------
    None
        Writes ``session_cer_vs_wer.json`` next to this script.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PUBLISHED)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "Willett 2021 (T5)": build_willett21(),
        "Fan 2023 (T5)": build_fan23(),
        "Jude 2026a (T17)": build_jude26a("t17"),
        "Jude 2026a (T18)": build_jude26a("t18"),
    }
    tot_c = sum(s.get("char_counts", 0) for v in out.values()
                for s in v["sessions"])
    tot_w = sum(s.get("word_counts", 0) for v in out.values()
                for s in v["sessions"])
    out["_meta"] = {
        "min_trials": MIN_TRIALS,
        "chars_per_word_pooled": round(tot_c / tot_w, 3),
        "chars_per_word_note": "reference characters (spaces included, "
            "matching the CER denominators where available) per word, "
            "pooled across the four plotted arms",
        "pooling": {k: v["pooling"] for k, v in out.items() if k != "_meta"},
        "sources": SOURCES,
    }
    for k in [k for k in out if k != "_meta"]:
        out[k].pop("pooling", None)
    n_below = sum(1 for v in out.values()
                  if isinstance(v, dict) and "sessions" in v
                  for s in v["sessions"] if s["wer"] < s["cer"])
    n_tot = sum(len(v["sessions"]) for v in out.values()
                if isinstance(v, dict) and "sessions" in v)
    print(f"{'study arm':24s} {'sessions':>8s} {'pooled CER':>10s} "
          f"{'pooled WER':>10s}")
    for k, v in out.items():
        if k == "_meta":
            continue
        p = v["pooled"]
        print(f"{k:24s} {len(v['sessions']):8d} {p['cer']:9.2f}% "
              f"{p['wer']:9.2f}%")
    print(f"WER < CER in {n_below}/{n_tot} sessions; pooled chars/word "
          f"= {out['_meta']['chars_per_word_pooled']}")
    with open(args.output_dir / "session_cer_vs_wer.json", "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("wrote session_cer_vs_wer.json")


if __name__ == "__main__":
    main()
