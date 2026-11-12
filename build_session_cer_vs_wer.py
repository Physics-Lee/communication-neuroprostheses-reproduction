#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Rebuild `session_cer_vs_wer.json`: session-level raw CER (no LM) vs.
post-LM WER for the four character-level study arms we reproduced.

This is the character-level analog of `build_session_per_vs_wer.py`
(phoneme-level, consumed by Figure 2 panel (g)) and feeds the WER-vs-CER
phase-diagram panel of `render_phase_memory_figure.py`.

Study arms and sources
----------------------
Willett et al., 2021 (T5, handwriting, bigram LM), 10 sessions:
    raw greedy CER from the self-trained PyTorch port's per-session
    ``errCounts.mat`` files in the sibling repository
    ``20261009_reproduce_20210512_willett`` (``*.mat`` is gitignored, so
    the verified per-session counts are embedded below; see
    ``WILLET21_RAW_CER`` provenance), post-LM WER from the archived
    ``20210512_willett/results/m3_own_decode_HeldOutTrials.csv``.
Fan et al., 2023 (T5, CORP recalibration arm, 5-gram+LM):
    per-session post-adaptation CER from
    ``20231201_fan/results/tier3_corp_sim_v6.json`` (``post`` field),
    post-LM WER from
    ``20231201_fan/results/rescore_official_outputs.json``
    (``arms.recalibration.per_session[*].wer_lm_pct``).
Jude et al., 2026a (T17 / T18, typing, 5-gram WFST + LM):
    ``20260316_jude/results/runs_multisess/lm_decode_summary.json``
    (``raw_cer`` = multisession greedy CER, ``post_lm_wer`` / word error
    counts after the 5-gram + LM stage).

Conventions (mirror `build_session_per_vs_wer.py`)
--------------------------------------------------
- Sessions with fewer than ``MIN_TRIALS`` scored trials are dropped
  (removes ``t18.2024.12.04``, n=1).
- Pooled study values are micro-averages from raw counts where counts
  exist (Willett both axes, Jude WER, Jude CER weighted by ``n_chars``);
  Fan's CER is a session mean because the v6 simulation archives rates,
  not counts. The pooling mode per arm is recorded in ``_meta``.
- Reference characters-per-word (spaces included, matching the CER
  denominators) is accumulated from the same trials and stored in
  ``_meta.chars_per_word_pooled`` for the no-LM baseline curve
  ``1 - (1 - CER)^L`` of the phase diagram.

Usage:
    python build_session_cer_vs_wer.py
"""
import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MIN_TRIALS = 3

# Willett 2021 HeldOutTrials raw greedy CER per session, self-trained port.
# Provenance: D:\\repositories\\20261009_reproduce_20210512_willett\\data\\
# 20210512_willett\\runs\\HeldOutTrials_synth\\step5_inference\\<session>_errCounts.mat
# (per-session charCounts/charErrors arrays, 0-based session-local test
# indices from handwritingBCIData/RNNTrainingSteps/
# trainTestPartitions_HeldOutTrials.mat, same aggregation as
# 20210512_willett/code/step5_eval_from_errcounts.py). Pooled = 4.71% CER,
# 63 sentences -- matches the Figure 2 panel (e) "ours" anchor.
WILLET21_RAW_CER = {
    "t5.2019.05.08": {"n": 10, "char_errs": 31.0, "char_counts": 693.0},
    "t5.2019.11.25": {"n": 7, "char_errs": 19.0, "char_counts": 399.0},
    "t5.2019.12.09": {"n": 7, "char_errs": 15.0, "char_counts": 362.0},
    "t5.2019.12.11": {"n": 7, "char_errs": 13.0, "char_counts": 274.0},
    "t5.2019.12.18": {"n": 7, "char_errs": 21.0, "char_counts": 453.0},
    "t5.2019.12.20": {"n": 7, "char_errs": 11.0, "char_counts": 375.0},
    "t5.2020.01.06": {"n": 5, "char_errs": 12.0, "char_counts": 273.0},
    "t5.2020.01.08": {"n": 7, "char_errs": 19.0, "char_counts": 387.0},
    "t5.2020.01.13": {"n": 3, "char_errs": 13.0, "char_counts": 140.0},
    "t5.2020.01.15": {"n": 3, "char_errs": 10.0, "char_counts": 128.0},
}

# Fan 2023 recalibration-arm reference character counts per session
# (chars incl. '>'-encoded spaces, '~' sentence-end markers dropped).
# Provenance: CORP_data_release.zip online_evaluation_data/recalibration/
# mat/*.mat in the sibling repository 20261022_reproduce_20231201_fan
# (word counts come from rescore_official_outputs.json n_ref_words).
FAN_RECAL_CHARS = {
    "t5.2022.09.29": 1597, "t5.2022.10.06": 1686, "t5.2022.10.18": 1673,
    "t5.2022.10.25": 1598, "t5.2022.10.27": 1708, "t5.2022.11.01": 1593,
    "t5.2022.11.03": 1695, "t5.2022.12.08": 1718, "t5.2022.12.15": 858,
    "t5.2023.02.28": 1662,
}


def build_willett21():
    """Assemble the Willett 2021 (T5) session table.

    Returns
    -------
    dict
        ``{"sessions": [...], "pooled": {...}}`` with per-session
        ``(cer, wer)`` in percent and micro-averaged pooled values.
    """
    path = os.path.join(HERE, "20210512_willett", "results",
                        "m3_own_decode_HeldOutTrials.csv")
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
        raw = WILLET21_RAW_CER[s]
        if raw["n"] < MIN_TRIALS:
            continue
        cer = 100.0 * raw["char_errs"] / raw["char_counts"]
        wer = 100.0 * agg[s]["we"] / agg[s]["wc"]
        sessions.append({"session": s, "n": raw["n"],
                         "cer": round(cer, 2), "wer": round(wer, 2),
                         "char_counts": int(raw["char_counts"]),
                         "word_counts": agg[s]["wc"]})
        tot_ce += raw["char_errs"]; tot_cc += raw["char_counts"]
        tot_we += agg[s]["we"]; tot_wc += agg[s]["wc"]
    return {
        "sessions": sessions,
        "pooled": {"cer": round(100.0 * tot_ce / tot_cc, 2),
                   "wer": round(100.0 * tot_we / tot_wc, 2)},
        "pooling": "micro (errCounts char errors x m3 bigram word errors)",
    }


def build_fan23():
    """Assemble the Fan 2023 (T5) CORP recalibration-arm session table.

    Returns
    -------
    dict
        ``{"sessions": [...], "pooled": {...}}``; pooled CER is the
        session mean (v6 archives rates only), pooled WER is the
        ``n_ref_words``-weighted micro average of ``wer_lm_pct``.
    """
    with open(os.path.join(HERE, "20231201_fan", "results",
                           "tier3_corp_sim_v6.json"), encoding="utf-8") as f:
        v6 = json.load(f)
    with open(os.path.join(HERE, "20231201_fan", "results",
                           "rescore_official_outputs.json"),
              encoding="utf-8") as f:
        rs = json.load(f)
    wer = {s["session"]: s for s in
           rs["arms"]["recalibration"]["per_session"]}
    sessions = []
    sum_cer = sum_we = sum_wc = n_used = 0
    for s in v6["sessions"]:
        w = wer[s["session"]]
        if s["n"] < MIN_TRIALS:
            continue
        sessions.append({"session": s["session"], "n": s["n"],
                         "cer": round(float(s["post"]), 2),
                         "wer": round(float(w["wer_lm_pct"]), 2),
                         "char_counts": FAN_RECAL_CHARS[s["session"]],
                         "word_counts": int(w["n_ref_words"])})
        sum_cer += float(s["post"])
        sum_we += float(w["wer_lm_pct"]) * w["n_ref_words"]
        sum_wc += w["n_ref_words"]
        n_used += 1
    return {
        "sessions": sessions,
        "pooled": {"cer": round(sum_cer / n_used, 2),
                   "wer": round(sum_we / sum_wc, 2)},
        "pooling": "CER = session mean (v6 archives rates only); "
                   "WER = n_ref_words-weighted micro average",
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
    with open(os.path.join(HERE, "20260316_jude", "results",
                           "runs_multisess", "lm_decode_summary.json"),
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
                         "word_counts": int(s["n_words"])})
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
            "matching the CER denominators) per word, pooled across the "
            "four plotted arms; Fan counts embedded from the sibling CORP "
            "release (FAN_RECAL_CHARS)",
        "pooling": {k: v["pooling"] for k, v in out.items() if k != "_meta"},
        "sources": {
            "Willett 2021 (T5)": "20210512_willett/results/"
                "m3_own_decode_HeldOutTrials.csv + WILLET21_RAW_CER "
                "(sibling errCounts.mat constants, see module docstring)",
            "Fan 2023 (T5)": "20231201_fan/results/tier3_corp_sim_v6.json "
                "+ rescore_official_outputs.json (recalibration arm)",
            "Jude 2026a (T17)": "20260316_jude/results/runs_multisess/"
                "lm_decode_summary.json (t17)",
            "Jude 2026a (T18)": "20260316_jude/results/runs_multisess/"
                "lm_decode_summary.json (t18)",
        },
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
    with open(os.path.join(HERE, "session_cer_vs_wer.json"), "w",
              encoding="utf-8", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print("wrote session_cer_vs_wer.json")


if __name__ == "__main__":
    main()
