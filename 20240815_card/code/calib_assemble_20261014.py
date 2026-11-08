# -*- coding: utf-8 -*-
"""Assemble the P1/P2-lite calibration comparison table.

Static baseline (no adaptation) per day comes from run_p4 val_greedy_per_trial.csv;
simulated arms from run_calib_20261014/*/summary.json (online PER = each trial
decoded BEFORE its label is used -- the honest metric; endpoint re-decode is
contaminated by memorization of disclosed trials and is not reported).

Usage: python calib_assemble_20261014.py
"""
import json
import os

import pandas as pd

REPO = r".\20240815_card"
CAL = os.path.join(REPO, r"data\20241009_card_BtoT25\run_calib_20261014")
P4CSV = os.path.join(REPO, r"data\20241009_card_BtoT25\run_p4_20261013\val_greedy_per_trial.csv")
OUT = os.path.join(REPO, r"result\calib_20261014_p1_summary.csv")

DAYS = ["t15.2025.01.10", "t15.2025.01.12", "t15.2025.03.14",
        "t15.2025.03.16", "t15.2025.03.30", "t15.2025.04.13",
        "t15.2023.08.18", "t15.2023.11.04"]


def main():
    """Assemble the calibration comparison table and write it to CSV.

    Reads the static per-trial greedy decoding CSV from the P4 run and the
    per-arm summary/trial files of the simulated online-calibration runs,
    computes static vs. online PER (including last-5-sentence windows), and
    writes one row per (session, lr, arm) to ``OUT``.

    Returns
    -------
    None
        Results are printed to stdout and saved to ``OUT``.
    """
    csv = pd.read_csv(P4CSV)
    static = {}
    static_last5 = {}
    for s, g in csv.groupby("session"):
        if s in DAYS:
            g = g.reset_index(drop=True)
            static[s] = 100 * g.edit_distance.sum() / g.true_len.sum()
            static_last5[s] = 100 * g.edit_distance.tail(5).sum() / g.true_len.tail(5).sum()

    rows = []
    for d in sorted(os.listdir(CAL)):
        p = os.path.join(CAL, d, "summary.json")
        if not os.path.exists(p):
            continue
        m = json.load(open(p))
        traj = pd.read_csv(os.path.join(CAL, d, "per_trial_online.csv"))
        last5 = 100 * traj.ed.tail(5).sum() / traj.len.tail(5).sum()
        rows.append({
            "session": m["session"], "lr": m["lr"], "arm": m["arm"],
            "static_PER": round(static.get(m["session"], float("nan")), 2),
            "online_PER": round(100 * m["online_PER"], 2),
            "last5_online": round(last5, 2),
            "last5_static_same_sentences": round(static_last5.get(m["session"], float("nan")), 2),
            "delta_vs_static": round(100 * m["online_PER"] - static.get(m["session"], float("nan")), 2),
            "ft_steps": m["total_ft_steps"], "wall_min": round(m["wall_min"], 1),
        })
    df = pd.DataFrame(rows).sort_values(["session", "lr", "arm"])
    df.to_csv(OUT, index=False)
    print(df.to_string(index=False))
    print(f"\nsaved: {OUT}")

    # pooled view over the six 2025 days per (lr, arm)
    six = df[df.session.str.startswith("t15.2025")]
    if len(six):
        print("\npooled six 2025 days:")
        for (lr, arm), g in six.groupby(["lr", "arm"]):
            print(f"  lr={lr} arm={arm}: mean online PER {g.online_PER.mean():.2f} "
                  f"(static mean {g.static_PER.mean():.2f}) over {len(g)} days")


if __name__ == "__main__":
    main()
