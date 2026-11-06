# -*- coding: utf-8 -*-
"""Fig S14-style within-session PER trajectories for the calibration sims.

One panel per 2025 session: rolling 5-trial PER for each available arm
(static = P4 without adaptation; copy/keep/eye = per-sentence fine-tune sims)
plus the first-trials cold-start annotation. Paper Fig S14 reference lines
(first-5 12.0% / first-20 5.8%) drawn for scale only -- different subject
regime, do not over-read.

Usage: python calib_plot_trajectories_20261014.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = r".\20240815_card"
CAL = os.path.join(REPO, r"data\20241009_card_BtoT25\run_calib_20261014")
P4CSV = os.path.join(REPO, r"data\20241009_card_BtoT25\run_p4_20261013\val_greedy_per_trial.csv")
OUT = os.path.join(REPO, r"result\figs\session_level\calib_20261014_trajectories.png")

DAYS = ["t15.2025.01.10", "t15.2025.01.12", "t15.2025.03.14",
        "t15.2025.03.16", "t15.2025.03.30", "t15.2025.04.13"]
COLORS = {"static": "#888888", "copy": "#1f77b4", "keep": "#2ca02c", "eye": "#d62728"}


def rolling_per(ed, ln, w=5):
    ed, ln = np.asarray(ed, float), np.asarray(ln, float)
    red = np.convolve(ed, np.ones(w), "valid")
    rln = np.convolve(ln, np.ones(w), "valid")
    y = np.r_[np.repeat(np.nan, w - 1), 100 * red / rln]
    y[:w - 1] = 100 * np.cumsum(ed[:w - 1]) / np.cumsum(ln[:w - 1])
    return y


def main():
    csv = pd.read_csv(P4CSV)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, s in zip(axes.ravel(), DAYS):
        g = csv[csv.session == s].reset_index(drop=True)
        ax.plot(g.index, rolling_per(g.edit_distance, g.true_len),
                color=COLORS["static"], lw=2, label="static P4 (no adapt)")
        for d in sorted(os.listdir(CAL)):
            p = os.path.join(CAL, d, "summary.json")
            if not os.path.exists(p):
                continue
            m = json.load(open(p))
            if m["session"] != s or m["lr"] != 0.004:
                continue
            t = pd.read_csv(os.path.join(CAL, d, "per_trial_online.csv"))
            ax.plot(t.i, rolling_per(t.ed, t.len), color=COLORS.get(m["arm"], "k"),
                    lw=1.6, alpha=0.9, label=f"{m['arm']} (online PER {100*m['online_PER']:.1f}%)")
        ax.axhline(10.3, color="k", ls=":", lw=1, alpha=0.5)
        ax.set_title(s, fontsize=11)
        ax.set_xlabel("trial (chronological)")
        ax.grid(alpha=0.25)
        ax.set_ylim(0, 80)
    axes[0, 0].set_ylabel("rolling-5 PER (%)")
    axes[1, 0].set_ylabel("rolling-5 PER (%)")
    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("S2.03 per-sentence fine-tune simulation on P4 (LR 0.004) — "
                 "dotted line: paper Fig S14 first-block avg 10.3%", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
