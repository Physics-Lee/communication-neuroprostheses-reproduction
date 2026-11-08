# -*- coding: utf-8 -*-
"""Card 2024 result figures regenerated from repo data (2026-10-13 P3b).

Card-only adaptation of the archive's code/plots/plot_figs_separate.py
panels C/D/E, with all inputs pinned to this repo (path split 2026-10-08):
  C: summary/card2024_btot25_per_by_corpus.png  <- result/val_greedy_per_trial.csv
  D: session_level/card2024_btot25_session_timeline.png <- same
  E: summary/card2024_btot25_lm_contribution.png <- static anchor values
Run: D:/apps/psychopy/python.exe code/plot_card24_figs.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK = "#172033"; BLUE = "#2457c5"; GREEN = "#17683c"; RED = "#a83232"
AMBER = "#9a5b00"; MUTED = "#5c667a"
FIGS = r".\20240815_card\result\figs"
CSV = r".\20240815_card\result\val_greedy_per_trial.csv"
SUMMARY = r".\20240815_card\result\val_greedy_summary.json"


def rot_labels(ax):
    """Rotate a matplotlib axis' x tick labels for readability.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis whose x tick labels should be rotated 30 degrees and
        right-aligned.

    Returns
    -------
    None
        The axis is modified in place.
    """
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


b25 = pd.read_csv(CSV)
byc = (b25.groupby("corpus")
          .apply(lambda g: 100 * g.edit_distance.sum() / g.true_len.sum(),
                 include_groups=False).sort_values().astype(float))
agg25 = float(100 * b25.edit_distance.sum() / b25.true_len.sum())

# ---------- C: by corpus ----------
fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar(byc.index, byc.values,
              color=[GREEN if v < agg25 else BLUE for v in byc], width=0.62)
ax.axhline(agg25, ls="--", c=AMBER, lw=1.4, label=f"our aggregate {agg25:.2f}%")
with open(SUMMARY, encoding="utf-8") as f:
    official = 100 * json.load(f)["checkpoint_saved_val_PER"]
ax.axhline(official, ls=":", c=RED, lw=1.4,
           label=f"official ckpt val PER {official:.2f}%")
for b, v in zip(bars, byc.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.2f}%",
            ha="center", fontsize=10)
ax.set_ylabel("greedy PER (%)"); ax.set_ylim(0, byc.max() * 1.2)
ax.set_title("BtoT25: greedy PER by corpus\n(pretrained RNN, validation partition, n=1,426)")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIGS}/summary/card2024_btot25_per_by_corpus.png", dpi=150)
plt.close(fig)

# ---------- D: session timeline ----------
b25d = b25.copy()
b25d["date"] = pd.to_datetime(b25d.session.str.replace("t15.", "", regex=False),
                              format="%Y.%m.%d")
day25 = (b25d.groupby("date")
            .apply(lambda g: 100 * g.edit_distance.sum() / g.true_len.sum(),
                   include_groups=False).sort_index().astype(float))
cnt25 = b25d.groupby("date").size().reindex(day25.index)
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.scatter(day25.index, day25.values, s=8 * cnt25.values, color=BLUE,
           edgecolors=INK, linewidths=0.4, zorder=3)
ax.axhline(agg25, ls="--", c=AMBER, lw=1.4, label=f"aggregate {agg25:.2f}%")
for d, v in day25.items():
    if v == day25.max() or v == day25.min():
        ax.annotate(f"{d:%Y-%m}\n{v:.1f}%", (d, v), fontsize=9,
                    xytext=(5, 6), textcoords="offset points")
ax.set_ylabel("per-session greedy PER (%)"); ax.set_xlabel("recording session")
ax.set_title("BtoT25: per-session greedy PER across 20 months\n(marker size = number of val trials)")
for n_ref in (10, 25, 50):
    ax.scatter([], [], s=8 * n_ref, color=MUTED, edgecolors=INK, linewidths=0.4,
               label=f"{n_ref} trials")
ax.legend(fontsize=9, labelspacing=1.3, borderpad=0.9); ax.grid(alpha=0.3)
rot_labels(ax)
fig.tight_layout()
fig.savefig(f"{FIGS}/session_level/card2024_btot25_session_timeline.png", dpi=150)
plt.close(fig)

# ---------- E: LM-step contribution (static anchor values) ----------
fig, ax = plt.subplots(figsize=(9, 5.5))
labels = ["greedy PER\n(ours, offline)", "greedy PER\n(online record)",
          "PER after LM\n(online record)", "WER after LM\n(online record)"]
vals = [10.19, 9.54, 2.66, 4.34]
bars = ax.bar(labels, vals, color=[BLUE, BLUE, GREEN, GREEN], width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}%", ha="center",
            fontsize=11, fontweight="bold")
ax.annotate("", xy=(2, 2.9), xytext=(1, 9.2),
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.8))
ax.text(1.5, 6.2, "LM step\n(not reproduced,\nquantified from\npaper's own records)",
        fontsize=9, color=RED, ha="center")
ax.set_ylabel("error rate (%)"); ax.set_ylim(0, 11.5)
ax.set_title("BtoT25: what the excluded LM step contributes\n(n-gram + reranking ran on the authors' online system; no LM run locally)")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(f"{FIGS}/summary/card2024_btot25_lm_contribution.png", dpi=150)
plt.close(fig)

print("wrote 3 card figures to", FIGS)
