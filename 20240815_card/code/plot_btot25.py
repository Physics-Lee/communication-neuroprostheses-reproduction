# -*- coding: utf-8 -*-
"""BtoT25 result figures from val_greedy_per_trial.csv (Phase-1 reproduction).

Outputs (to .\20240815_card):
- btot25_per_by_corpus.png   : aggregate + per-corpus greedy PER
- btot25_per_timeline.png    : per-session greedy PER over 20 months,
                               marker size = trial count
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

CSV = r".\20240815_card\result\val_greedy_per_trial.csv"
SUMMARY = r".\20240815_card\result\val_greedy_summary.json"
OUT1 = r".\20240815_card\result\figs\legacy\btot25_per_by_corpus.png"
OUT2 = r".\20240815_card\result\figs\legacy\btot25_per_timeline.png"

df = pd.read_csv(CSV)

# ---- fig 1: PER by corpus ----
by_corpus = df.groupby("corpus").apply(
    lambda g: 100 * g["edit_distance"].sum() / g["true_len"].sum(),
    include_groups=False).sort_values()
by_corpus = by_corpus.astype(float)
overall = float(100 * df["edit_distance"].sum() / df["true_len"].sum())
with open(SUMMARY, encoding="utf-8") as f:
    official = 100 * json.load(f)["checkpoint_saved_val_PER"]

fig, ax = plt.subplots(figsize=(9, 5))
colors = ["#17683c" if v < overall else "#2457c5" for v in by_corpus]
bars = ax.bar(by_corpus.index, by_corpus.values, color=colors, width=0.62)
ax.axhline(overall, ls="--", c="#9a5b00", lw=1.5,
           label=f"our aggregate {overall:.2f}%")
ax.axhline(official, ls=":", c="#a83232", lw=1.5,
           label=f"official ckpt val_PER {official:.2f}%")
for b, v in zip(bars, by_corpus.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.2f}%",
            ha="center", fontsize=10)
ax.set_ylabel("greedy-CTC phoneme error rate (%)")
ax.set_title("BtoT25 offline reproduction: greedy PER by corpus\n"
             "(pretrained RNN, validation partition, n=1,426 trials)")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(by_corpus.max(), overall) * 1.18)
fig.tight_layout()
fig.savefig(OUT1, dpi=150)
plt.close(fig)

# ---- fig 2: per-session PER timeline ----
df["date"] = pd.to_datetime(df["session"].str.replace("t15.", "", regex=False),
                            format="%Y.%m.%d")
day = df.groupby("date").apply(
    lambda g: 100 * g["edit_distance"].sum() / g["true_len"].sum(),
    include_groups=False).sort_index().astype(float)
cnt = df.groupby("date").size().reindex(day.index)

fig, ax = plt.subplots(figsize=(12, 5))
ax.scatter(day.index, day.values, s=8 * cnt.values, color="#2457c5",
           edgecolors="#172033", linewidths=0.4, zorder=3)
ax.axhline(overall, ls="--", c="#9a5b00", lw=1.5,
           label=f"aggregate {overall:.2f}%")
for d, v in day.items():
    if v == day.max() or v == day.min():
        ax.annotate(f"{d:%Y-%m}\n{v:.1f}%", (d, v), fontsize=8,
                    xytext=(5, 8 if v > overall else -16),
                    textcoords="offset points")
ax.set_ylabel("per-session greedy PER (%)")
ax.set_xlabel("recording session (2023-08 → 2025-04, 45 sessions)")
ax.set_title("BtoT25: per-session greedy PER across 20 months\n"
             "(marker size = number of val trials)")
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
ax.grid(alpha=0.3); ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT2, dpi=150)
plt.close(fig)

print("wrote:", OUT1)
print("wrote:", OUT2)
print(f"\naggregate {overall:.2f}% | official {official:.2f}%")
print("best session:", day.idxmin().date(), f"{day.min():.2f}%")
print("worst session:", day.idxmax().date(), f"{day.max():.2f}%")
