# -*- coding: utf-8 -*-
"""Overview dashboard of the 2026-08-31 BtoT24/BtoT25 reproduction run.

One figure, six panels:
A  BtoT24 training: val greedy PER trajectory + final test point + anchors
B  BtoT24 per-day test-partition PER (marker size = trials)
C  BtoT25 greedy PER by corpus
D  BtoT25 per-session PER across 20 months
E  LM-step contribution (from the paper's own online records; no LM run locally)
F  Headline: ours vs official/paper anchors
Output: .\shared\result\figs\btot_results_overview.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK = "#172033"
BLUE = "#2457c5"
GREEN = "#17683c"
RED = "#a83232"
AMBER = "#9a5b00"
MUTED = "#5c667a"

fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))

# ---------- A: BtoT24 training trajectory ----------
curves = pd.read_csv(r".\20230831_willett\data\20230616_willett_BtoT24\run_60k\curves.csv")
va = curves[(curves.metric == "val_PER")].sort_values("batch")
ax = axes[0, 0]
ax.plot(va.batch, 100 * va.value, marker="o", ms=4, color=GREEN, label="val greedy PER")
ax.axhline(19.7, ls="--", c=RED, lw=1.2, label="Willett 2023 paper raw PER 19.7%")
ax.axhspan(15.0, 16.7, color=AMBER, alpha=0.15, label="BtoT24 top teams val PER 15-16.7%")
ax.scatter([60000], [23.42], marker="*", s=220, color=AMBER, zorder=5,
           label=f"final test partition 23.42%")
ax.set_xlabel("training batch"); ax.set_ylabel("PER (%)")
ax.set_title("A  BtoT24: training trajectory (60k batches, 225.7 min)")
ax.legend(fontsize=7.5); ax.grid(alpha=0.3)

# ---------- B: BtoT24 per-day test PER ----------
tdf = pd.read_csv(r".\20230831_willett\data\20230616_willett_BtoT24\run_60k\test_partition_greedy_per_trial.csv")
day = (tdf.groupby("session")
          .apply(lambda g: 100 * g.edit_distance.sum() / g.true_len.sum(),
                 include_groups=False)
          .sort_index().astype(float))
cnt = tdf.groupby("session").size().reindex(day.index)
dates = pd.to_datetime([s.replace("t12.", "") for s in day.index], format="%Y.%m.%d")
ax = axes[0, 1]
ax.scatter(dates, day.values, s=10 * cnt.values, color=BLUE,
           edgecolors=INK, linewidths=0.4, zorder=3)
agg = 100 * tdf.edit_distance.sum() / tdf.true_len.sum()
ax.axhline(agg, ls="--", c=AMBER, lw=1.4, label=f"aggregate test PER {agg:.2f}%")
ax.axhline(19.7, ls="--", c=RED, lw=1.2, label="paper raw PER 19.7%")
for d, v, s in zip(dates, day.values, day.index):
    if v == day.max() or v == day.min():
        ax.annotate(f"{s[-5:]}\n{v:.1f}%", (d, v), fontsize=8,
                    xytext=(6, 6), textcoords="offset points")
ax.set_ylabel("test-partition greedy PER (%)")
ax.set_title("B  BtoT24: per-day test PER (worst = earliest sessions)")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

# ---------- C: BtoT25 by corpus ----------
b25 = pd.read_csv(r".\20240815_card\result\val_greedy_per_trial.csv")
byc = (b25.groupby("corpus")
          .apply(lambda g: 100 * g.edit_distance.sum() / g.true_len.sum(),
                 include_groups=False)
          .sort_values().astype(float))
agg25 = float(100 * b25.edit_distance.sum() / b25.true_len.sum())
ax = axes[0, 2]
bars = ax.bar(byc.index, byc.values, color=[GREEN if v < agg25 else BLUE for v in byc],
              width=0.62)
ax.axhline(agg25, ls="--", c=AMBER, lw=1.4, label=f"our aggregate {agg25:.2f}%")
ax.axhline(10.10, ls=":", c=RED, lw=1.4, label="official ckpt val PER 10.10%")
for b, v in zip(bars, byc.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.1f}", ha="center", fontsize=9)
ax.set_ylabel("greedy PER (%)"); ax.set_ylim(0, byc.max() * 1.2)
ax.set_title("C  BtoT25: greedy PER by corpus (n=1,426)")
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

# ---------- D: BtoT25 session timeline ----------
b25d = b25.copy()
b25d["date"] = pd.to_datetime(b25d.session.str.replace("t15.", "", regex=False),
                              format="%Y.%m.%d")
day25 = (b25d.groupby("date")
            .apply(lambda g: 100 * g.edit_distance.sum() / g.true_len.sum(),
                   include_groups=False)
            .sort_index().astype(float))
cnt25 = b25d.groupby("date").size().reindex(day25.index)
ax = axes[1, 0]
ax.scatter(day25.index, day25.values, s=8 * cnt25.values, color=BLUE,
           edgecolors=INK, linewidths=0.4, zorder=3)
ax.axhline(agg25, ls="--", c=AMBER, lw=1.4, label=f"aggregate {agg25:.2f}%")
for d, v in day25.items():
    if v == day25.max() or v == day25.min():
        ax.annotate(f"{d:%Y-%m}\n{v:.1f}%", (d, v), fontsize=8,
                    xytext=(5, 6), textcoords="offset points")
ax.set_ylabel("per-session greedy PER (%)")
ax.set_title("D  BtoT25: per-session PER over 20 months")
ax.legend(fontsize=8); ax.grid(alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

# ---------- E: LM-step contribution (paper's online records) ----------
ax = axes[1, 1]
labels = ["greedy PER\n(ours, offline)", "greedy PER\n(online record)",
          "PER after LM\n(online record)", "WER after LM\n(online record)"]
vals = [10.19, 9.54, 2.66, 4.34]
cols = [BLUE, BLUE, GREEN, GREEN]
bars = ax.bar(labels, vals, color=cols, width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}%", ha="center",
            fontsize=10, fontweight="bold")
ax.annotate("", xy=(2, 2.9), xytext=(1, 9.2),
            arrowprops=dict(arrowstyle="->", color=RED, lw=1.6))
ax.text(1.5, 6.3, "LM step\n(not reproduced,\nquantified from records)",
        fontsize=8.5, color=RED, ha="center")
ax.set_ylabel("error rate (%)"); ax.set_ylim(0, 11.5)
ax.set_title("E  BtoT25: what the excluded LM step contributes")
ax.grid(axis="y", alpha=0.3)

# ---------- F: headline comparison ----------
ax = axes[1, 2]
rows = [
    ("BtoT25 greedy PER — ours", 10.19, BLUE),
    ("BtoT25 greedy PER — official ckpt", 10.10, MUTED),
    ("BtoT24 greedy PER — ours (test)", 23.42, BLUE),
    ("BtoT24 raw PER — Willett 2023 paper", 19.70, MUTED),
    ("BtoT24 val PER — BtoT24 top teams", 15.85, MUTED),
]
ypos = range(len(rows), 0, -1)
for y, (name, v, c) in zip(ypos, rows):
    ax.barh(y, v, color=c, height=0.55)
    ax.text(v + 0.3, y, f"{v:.2f}%", va="center", fontsize=10, fontweight="bold")
ax.set_yticks(list(ypos)); ax.set_yticklabels([r[0] for r in rows], fontsize=9)
ax.set_xlim(0, 27); ax.set_xlabel("phoneme error rate (%)")
ax.set_title("F  Headline: reproduction vs anchors")
ax.grid(axis="x", alpha=0.3)

fig.suptitle("BtoT24 / BtoT25 reproduction to the pre-LM step — run of 2026-08-31 "
             "(Win11 · RTX 5060 Ti · official Card-2024 PyTorch RNN)",
             fontsize=14, y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = r".\shared\result\figs\btot_results_overview.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
print("wrote:", out)
