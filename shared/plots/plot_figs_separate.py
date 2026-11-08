# -*- coding: utf-8 -*-
"""Six separate result figures (panels A-F of the overview dashboard).

Output: D:\\repositories\\shared\\figs\\willett2023_btot24_training_trajectory.png ... willett2023_card2024_headline_anchors.png
Data sources: run_60k CSVs + val_greedy_per_trial.csv (all reproducible).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

INK = "#172033"; BLUE = "#2457c5"; GREEN = "#17683c"; RED = "#a83232"
AMBER = "#9a5b00"; MUTED = "#5c667a"
OUT = r".\shared\result\figs"


def rot_labels(ax):
    """Rotate a figure's x tick labels for readability.

    Sets the x tick labels of ``ax`` to a 30-degree right-anchored
    rotation.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes whose tick labels should be rotated.
    """
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")


# ---------- A: BtoT24 training trajectory ----------
curves = pd.read_csv(r".\20230831_willett\data\20230616_willett_BtoT24\run_60k\curves.csv")
va = curves[curves.metric == "val_PER"].sort_values("batch")
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(va.batch, 100 * va.value, marker="o", ms=5, color=GREEN, label="val greedy PER")
ax.axhline(19.7, ls="--", c=RED, lw=1.2, label="Willett 2023 paper raw PER 19.7%")
ax.axhspan(15.0, 16.7, color=AMBER, alpha=0.15, label="BtoT24 top teams val PER 15-16.7%")
ax.scatter([60000], [23.42], marker="*", s=260, color=AMBER, zorder=5,
           label="final test partition 23.42%")
ax.set_xlabel("training batch"); ax.set_ylabel("PER (%)")
ax.set_title("BtoT24: training trajectory\n(60k batches, 225.7 min on RTX 5060 Ti)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/training_trajectory/willett2023_btot24_training_trajectory.png", dpi=150)
plt.close(fig)

# ---------- B: BtoT24 per-day test PER ----------
tdf = pd.read_csv(r".\20230831_willett\data\20230616_willett_BtoT24\run_60k\test_partition_greedy_per_trial.csv")
day = (tdf.groupby("session")
          .apply(lambda g: 100 * g.edit_distance.sum() / g.true_len.sum(),
                 include_groups=False).sort_index().astype(float))
cnt = tdf.groupby("session").size().reindex(day.index)
dates = pd.to_datetime([s.replace("t12.", "") for s in day.index], format="%Y.%m.%d")
fig, ax = plt.subplots(figsize=(11, 5.5))
# per-session scatter: no connecting line (sessions are discrete) and no color
# encoding of PER (the y-axis already shows it); marker size encodes trial count
ax.scatter(dates, day.values, s=10 * cnt.values, color=BLUE,
           edgecolors=INK, linewidths=0.4, zorder=3)
agg = 100 * tdf.edit_distance.sum() / tdf.true_len.sum()
ax.axhline(agg, ls="--", c=AMBER, lw=1.4, label=f"aggregate test PER {agg:.2f}%")
ax.axhline(19.7, ls="--", c=RED, lw=1.2, label="paper raw PER 19.7%")
for d, v, s in zip(dates, day.values, day.index):
    if v == day.max() or v == day.min():
        ax.annotate(f"{s[-5:]}\n{v:.1f}%", (d, v), fontsize=9,
                    xytext=(6, 6), textcoords="offset points")
ax.set_ylabel("test-partition greedy PER (%)"); ax.set_xlabel("recording session")
ax.set_title("BtoT24: per-day test-partition PER\n(marker size = number of trials; worst days are the earliest sessions)")
# size legend: reference marker sizes in trial counts
for n_ref in (20, 30, 40):
    ax.scatter([], [], s=10 * n_ref, color=MUTED, edgecolors=INK, linewidths=0.4,
               label=f"{n_ref} trials")
ax.legend(fontsize=9, labelspacing=1.3, borderpad=0.9); ax.grid(alpha=0.3); rot_labels(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/session_level/willett2023_btot24_per_day_test_per.png", dpi=150)
plt.close(fig)

# ---------- C: BtoT25 by corpus ----------
b25 = pd.read_csv(r".\20240815_card\result\val_greedy_per_trial.csv")
byc = (b25.groupby("corpus")
          .apply(lambda g: 100 * g.edit_distance.sum() / g.true_len.sum(),
                 include_groups=False).sort_values().astype(float))
agg25 = float(100 * b25.edit_distance.sum() / b25.true_len.sum())
fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.bar(byc.index, byc.values, color=[GREEN if v < agg25 else BLUE for v in byc],
              width=0.62)
ax.axhline(agg25, ls="--", c=AMBER, lw=1.4, label=f"our aggregate {agg25:.2f}%")
ax.axhline(10.10, ls=":", c=RED, lw=1.4, label="official ckpt val PER 10.10%")
for b, v in zip(bars, byc.values):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.25, f"{v:.2f}%", ha="center", fontsize=10)
ax.set_ylabel("greedy PER (%)"); ax.set_ylim(0, byc.max() * 1.2)
ax.set_title("BtoT25: greedy PER by corpus\n(pretrained RNN, validation partition, n=1,426)")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/summary/card2024_btot25_per_by_corpus.png", dpi=150)
plt.close(fig)

# ---------- D: BtoT25 session timeline ----------
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
# size legend: reference marker sizes in trial counts
for n_ref in (10, 25, 50):
    ax.scatter([], [], s=8 * n_ref, color=MUTED, edgecolors=INK, linewidths=0.4,
               label=f"{n_ref} trials")
ax.legend(fontsize=9, labelspacing=1.3, borderpad=0.9); ax.grid(alpha=0.3); rot_labels(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/session_level/card2024_btot25_session_timeline.png", dpi=150)
plt.close(fig)

# ---------- E: LM-step contribution ----------
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
fig.tight_layout(); fig.savefig(f"{OUT}/summary/card2024_btot25_lm_contribution.png", dpi=150)
plt.close(fig)

# ---------- F: headline comparison ----------
fig, ax = plt.subplots(figsize=(10, 5))
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
    ax.text(v + 0.3, y, f"{v:.2f}%", va="center", fontsize=11, fontweight="bold")
ax.set_yticks(list(ypos)); ax.set_yticklabels([r[0] for r in rows], fontsize=10)
ax.set_xlim(0, 27); ax.set_xlabel("phoneme error rate (%)")
ax.set_title("Headline: reproduction vs anchors\n(all numbers greedy/acoustic-only except where noted)")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/summary/willett2023_card2024_headline_anchors.png", dpi=150)
plt.close(fig)

print("wrote 6 figures to", OUT)
