# -*- coding: utf-8 -*-
"""Figures for the kunz + jude 20260428 CTC-pipeline runs (companion of
plot_figs_separate.py conventions; paper-prefixed outputs in D:\\repositories\\shared\\figs\\).

  willett2023_card2024_kunz2025_jude2026_greedy_per_spectrum
      five-participant same-pipeline difficulty spectrum (bootstrap 95% CI)
  kunz2025_greedy_per_vs_author_online_wer
      ours vs author online WER on the same eval trials
  jude2026_per_session_greedy_per_vs_author_wer
      per-session grouped bars (+vocab/loop breakdowns)
  jude2026_per_session_greedy_per
      per-session scatter in the willett2023/card2024 style
      (single color, marker size = trial count, paper anchor line)
  kunz2025_training_trajectories / jude2026_training_trajectory
      val greedy PER vs training batch (one figure per dataset)

Author-online-WER numbers are hardcoded from the recomputation scripts
(decodedSentenceText / clock-aligned ngram_decoder_final_output), see the
note 20260901_kunz与jude意图说话数据集_CTC管线实跑记录.html for provenance.
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK = "#172033"; BLUE = "#2457c5"; GREEN = "#009E73"; RED = "#a83232"
AMBER = "#E69F00"; MUTED = "#5c667a"; PURPLE = "#6d3fa4"
OUT = r".\shared\result\figs"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False


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

# ---------------- data ----------------
K12 = pd.read_csv(r".\20250821_kunz\data\20250821_kunz\run_t12_mixed\val_greedy_per_trial.csv")
K15 = pd.read_csv(r".\20250821_kunz\data\20250821_kunz\run_t15_mixed\val_greedy_per_trial.csv")
K16 = pd.read_csv(r".\20250821_kunz\data\20250821_kunz\run_t16_mixed\val_greedy_per_trial.csv")
JU = pd.read_csv(r".\20260428_jude\data\20260428_jude\run_25k\val_greedy_per_trial.csv")
B24 = pd.read_csv(r".\20230831_willett\data\20230616_willett_BtoT24\run_60k\test_partition_greedy_per_trial.csv")
B25 = pd.read_csv(r".\20240815_card\result\val_greedy_per_trial.csv")  # BtoT25 official-ckpt eval


def agg(df):
    """Compute the pooled greedy PER (percent) of a per-trial DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-trial frame with ``edit_distance`` and ``true_len`` columns.

    Returns
    -------
    float
        100 * sum(edit_distance) / sum(true_len).
    """
    return 100 * df.edit_distance.sum() / df.true_len.sum()


def boot_ci(df, n_boot=2000, seed=0):
    """Bootstrap 95% confidence interval for the pooled PER.

    Resamples per-trial rows with replacement and recomputes the pooled
    ratio each time, processing the replicates in chunks for memory
    efficiency.

    Parameters
    ----------
    df : pandas.DataFrame
        Per-trial frame with ``edit_distance`` and ``true_len`` columns.
    n_boot : int, optional
        Number of bootstrap replicates.
    seed : int, optional
        Seed for the numpy random generator (reproducible).

    Returns
    -------
    tuple of (float, float)
        ``(lo, hi)`` percentile-2.5 and percentile-97.5 bounds of the
        pooled PER in percent.
    """
    rng = np.random.default_rng(seed)
    ed = df.edit_distance.to_numpy()
    ln = df.true_len.to_numpy()
    m = len(ed)
    vals = []
    chunk = 500
    for s in range(0, n_boot, chunk):
        k = min(chunk, n_boot - s)
        idx = rng.integers(0, m, (k, m))
        vals.append(ed[idx].sum(1) / ln[idx].sum(1))
    v = np.concatenate(vals)
    return 100 * np.percentile(v, 2.5), 100 * np.percentile(v, 97.5)


# author online WER, recomputed from released decode streams (see note)
KUNZ_AUTH = {  # (participant, session) -> WER%
    ("T12", "t12.2023.11.28"): 22.85,
    ("T15", "t15.2024.03.03"): 14.29,
    ("T15", "t15.2024.12.15"): 26.53,
    ("T15", "t15.2025.01.05"): 60.16,
    ("T16", "t16.2024.04.29"): 30.69,
    ("T16", "t16.2024.12.18"): 44.39,
}
JUDE_AUTH = {"t17.2024.03.26": 30.98, "t17.2024.04.02": 91.80,
             "t17.2024.04.17": 110.34, "t17.2024.05.07": 49.81,
             "t17.2024.05.14": 63.94, "t17.2024.05.28": 72.65,
             "t17.2024.08.12": 36.69, "t17.2024.08.13": 64.71}

# ---------------- G: five-participant spectrum ----------------
# all-English figure; x = participant/task, y = PER. X tick labels (4 lines):
# task / vocabulary composition / dataset provenance (yyyymmdd_author, the
# repo dataset folder convention) / val+train trial counts.
# ordered by dataset date (yyyymmdd_author): willett -> card -> kunz -> jude
rows = [
    ("T12 · overt", "125k vocab", B24, GREEN,
     "20230616_willett", "8,800"),
    ("T15 · overt (copy task, ckpt)", "125k + 50 vocab", B25, GREEN,
     "20241009_card", "9,000"),
    ("T15 · inner", "125k + 50 vocab", K15, AMBER,
     "20250821_kunz", "615"),
    ("T12 · inner", "50 vocab", K12, AMBER,
     "20250821_kunz", "400"),
    ("T16 · inner", "125k + 50 vocab", K16, AMBER,
     "20250821_kunz", "365"),
    ("T17 · intended (locked-in)", "125k + 50 vocab", JU, BLUE,
     "20260428_jude", "3,107"),
]
fig, ax = plt.subplots(figsize=(11, 6.6))
vals, errs, colors = [], [], []
for name, vocab, df, c, src, ntr in rows:
    vals.append(agg(df)); errs.append(boot_ci(df)); colors.append(c)
x = np.arange(len(rows))
vals = np.array(vals); errs = np.array(errs)
ax.bar(x, vals, color=colors, alpha=0.9, width=0.6,
       yerr=[vals - errs[:, 0], errs[:, 1] - vals],
       error_kw=dict(ecolor=INK, lw=1.1, capsize=3.5))
for xi, v, ci in zip(x, vals, errs):
    ax.text(xi, v + 2.2, f"{v:.2f}\n[{ci[0]:.1f}, {ci[1]:.1f}]",
            ha="center", fontsize=9, color=INK)
ax.set_xticks(x)
ax.set_xticklabels([f"{name}\n{vocab}\n{src}\nval {len(df):,} / train {ntr}"
                    for name, vocab, df, c, src, ntr in rows], fontsize=8.4)
ax.set_ylabel("Greedy CTC phoneme error rate PER (%)")
ax.set_xlabel("participant / task")
ax.set_ylim(0, 78)
from matplotlib.patches import Patch as _Patch
ax.legend(handles=[_Patch(color=GREEN, label="overt speech"),
                   _Patch(color=BLUE, label="intended speech"),
                   _Patch(color=AMBER, label="inner speech")],
          fontsize=10, loc="upper left")
ax.set_title("Decoding difficulty spectrum under the same GRU+CTC+greedy pipeline (six models)\n"
             "overt speech < intended speech < inner speech", fontsize=12)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}\\summary\\willett2023_card2024_kunz2025_jude2026_greedy_per_spectrum.png", dpi=150); plt.close(fig)

# ---------------- H: kunz ours vs author online ----------------
groups = [
    ("T12\n50w", K12, "t12.2023.11.28"),
    ("T15\n50w", K15[K15.session.str.contains("03.03")], "t15.2024.03.03"),
    ("T15\n125k Ⅰ", K15[K15.session.str.contains("12.15")], "t15.2024.12.15"),
    ("T15\n125k Ⅱ", K15[K15.session.str.contains("01.05")], "t15.2025.01.05"),
    ("T16\n50w", K16[K16.session.str.contains("04.29")], "t16.2024.04.29"),
    ("T16\n125k", K16[K16.session.str.contains("12.18")], "t16.2024.12.18"),
]
fig, ax = plt.subplots(figsize=(10.5, 5.8))
x = np.arange(len(groups)); w = 0.38
ours = [agg(g[1]) for g in groups]
auth = [KUNZ_AUTH[(g[0].split("\n")[0], g[2])] for g in groups]
b1 = ax.bar(x - w / 2, ours, w, color=BLUE, alpha=0.9, label="本管线贪心 PER（无 LM）")
b2 = ax.bar(x + w / 2, auth, w, color=MUTED, alpha=0.85, hatch="//",
            label="作者在线 WER（RNN+5gram(+OPT)，decodedSentenceText 复算）")
for xi, v, g in zip(x, ours, groups):
    ax.text(xi - w / 2, v + 1, f"{v:.1f}", ha="center", fontsize=9, color=BLUE)
for xi, v in zip(x, auth):
    ax.text(xi + w / 2, v + 1, f"{v:.1f}", ha="center", fontsize=9, color=MUTED)
ax.set_xticks(x); ax.set_xticklabels([g[0] for g in groups], fontsize=10)
ax.set_ylabel("错误率 (%)")
ax.set_ylim(0, 80)
ax.set_title("kunz inner speech：本管线贪心 PER vs 作者在线 WER（同 eval trial）\n"
             "两指标口径不同（音素/词、无LM/带LM），但逐会话排序一致：50w < 125k，T15 125k Ⅱ 两边最差", fontsize=11.5)
ax.legend(fontsize=9, loc="upper left")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}\\summary\\kunz2025_greedy_per_vs_author_online_wer.png", dpi=150); plt.close(fig)

# ---------------- I: jude per-session (grouped bars) + breakdowns ----------------
sess = sorted(JU.session.unique())
fig = plt.figure(figsize=(12.6, 8.8))
gs = fig.add_gridspec(2, 2, height_ratios=[2.15, 1], hspace=0.5, wspace=0.22)
ax = fig.add_subplot(gs[0, :])

per = JU.groupby("session").apply(lambda g: agg(g), include_groups=False).reindex(sess)
vocab = JU.groupby("session").vocab.first().reindex(sess)
ours_v = per.to_numpy(float)
auth_v = np.array([JUDE_AUTH.get(s, np.nan) for s in sess])
x = np.arange(len(sess)); w = 0.38
colors_s = [BLUE if v == "50w" else PURPLE for v in vocab]
ax.bar(x - w / 2, ours_v, w, color=colors_s, alpha=0.92)
mask = ~np.isnan(auth_v)
ax.bar(x[mask] + w / 2, auth_v[mask], w, color="#98a3b4", alpha=0.92)
for xi, v in zip(x, ours_v):
    ax.text(xi - w / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=8.5, color=INK)
for xi, v in zip(x[mask], auth_v[mask]):
    ax.text(xi + w / 2, v + 1.5, f"{v:.1f}", ha="center", fontsize=8.5, color=MUTED)
for xi in np.where(~mask)[0]:
    ax.text(xi + w / 2, 2, "无在线\n输出", ha="center", fontsize=7.5, color=MUTED)
ax.axhline(agg(JU), ls="--", lw=1.3, color=BLUE, alpha=0.75)
ax.axhline(57.23, ls="--", lw=1.3, color="#98a3b4", alpha=0.9)
ax.set_xticks(x)
ax.set_xticklabels([f"{s.replace('t17.', '').replace('.', '-')}\n{v}"
                    for s, v in zip(sess, vocab)], fontsize=8.8)
ax.set_ylabel("错误率 (%)")
ax.set_ylim(0, 118)
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
ax.legend(handles=[
    Patch(color=BLUE, label="本管线贪心 PER · 50 词闭集"),
    Patch(color=PURPLE, label="本管线贪心 PER · 125k 开放词表"),
    Patch(color="#98a3b4", label="作者在线 WER（同块，RNN+5gram）"),
    Line2D([], [], ls="--", color=BLUE, label="本管线总体 39.97%（417 trial）"),
    Line2D([], [], ls="--", color="#98a3b4", label="作者在线总体 57.23%（256 trial）"),
], fontsize=8.8, loc="upper left", ncol=2, columnspacing=1.1, framealpha=0.95)
ax.set_title("jude 20260428（T17 闭锁综合征意图说话）：逐会话解码成绩\n"
             "val = 每个采集日最后一块；闭环块的作者在线输出与本文同 trial 可比", fontsize=11.5)
ax.grid(axis="y", alpha=0.3)

for k, (col, ttl, cs) in enumerate([("vocab", "按词表", [BLUE, PURPLE]),
                                    ("loop", "按开/闭环", [GREEN, AMBER])]):
    ax2 = fig.add_subplot(gs[1, k])
    g = JU.groupby(col).apply(lambda t: pd.Series(
        {"PER": agg(t), "n": len(t)}), include_groups=False)
    xx = np.arange(len(g))
    ax2.bar(xx, g.PER.values, 0.5, color=cs, alpha=0.92)
    for xi, v in zip(xx, g.PER.values):
        ax2.text(xi, v + 1.2, f"{v:.2f}%", ha="center", fontsize=10, color=INK)
    ax2.set_xticks(xx)
    ax2.set_xticklabels([f"{i}\n(n={int(n)})" for i, n in zip(g.index, g.n)], fontsize=10)
    ax2.set_ylim(0, 50)
    ax2.set_ylabel("贪心 PER (%)")
    ax2.set_title(ttl, fontsize=11)
    ax2.grid(axis="y", alpha=0.3)
fig.savefig(f"{OUT}\\session_level\\jude2026_per_session_greedy_per_vs_author_wer.png", dpi=150); plt.close(fig)

# ---------------- I2: jude per-session scatter (willett2023/card2024 style) ----------------
# single-color scatter, no connecting line, no color encoding (y-axis already
# shows PER); marker size = val trial count; aggregate + paper anchor lines
ju_day = (JU.groupby("session")
            .apply(lambda g: agg(g), include_groups=False).sort_index().astype(float))
ju_cnt = JU.groupby("session").size().reindex(ju_day.index)
ju_dates = pd.to_datetime([s.replace("t17.", "") for s in ju_day.index],
                          format="%Y.%m.%d")
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.scatter(ju_dates, ju_day.values, s=10 * ju_cnt.values, color=BLUE,
           edgecolors=INK, linewidths=0.4, zorder=3)
ax.axhline(agg(JU), ls="--", c=AMBER, lw=1.4, label=f"aggregate val PER {agg(JU):.2f}%")
ax.axhline(34.0, ls=":", c=RED, lw=1.4, label="paper offline best raw PER 34% (Jude 2026)")
for d, v, s in zip(ju_dates, ju_day.values, ju_day.index):
    if v == ju_day.max() or v == ju_day.min():
        ax.annotate(f"{s[-5:]}\n{v:.1f}%", (d, v), fontsize=9,
                    xytext=(6, 6), textcoords="offset points")
ax.set_ylabel("val (last-block) greedy PER (%)"); ax.set_xlabel("recording session")
ax.set_title("jude 20260428: per-session val greedy PER (T17)\n(marker size = number of val trials)")
for n_ref in (10, 25, 40):
    ax.scatter([], [], s=10 * n_ref, color=MUTED, edgecolors=INK, linewidths=0.4,
               label=f"{n_ref} trials")
ax.legend(fontsize=9, labelspacing=1.3, borderpad=0.9); ax.grid(alpha=0.3); rot_labels(ax)
fig.tight_layout(); fig.savefig(f"{OUT}\\session_level\\jude2026_per_session_greedy_per.png", dpi=150)
plt.close(fig)

# ---------------- J: training trajectories ----------------
def parse_log(path):
    """Extract validation PER curve points from a trainer log.

    Parameters
    ----------
    path : str
        Path to a ``training_log`` file containing lines of the form
        ``Val batch <n>: PER (avg): <x>``.

    Returns
    -------
    tuple of numpy.ndarray
        ``(batches, pers)``: validation batch indices and the
        corresponding PER values scaled to percent.
    """
    batches, pers = [], []
    for line in open(path, encoding="utf-8", errors="ignore"):
        m = re.search(r"Val batch (\d+): PER \(avg\): ([\d.]+)", line)
        if m:
            batches.append(int(m.group(1))); pers.append(100 * float(m.group(2)))
    return np.array(batches), np.array(pers)

def trajectory_figure(runs, xlim, title, out_png):
    """Plot validation PER training trajectories for one or more runs.

    Draws one curve per run, star-marks each run's best (minimum) PER,
    and saves the figure to ``out_png``.

    Parameters
    ----------
    runs : list of tuple
        ``(label, training_log_path, color)`` entries, one per run.
    xlim : float
        Right limit of the x axis (training batch).
    title : str
        Figure title (may contain Chinese text).
    out_png : str
        Output path for the saved PNG.
    """
    fig, ax = plt.subplots(figsize=(9, 5.4))
    for name, path, c in runs:
        b, p = parse_log(path)
        ax.plot(b, p, "-o", ms=3.5, lw=1.4, color=c,
                label=f"{name}  best {p.min():.2f}% @ {b[p.argmin()]:,} 批")
        ax.scatter([b[p.argmin()]], [p.min()], marker="*", s=210, color=c,
                   zorder=5, edgecolors=INK, linewidths=0.5)
    ax.set_xlim(-100, xlim)
    ax.set_ylim(0, 105)
    ax.set_xlabel("训练批数")
    ax.set_ylabel("val 贪心 PER (%)")
    ax.set_title(title, fontsize=11.5)
    ax.legend(fontsize=9.5, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(out_png, dpi=150); plt.close(fig)


trajectory_figure(
    [("T12 (400 trial)", r".\20250821_kunz\data\20250821_kunz\run_t12_mixed\training_log", "#D55E00"),
     ("T15 (615 trial)", r".\20250821_kunz\data\20250821_kunz\run_t15_mixed\training_log", "#009E73"),
     ("T16 (365 trial)", r".\20250821_kunz\data\20250821_kunz\run_t16_mixed\training_log", "#CC79A7")],
    4200,
    "kunz（mixed 臂）：小数据集在 750–1,500 批达最优后过拟合\n（星标 = best_checkpoint 所在批）",
    f"{OUT}\\training_trajectory\\kunz2025_training_trajectories.png")

trajectory_figure(
    [("T17 (3,107 trial)", r".\20260428_jude\data\20260428_jude\run_25k\training_log", BLUE)],
    14000,
    "jude T17 意图说话：数据量足够，val 贪心 PER 平稳下降\n（星标 = best_checkpoint 所在批）",
    f"{OUT}\\training_trajectory\\jude2026_training_trajectory.png")

import os
for p in [r"summary\willett2023_card2024_kunz2025_jude2026_greedy_per_spectrum",
          r"summary\kunz2025_greedy_per_vs_author_online_wer",
          r"session_level\jude2026_per_session_greedy_per_vs_author_wer",
          r"session_level\jude2026_per_session_greedy_per",
          r"training_trajectory\kunz2025_training_trajectories",
          r"training_trajectory\jude2026_training_trajectory"]:
    full = os.path.join(OUT, p + ".png")
    print(f"{full}  exists={os.path.exists(full)}")
