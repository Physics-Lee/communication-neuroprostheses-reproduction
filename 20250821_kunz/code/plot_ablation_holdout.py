# -*- coding: utf-8 -*-
"""Figures for the review follow-up tasks (A.4) of the kunz/jude note:

  summary/kunz2025_ablation_arms
      per participant: mixed vs imagined-only vs attempted-only training arms,
      all evaluated on the SAME closed-loop imagined *_eval blocks
      (bootstrap 95% CI), plus per-arm training trajectories
  (regenerates) summary/willett2023_card2024_kunz2025_jude2026_greedy_per_spectrum
      gains a T16 overt-speech row from the 3-fold block-CV holdout baseline

Same conventions as plot_kunz_jude.py (paper-prefixed names under figs/).
"""
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK = "#172033"; BLUE = "#2457c5"; GREEN = "#009E73"; RED = "#a83232"
AMBER = "#E69F00"; MUTED = "#5c667a"; PURPLE = "#6d3fa4"
OUT = r".\20250821_kunz\result\figs"
KUNZ = r".\20250821_kunz\data\20250821_kunz"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

SUBJ = ["t12", "t15", "t16"]
SUBJ_LABEL = {"t12": "T12", "t15": "T15", "t16": "T16"}
ARM_TRIALS = {  # open-loop training trials per (participant, arm)
    ("t12", "mixed"): 400, ("t12", "imagined"): 200, ("t12", "attempted"): 200,
    ("t15", "mixed"): 615, ("t15", "imagined"): 415, ("t15", "attempted"): 200,
    ("t16", "mixed"): 365, ("t16", "imagined"): 245, ("t16", "attempted"): 120,
}


def agg(df):
    return 100 * df.edit_distance.sum() / df.true_len.sum()


def boot_ci(df, n_boot=2000, seed=0):
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


def arm_df(subj, arm):
    return pd.read_csv(rf"{KUNZ}\run_{subj}_{arm}\val_greedy_per_trial.csv")


# ---------------- ablation arms ----------------
fig = plt.figure(figsize=(12.6, 6.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1.55, 1], hspace=0.62, wspace=0.26)
ax = fig.add_subplot(gs[0, :])

arms = ["mixed", "imagined", "attempted"]
arm_label = {"mixed": "mixed\n(overt+inner)", "imagined": "imagined-only\n(inner)",
             "attempted": "attempted-only\n(overt)"}
# Okabe-Ito, matching the updated plot_kunz_jude.py palette
arm_color = {"mixed": "#E69F00", "imagined": "#D55E00", "attempted": "#009E73"}
x = np.arange(len(SUBJ))
w = 0.26
for j, arm in enumerate(arms):
    vals, errs = [], []
    for subj in SUBJ:
        df = arm_df(subj, arm)
        vals.append(agg(df)); errs.append(boot_ci(df, seed=j))
    vals = np.array(vals); errs = np.array(errs)
    ax.bar(x + (j - 1) * w, vals, w * 0.92, color=arm_color[arm], alpha=0.92,
           yerr=[vals - errs[:, 0], errs[:, 1] - vals],
           error_kw=dict(ecolor=INK, lw=1.0, capsize=3))
    for xi, v, ci, subj in zip(x, vals, errs, SUBJ):
        ax.text(xi + (j - 1) * w, v + 2.4, f"{v:.1f}", ha="center",
                fontsize=8.6, color=INK)
ax.set_xticks(x)
ax.set_xticklabels([f"{SUBJ_LABEL[s]}\nval = imagined 闭环 _eval 块" for s in SUBJ],
                   fontsize=10)
ax.set_ylabel("贪心 CTC PER (%)")
ax.set_ylim(0, 100)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=arm_color[a], label=f"{arm_label[a]}（训练 trial："
        + " / ".join(str(ARM_TRIALS[(s, a)]) for s in SUBJ) + "）") for a in arms],
    fontsize=9, loc="upper left", ncol=3, columnspacing=1.0, framealpha=0.95)
ax.set_title("kunz 消融臂：训练数据任务构成 vs inner speech 解码成绩（同一评估集）\n"
             "误差条 = trial 级 bootstrap 95% CI（2,000 次重采样）", fontsize=12)
ax.grid(axis="y", alpha=0.3)

for k, subj in enumerate(SUBJ):
    ax2 = fig.add_subplot(gs[1, k])
    for j, arm in enumerate(arms):
        path = rf"{KUNZ}\run_{subj}_{arm}\training_log"
        batches, pers = [], []
        for line in open(path, encoding="utf-8", errors="ignore"):
            m = re.search(r"Val batch (\d+): PER \(avg\): ([\d.]+)", line)
            if m:
                batches.append(int(m.group(1))); pers.append(100 * float(m.group(2)))
        b, p = np.array(batches), np.array(pers)
        ax2.plot(b, p, "-", lw=1.3, color=arm_color[arm],
                 label=f"{arm} best {p.min():.1f}% @ {b[p.argmin()]:,}")
    ax2.set_ylim(0, 105)
    ax2.set_title(f"{SUBJ_LABEL[subj]} 训练轨迹", fontsize=10)
    ax2.set_xlabel("训练批数", fontsize=9)
    ax2.set_ylabel("val PER (%)", fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.legend(fontsize=7.2, loc="upper right")
    ax2.grid(alpha=0.3)
fig.savefig(rf"{OUT}\summary\kunz2025_ablation_arms.png", dpi=150); plt.close(fig)

# ---------------- spectrum + T16 overt holdout row ----------------
# Regenerates the all-English vertical spectrum of plot_kunz_jude.py in the
# same style, adding the T16 overt row from the 3-fold block-CV holdout.
B24 = pd.read_csv(r".\20230831_willett\data\20230616_willett_BtoT24\run_60k\test_partition_greedy_per_trial.csv")
B25 = pd.read_csv(r".\20240815_card\result\val_greedy_per_trial.csv")
JU = pd.read_csv(r".\20260428_jude\data\20260428_jude\run_25k\val_greedy_per_trial.csv")
K = {s: {a: arm_df(s, a) for a in arms} for s in SUBJ}
HO = pd.concat([pd.read_csv(rf"{KUNZ}\run_t16_holdout_fold{f}\val_greedy_per_trial.csv")
                for f in (1, 2, 3)], ignore_index=True)

# paper-side PER anchors, only where the paper itself reports a PER value
# (same-metric comparison; rows without a paper PER get no overlay):
#   Willett T12: paper Table 1 raw RNN PER 19.70 (vocal, 125k; note 20260731)
#   Card T15   : authors' online greedy PER record 9.54 (fig card2024_btot25_lm_contribution)
#   Jude T17   : paper offline best raw PER 34.0 (note 20260901)
# Kunz 2025 rows are excluded: the paper reports WER only, no PER.
# Rows ordered by paper publication date (willett 2023 -> card 2024 ->
# kunz 2025 -> jude 2026).
ANCHORS = {
    "T12 · overt": 19.70,
    "T15 · overt (copy task, ckpt)": 9.54,
    "T17 · intended (locked-in)": 34.0,
}
rows = [
    ("T12 · overt", "125k vocab", B24, GREEN,
     "20230616_willett", "8,800"),
    ("T15 · overt (copy task, ckpt)", "125k + 50 vocab", B25, GREEN,
     "20241009_card", "9,000"),
    ("T16 · overt (3-fold CV holdout)", "50 vocab", HO, GREEN,
     "20250821_kunz", "80/fold"),
    ("T15 · inner", "125k + 50 vocab", K["t15"]["mixed"], AMBER,
     "20250821_kunz", "615"),
    ("T12 · inner", "50 vocab", K["t12"]["mixed"], AMBER,
     "20250821_kunz", "400"),
    ("T16 · inner", "125k + 50 vocab", K["t16"]["mixed"], AMBER,
     "20250821_kunz", "365"),
    ("T17 · intended (locked-in)", "125k + 50 vocab", JU, BLUE,
     "20260428_jude", "3,107"),
]
fig, ax = plt.subplots(figsize=(14.5, 6.9))
vals, errs, colors = [], [], []
for name, vocab, df, c, src, ntr in rows:
    vals.append(agg(df)); errs.append(boot_ci(df)); colors.append(c)
x = np.arange(len(rows))
vals = np.array(vals); errs = np.array(errs)
# paper anchor = solid fill; our reproduction = hollow outline, same x position
for xi, (name, vocab, df, c, src, ntr) in zip(x, rows):
    if name not in ANCHORS:
        continue
    av = ANCHORS[name]
    ax.bar(xi, av, color=c, alpha=0.55, width=0.6, zorder=2)
    ax.text(xi, av - 3.2, f"{av:.2f}", ha="center", fontsize=8.6,
            color="white", fontweight="bold", zorder=4)
ax.bar(x, vals, facecolor="none", edgecolor=colors, lw=1.8, width=0.6,
       zorder=3, yerr=[vals - errs[:, 0], errs[:, 1] - vals],
       error_kw=dict(ecolor=INK, lw=1.1, capsize=3.5))
for xi, v, ci in zip(x, vals, errs):
    ax.text(xi, ci[1] + 2.2, f"{v:.2f}\n[{ci[0]:.1f}, {ci[1]:.1f}]",
            ha="center", fontsize=9, color=INK, zorder=4)
ax.set_xticks(x)
ax.set_xticklabels([f"{name}\n{vocab}\n{src}\nval {len(df):,} / train {ntr}"
                    for name, vocab, df, c, src, ntr in rows], fontsize=8.2)
ax.set_ylabel("Greedy CTC phoneme error rate PER (%)")
ax.set_xlabel("participant / task")
ax.set_ylim(0, 80)
from matplotlib.patches import Patch as _Patch
# anchor patch style matches the drawn anchor bars (row-color, alpha 0.55)
ax.legend(handles=[_Patch(facecolor="none", edgecolor=INK, lw=1.5,
                          label="ours (hollow, greedy PER + bootstrap 95% CI)"),
                   _Patch(color=GREEN, alpha=0.55,
                          label="PER anchor, solid (paper-printed or authors' online record)"),
                   _Patch(color=GREEN, label="overt speech"),
                   _Patch(color=BLUE, label="intended speech"),
                   _Patch(color=AMBER, label="inner speech")],
          fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.16), ncol=5,
          columnspacing=0.9, handletextpad=0.5)
ax.set_title("Decoding difficulty spectrum under the same GRU+CTC+greedy pipeline (seven models)\n"
             "hollow = our reproduction, solid = paper PER anchor; "
             "overt speech < intended speech < inner speech\n"
             "x ordered by paper publication: Willett 2023 → Card 2024 → Kunz 2025 → Jude 2026",
             fontsize=12)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(rf"{OUT}\summary\willett2023_card2024_kunz2025_jude2026_greedy_per_spectrum.png",
            dpi=150); plt.close(fig)

import os
for p in [r"summary\kunz2025_ablation_arms",
          r"summary\willett2023_card2024_kunz2025_jude2026_greedy_per_spectrum"]:
    full = os.path.join(OUT, p + ".png")
    print(f"{full}  exists={os.path.exists(full)}")

# console summary for the note
print("\n=== ablation arms (greedy PER on imagined _eval) ===")
for subj in SUBJ:
    for arm in arms:
        df = arm_df(subj, arm)
        ci = boot_ci(df, seed=7)
        print(f"{subj:4s} {arm:10s} n={len(df):3d}  {agg(df):6.2f}%  "
              f"[{ci[0]:.1f}, {ci[1]:.1f}]  ed/len={df.edit_distance.sum()}/{df.true_len.sum()}")
print("\n=== T16 attempted holdout (3-fold CV pooled) ===")
print(f"n={len(HO)}  {agg(HO):6.2f}%  ed/len={HO.edit_distance.sum()}/{HO.true_len.sum()}")
for f in (1, 2, 3):
    d = pd.read_csv(rf"{KUNZ}\run_t16_holdout_fold{f}\val_greedy_per_trial.csv")
    print(f"fold{f}: n={len(d)}  {agg(d):6.2f}%  ed/len={d.edit_distance.sum()}/{d.true_len.sum()}")
