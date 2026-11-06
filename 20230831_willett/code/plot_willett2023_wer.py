# -*- coding: utf-8 -*-
"""Willett 2023 (T12, 125k vocab) WER reproduction figure.

summary/willett2023_wer_3gram_lm
    PER vs WER, paper anchor (solid) vs our reproduction (hollow, bootstrap CI)

Inputs (all produced by earlier steps, nothing recomputed here):
  data/20230616_willett_BtoT24/run_60k/test_partition_greedy_per_trial.csv
      greedy PER per trial (eval_btot24.py, run_60k checkpoint)
  data/20230616_willett_BtoT24/run_60k/word_decode_joint3gram.csv
      per-trial word edit distances (decode_words_btot24_v2.py:
      joint lexicon + 3-gram KenLM CTC beam search, alpha=2.0 beta=0.15)

Anchors (paper-printed):
  PER 19.70  (Table 1 raw RNN PER, vocal, 125k; same value as the spectrum fig)
  WER 23.80  (paper headline online median WER, 125k vocab; dataset README)

Run: "D:\apps\psychopy\python.exe" plot_willett2023_wer.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK = "#172033"; GREEN = "#009E73"; MUTED = "#5c667a"
OUT = r".\20230831_willett\result\figs"
RUN = (r".\20230831_willett"
       r"\data\20230616_willett_BtoT24\run_60k")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

per_df = pd.read_csv(RUN + r"\test_partition_greedy_per_trial.csv")
wer_df = pd.read_csv(RUN + r"\word_decode_joint3gram.csv").rename(
    columns={"ref_len": "true_len"})


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


# per-day median WER (protocol closest to the paper's day-median headline)
day = wer_df.groupby("session").apply(
    lambda g: g.edit_distance.sum() / g.true_len.sum(),
    include_groups=False)
wer_day_median = 100 * day.median()

anchors = {"PER": 19.70, "WER": 23.80}
ours = {"PER": agg(per_df), "WER": agg(wer_df)}
ci = {"PER": boot_ci(per_df), "WER": boot_ci(wer_df)}

fig, ax = plt.subplots(figsize=(9.2, 6.2))
x = np.arange(2)
w = 0.32
# paper anchors: solid; ours: hollow with bootstrap CI (spectrum-fig style)
ax.bar(x - w / 2, [anchors["PER"], anchors["WER"]], w, color=GREEN,
       alpha=0.55, zorder=2)
ax.bar(x + w / 2, [ours["PER"], ours["WER"]], w, facecolor="none",
       edgecolor=GREEN, lw=1.8, zorder=3,
       yerr=[[ours[k] - ci[k][0] for k in ("PER", "WER")],
             [ci[k][1] - ours[k] for k in ("PER", "WER")]],
       error_kw=dict(ecolor=INK, lw=1.1, capsize=3.5))
for xi, k in zip(x, ("PER", "WER")):
    av, ov = anchors[k], ours[k]
    ax.text(xi - w / 2, av - 2.2, f"{av:.2f}", ha="center", fontsize=9.5,
            color="white", fontweight="bold", zorder=4)
    ax.text(xi + w / 2, ci[k][1] + 1.4, f"{ov:.2f}\n[{ci[k][0]:.1f}, "
            f"{ci[k][1]:.1f}]", ha="center", fontsize=9, color=INK, zorder=4)
ax.axhline(23.80, color=MUTED, lw=0.8, ls=":", zorder=1)
ax.set_xticks(x)
ax.set_xticklabels(
    ["greedy PER\n(phonemes, no LM)\n880 test sentences",
     "WER\n(joint lexicon + 3-gram KenLM beam, α=2.0 β=0.15)\n880 test sentences"],
    fontsize=9.5)
ax.set_ylabel("error rate (%)")
ax.set_ylim(0, 44)
ax.set_title("Willett 2023 (T12, 125k vocab): PER → WER reproduction\n"
             "hollow = ours (60k-step rerun; greedy PER 23.42 vs paper 19.70), "
             "solid = paper anchor\npaper WER 23.8 = online day-median headline; "
             f"ours: aggregate {ours['WER']:.2f}, per-day median {wer_day_median:.2f}",
             fontsize=11)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="none", edgecolor=INK, lw=1.5,
                         label="ours (hollow, bootstrap 95% CI)"),
                   Patch(color=GREEN, alpha=0.55,
                         label="paper anchor (solid)")],
          fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.14), ncol=2)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(rf"{OUT}\summary\willett2023_wer_3gram_lm.png", dpi=150)
plt.close(fig)
print("saved", rf"{OUT}\summary\willett2023_wer_3gram_lm.png")
print(f"PER ours {ours['PER']:.2f} anchor {anchors['PER']}")
print(f"WER ours {ours['WER']:.2f} (day-median {wer_day_median:.2f}) "
      f"anchor {anchors['WER']}")
