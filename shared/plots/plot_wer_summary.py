# -*- coding: utf-8 -*-
"""Four-paper WER reproduction summary (Willett/Card/Jude/Kunz).

figs/summary/wer_reproduction_4papers.png
  panel A: raw greedy PER, ours (hollow) vs paper anchor (solid)
  panel B: WER, paper anchor (solid) vs ours 5-gram TLG (hollow) and
           ours 3-gram (thin dotted-edge bar)

All numbers from the archived per-trial files (word_decode_tlg5gram.tsv /
word_decode_3gram.csv / *_greedy_per_trial.csv) and the paper texts in
result/papers_txt/. Kunz caveats: paper anchor is the 50-word inner-speech
online WER of their production system (24/14/33 for T12/T15/T16), which our
small-data ablation arms cannot match in scope; plotted with hatch.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

INK = "#172033"; GREEN = "#009E73"; MUTED = "#5c667a"; RED = "#a83232"
OUT = r".\shared\result\figs"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

papers = ["Willett 2023\nT12 · 125k", "Card 2024\nT15 · 125k",
          "Jude 2026\nT17 · conversational", "Kunz 2025\nT12/T15/T16 · inner"]
per_paper = [19.70, 9.54, 34.0, None]          # paper PER anchor (None = not reported)
per_ours = [23.42, 10.19, 39.97, 61.49]        # kunz = mean of 3 arms' val PER
wer_paper = [23.8, 9.8, 50.0, None]            # kunz: 24/14/33 (50w) plotted as note
wer_ours5 = [23.58, 23.67, 50.58, None]        # willett = day-median; kunz 106.9/78.4/87.6
wer_ours3 = [28.68, 29.77, 58.13, None]        # willett/card/jude aggregate; kunz 92.9/78.8/78.2

fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.4))
x = np.arange(4)
w = 0.3

# ---- panel A: PER ----
ax = axes[0]
ax.bar(x[:3], [p for p in per_paper if p is not None], w * 1.6, color=GREEN,
       alpha=0.55, zorder=2)
ax.bar(x[:3], per_ours[:3], w * 1.6, facecolor="none", edgecolor=GREEN,
       lw=1.8, zorder=3)
for xi, (a, o) in enumerate(zip(per_paper[:3], per_ours[:3])):
    ax.text(xi, a - 1.6, f"{a:.2f}", ha="center", color="white",
            fontweight="bold", fontsize=9, zorder=4)
    ax.text(xi, o + 1.2, f"{o:.2f}", ha="center", fontsize=9, color=INK, zorder=4)
ax.text(3, 30, "论文未报告\nraw PER\n(仅 WER)", ha="center", fontsize=9,
        color=MUTED, zorder=4)
ax.text(3, 61.49 + 1.5, "61.49*\n(3臂均值)", ha="center", fontsize=8.5, color=INK)
ax.set_xticks(x)
ax.set_xticklabels(papers, fontsize=8.6)
ax.set_ylabel("greedy PER (%)")
ax.set_ylim(0, 70)
ax.set_title("raw PER：声学模型侧", fontsize=11)

# ---- panel B: WER ----
ax = axes[1]
pp = [v for v in wer_paper if v is not None]
ax.bar(x[:3], pp, w * 1.6, color=GREEN, alpha=0.55, zorder=2)
o5 = [wer_ours5[0], 22.87, wer_ours5[2]]  # card = 5g+OPT rescoring
o3 = [wer_ours3[0], wer_ours3[1], wer_ours3[2]]
ax.bar(x[:3] - w / 2, o5, w * 0.8, facecolor="none", edgecolor=INK, lw=1.8,
       zorder=3, label="ours 5-gram TLG")
ax.bar(x[:3] + w / 2, o3, w * 0.8, facecolor="none", edgecolor=INK, lw=1.0,
       ls=(0, (2, 2)), zorder=3, label="ours 3-gram")
for xi, (a, o) in enumerate(zip(pp, o5)):
    ax.text(xi - w / 2, a - 2.6, f"{a:.1f}", ha="center", color="white",
            fontweight="bold", fontsize=9, zorder=4)
    extra = " (+OPT 22.87)" if xi == 1 else ""
    ax.text(xi - w / 2, o + 1.2, f"{o:.2f}{extra}", ha="center", fontsize=8.2, color=INK)
    ax.text(xi + w / 2, o3[xi] + 1.2, f"{o3[xi]:.1f}", ha="center", fontsize=8.6,
            color=MUTED)
# Kunz: incomparable protocol, hatched ours-only bars
kw = [106.88, 78.44, 87.59]
ax.bar([3 - w / 2, 3 + w / 2], [106.88, 78.44], [w * 0.7, w * 0.7],
       facecolor="none", edgecolor=RED, lw=1.2, hatch="//", zorder=3)
ax.text(3 - w / 2, 106.88 + 1.5, "106.9", ha="center", fontsize=8.5, color=RED)
ax.text(3 + w / 2, 78.44 + 1.5, "78.4", ha="center", fontsize=8.5, color=RED)
ax.text(3, 115, "论文锚点 24/14/33 (50词)\n为生产级系统口径，不可比",
        ha="center", fontsize=8, color=RED)
ax.set_xticks(x)
ax.set_xticklabels(papers, fontsize=8.6)
ax.set_ylabel("WER (%)")
ax.set_ylim(0, 130)
ax.set_title("WER：语言模型解码侧（5-gram 为主）", fontsize=11)

fig.suptitle("四篇论文 WER 复现汇总（Willett/Card/Jude/Kunz）——实心=论文锚点，"
             "空心=我们；Willett WER 用按天中位数（论文口径），其余为聚合", fontsize=12)
fig.legend(handles=[Patch(facecolor="none", edgecolor=INK, lw=1.5,
                          label="ours (hollow)"),
                    Patch(color=GREEN, alpha=0.55, label="paper anchor (solid)")],
           fontsize=9, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2)
for ax in axes:
    ax.grid(axis="y", alpha=0.3)
fig.tight_layout(rect=(0, 0.05, 1, 1))
fig.savefig(OUT + r"\summary\wer_reproduction_4papers.png", dpi=150)
plt.close(fig)
print("saved", OUT + r"\summary\wer_reproduction_4papers.png")
