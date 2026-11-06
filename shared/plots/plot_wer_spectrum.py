# -*- coding: utf-8 -*-
"""WER reproduction spectrum for four papers, same visual conventions as the
greedy-PER spectrum (willett2023_card2024_kunz2025_jude2026_greedy_per_spectrum):
solid = paper anchor, hollow = ours, bootstrap 95% CI whiskers, mode colors.

figs/summary/willett2023_card2024_kunz2025_jude2026_wer_spectrum.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

INK = "#172033"; GREEN = "#009E73"; BLUE = "#2457c5"; AMBER = "#E69F00"
RED = "#a83232"; MUTED = "#5c667a"
OUT = r".\shared\result\figs"

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# ---- data (all from archived per-trial files; CIs = bootstrap 95%) ----
# paper anchors
anchor = [23.8, 2.5, 50.0, 40.0]  # kunz = mid of 26-54 (125k, T15/T16)
anchor_note = ["3g headline\n按天中位·在线", "final session\n5g+Transformer·在线",
               "day68 conversational\n5g WFST·在线", "50词 inner speech\n生产系统·在线均值\n(范围 14–33)"]
# ours 5-gram (primary) and 3-gram (reference)
ours5 = [25.70, 22.87, 50.58, 90.97]
ci5 = [[24.06, 27.35], [21.31, 24.52], [47.54, 53.52], [78.44, 106.88]]
ours3 = [31.76, 29.77, 58.13, 83.28]
ci3 = [[30.19, 33.37], [28.27, 31.37], [55.30, 60.90], [78.16, 92.87]]
ours5_note = ["天中位 23.58", "+OPT 22.87", "", ""]
colors = [GREEN, GREEN, BLUE, AMBER]
xticks = [
    "Willett · T12 overt\n125k vocab\n20230616_willett\n880 test trials",
    "Card · T15 overt\n125k vocab\n20241009_card\n1,426 val trials",
    "Jude · T17 intended\nconversational\n20260428_jude\n417 val trials",
    "Kunz · T12/T15/T16 inner\n50w+125k\n20250821_kunz\n290 eval trials",
]

fig, ax = plt.subplots(figsize=(15.5, 7.2))
x = np.arange(4)
bw = 0.26

for xi in range(4):
    c = colors[xi]
    # paper anchor: solid
    ax.bar(xi - bw / 2 - 0.03, anchor[xi], bw, color=c, alpha=0.55, zorder=2)
    lab = "40.0" if xi == 3 else f"{anchor[xi]:.1f}"
    ax.text(xi - bw / 2 - 0.03, anchor[xi] - 3.4, lab,
            ha="center", color="white", fontweight="bold", fontsize=9, zorder=4)
    # ours 5-gram: hollow thick + CI
    lo, hi = ci5[xi]
    ax.bar(xi + bw / 2 - 0.12, ours5[xi], bw, facecolor="none", edgecolor=INK,
           lw=1.8, zorder=3,
           yerr=[[ours5[xi] - lo], [hi - ours5[xi]]],
           error_kw=dict(ecolor=INK, lw=1.1, capsize=3.5))
    note = ours5_note[xi]
    txt = f"{ours5[xi]:.2f}" + (f"\n{note}" if note else "")
    ax.text(xi + bw / 2 - 0.12, hi + 2.0, txt, ha="center", fontsize=8.4,
            color=INK, zorder=4)
    # ours 3-gram: hollow dashed thinner
    lo3, hi3 = ci3[xi]
    ax.bar(xi + bw / 2 + bw / 2 + 0.06, ours3[xi], bw * 0.7, facecolor="none",
           edgecolor=MUTED, lw=1.2, ls=(0, (3, 2)), zorder=3,
           yerr=[[ours3[xi] - lo3], [hi3 - ours3[xi]]],
           error_kw=dict(ecolor=MUTED, lw=1.0, capsize=3))
    ax.text(xi + bw / 2 + bw / 2 + 0.06, hi3 + 2.0, f"{ours3[xi]:.2f}",
            ha="center", fontsize=8.4, color=MUTED, zorder=4)

# Kunz incomparable-protocol flag
ax.text(3, 120, "口径不可比：\n论文=50词受限·生产级在线系统\n我们=125k开放词表·离线小数据消融臂",
        ha="center", fontsize=8.5, color=RED, zorder=5)
ax.text(3 - bw / 2 - 0.03, anchor[3] + 1.2, "24/14/33\n均值 23.7",
        ha="center", fontsize=7.6, color=INK, zorder=5)

ax.set_xticks(x)
ax.set_xticklabels(xticks, fontsize=8.6)
ax.set_ylabel("Word error rate (%)")
ax.set_ylim(0, 135)
ax.set_title(
    "WER 复现谱图（四篇论文，同一 GRU+CTC 声学模型 + 各论文同源 5-gram WFST 解码）\n"
    "实心 = 论文锚点，粗空心 = 我们 5-gram TLG Viterbi，细虚线空心 = 我们 3-gram 对照；"
    "误差条 = bootstrap 95% CI\n"
    "Willett 按天中位（论文口径）23.58 vs 23.8；Card 论文含逐 session 在线适配与 "
    "Transformer 重打分（我们 5g+OPT 22.87）；Kunz 口径不可比如实记录",
    fontsize=11)
ax.grid(axis="y", alpha=0.3)
ax.legend(handles=[
    Patch(facecolor="none", edgecolor=INK, lw=1.8, label="ours 5-gram TLG (hollow) + bootstrap 95% CI"),
    Patch(facecolor="none", edgecolor=MUTED, lw=1.2, ls=(0, (3, 2)), label="ours 3-gram (reference)"),
    Patch(color=GREEN, alpha=0.55, label="paper anchor (solid) · overt"),
    Patch(color=BLUE, alpha=0.55, label="paper anchor · intended"),
    Patch(color=AMBER, alpha=0.55, label="paper anchor · inner"),
], fontsize=9, loc="upper left", bbox_to_anchor=(0, -0.17), ncol=5,
   columnspacing=0.9, handletextpad=0.5)
fig.tight_layout()
fig.savefig(OUT + r"\summary\willett2023_card2024_kunz2025_jude2026_wer_spectrum.png",
            dpi=150)
plt.close(fig)
print("saved", OUT + r"\summary\willett2023_card2024_kunz2025_jude2026_wer_spectrum.png")
