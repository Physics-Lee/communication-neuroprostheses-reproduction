# -*- coding: utf-8 -*-
"""Visualize the two experiments from 2026-09-29/30:

  fig1  official-decoder port: (a) 4-paper WER old vs official vs +OPT,
        (b) Card per-session trajectory, (c) Kunz arms, (d) pool-oracle
        root-cause evidence.
  fig2  rescoring-model swap: (a) 5g / +OPT / +Qwen across all datasets,
        (b) rank-probe diagnosis.

Output: result/figs/summary/exp1_official_decoder.png
        result/figs/summary/exp2_rescorer_swap.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

C_OLD = "#9aa0a6"   # old decoder - gray
C_5G = "#8ab4d8"    # official 5g - light blue
C_OPT = "#1a73e8"   # +OPT - blue
C_QWEN = "#d93025"  # +Qwen instruct - red
C_QWBASE = "#f29900"  # +Qwen base - orange
C_ANCHOR = "#202124"

OUT = r".\shared\result\figs\summary"

# ---------------------------------------------------------------- fig 1
fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))

# (a) Willett / Card / Jude — pooled bars + paper-口径-matched green bar
ax = axes[0][0]
groups = ["Willett 2023\n(T12 · 880句)", "Card 2024\n(T15 · 1426句)", "Jude 2026\n(T17 · 417句)"]
old_v = [25.70, 23.67, 50.58]
g5_v = [24.85, 23.14, 47.39]
opt_v = [23.22, 22.16, 46.01]
# our number computed in the SAME way the paper reports its anchor
matched_v = [21.60, 5.05, 46.01]
matched_lab = ["21.6\n按天中位", "5.05\n第2天", "46.0\n对话语料"]
anchors = [23.8, 9.8, 50.0]
anchor_txt = ["论文 23.8\n按天中位", "论文 9.8\n第2天在线", "论文 ~50\n对话语料"]
anchor_above = [False, False, True]
C_MATCH = "#188038"
x = np.arange(3)
w = 0.2
for i, (v, c, lab) in enumerate([(old_v, C_OLD, "旧解码器（汇总）"),
                                 (g5_v, C_5G, "官方管线 5-gram（汇总）"),
                                 (opt_v, C_OPT, "官方管线 + OPT（汇总）"),
                                 (matched_v, C_MATCH, "+OPT · 论文同口径")]):
    bars = ax.bar(x + (i - 1.5) * w, v, w, color=c, label=lab)
    for b, vv in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, vv + 0.6, f"{vv:.1f}",
                ha="center", va="bottom", fontsize=7.8)
for i, (a, t, above) in enumerate(zip(anchors, anchor_txt, anchor_above)):
    ax.plot([x[i] - 2 * w, x[i] + 2 * w], [a, a], color=C_ANCHOR, lw=1.8)
    ax.text(x[i], a + 1.2 if above else a - 1.8, t, fontsize=8,
            va="bottom" if above else "top", ha="center", color=C_ANCHOR)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=9.5)
ax.set_ylabel("WER (%)")
ax.set_ylim(0, 62)
ax.set_title("(a) 词错误率：旧解码器 → 官方解码器移植\n（蓝柱＝全部句子汇总；绿柱＝按论文同口径计算）",
             fontsize=10.5)
ax.legend(fontsize=8.5, frameon=False, loc="upper left", ncol=2)
ax.spines[["top", "right"]].set_visible(False)
ax.text(0, -0.16, "Card 的汇总柱被 2024–25 年信号漂移 session 拖高（论文未评测该时段）；论文同口径对比看绿柱与黑线",
        transform=ax.transAxes, fontsize=7.8, color="#5f6368")

# (b) Card per-session trajectory (+OPT), x = days since implantation.
# Paper Fig 2: session 1 = day 25, last plotted eval session = #80 (day 244).
# Our dataset's first day (2023-08-13) = day 25 -> implant = 2023-07-19.
ax = axes[0][1]
from datetime import date as _date
mmdd = ["08-13","08-18","08-20","08-25","08-27","09-01","09-03","09-24","09-29",
        "10-01","10-06","10-08","10-13","10-15","10-20","10-22","11-03","11-04",
        "11-17","11-19","11-26","12-03","12-08","12-10","12-17","12-29",
        "02-25","03-08","03-15","03-17","05-10","06-14","07-19","07-21","07-28",
        "01-10","01-12","03-14","03-16","03-30","04-13"]
wer_sess = [18.70,5.05,4.79,4.40,15.94,5.69,9.29,19.90,7.45,6.39,5.41,18.00,27.27,
            6.85,23.94,12.21,24.56,0.00,3.57,1.54,34.53,5.19,4.80,0.60,15.88,8.71,
            13.41,28.57,27.07,7.14,16.67,29.80,68.00,27.83,14.43,
            91.33,76.51,88.27,49.42,62.25,25.79]
yr = [2023]*26 + [2024]*9 + [2025]*6
IMPLANT = _date(2023, 7, 19)
days = np.array([(_date(y, int(s[:2]), int(s[3:])) - IMPLANT).days
                 for y, s in zip(yr, mmdd)])
yra = np.array(yr)
wera = np.array(wer_sess)
colors = {2023: "#1a73e8", 2024: "#f29900", 2025: "#d93025"}
for y in (2023, 2024, 2025):
    m = yra == y
    ax.plot(days[m], wera[m], "o-", ms=4, color=colors[y], label=f"{y} 年")
# paper's official trajectory: 15 evaluation sessions digitized from Fig 2
# (eyeball reads of the published figure; Table S3 gives final-five dailies)
paper_pts = [(27, 9.8), (34, 13.3), (46, 4.5), (48, 5.0), (69, 5.7), (74, 3.3),
             (76, 4.5), (81, 3.7), (83, 4.2), (88, 2.5), (90, 3.0), (95, 0.99),
             (223, 2.7), (244, 1.8)]
px_, py_ = zip(*paper_pts)
ax.plot(px_, py_, "o-", color="#0b8043", lw=2.2, ms=4, zorder=4,
        label="论文在线系统（Fig2 逐点读数）")
ax.fill_between([223, 244], 1.0, 3.3, color="#0b8043", alpha=0.15, zorder=1,
                label="论文日均值 1.0-3.3%（S3）")
ax.axvline(244, color="#0b8043", lw=1.0, ls="--")
ax.axvspan(244, 645, color="#000000", alpha=0.05, zorder=0)
ax.text(300, 95, "论文最后一个评测 session：第80号，植入后244天\n≈ 2024-03-19；右侧灰区＝论文未评测时段",
        fontsize=8.5, color="#0b8043", va="top")
ax.annotate("session 2\n我们 5.05%", xy=(30, 5.05), xytext=(60, 42),
            fontsize=9, arrowprops=dict(arrowstyle="->", color="#5f6368"))
ax.text(145, 55, "2023：中位 7.45%", fontsize=9, ha="center", color="#1a73e8")
ax.text(465, 8, "2024 下半年起：信号漂移\n（缺当日校准，WER 远高于 PER）", fontsize=8.5,
        ha="center", va="bottom", color="#d93025")
ax.set_xticks([25, 100, 200, 244, 366, 540, 634])
ax.set_xticklabels(["25\n23-08-13", "100", "200", "244\n24-03-19", "366",
                    "540", "634\n25-04-13"], fontsize=8)
ax.set_xlabel("植入后天数")
ax.set_ylabel("WER (%)")
ax.set_xlim(10, 645)
ax.set_ylim(-2, 100)
ax.set_title("(b) Card 逐 session WER vs 论文官方轨迹（官方管线 + OPT）", fontsize=11)
ax.legend(fontsize=8.5, frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

# (c) Kunz arms
ax = axes[1][0]
groups = ["T12", "T15", "T16"]
g5_v = [100.49, 79.55, 85.61]
opt_v = [97.30, 78.81, 83.13]
x = np.arange(3)
for i, (v, c, lab) in enumerate([(g5_v, C_5G, "官方管线 5-gram（bp=1）"),
                                 (opt_v, C_OPT, "+ OPT-6.7B")]):
    bars = ax.bar(x + (i - 0.5) * 0.32, v, 0.32, color=c, label=lab)
    for b, vv in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, vv + 1, f"{vv:.1f}",
                ha="center", va="bottom", fontsize=8.5)
ax.axhspan(26, 54, color="#fce8e6", zorder=0)
ax.text(2.35, 40, "论文在线\n26–54%", fontsize=8.5, color="#d93025", va="center")
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_xlim(-0.55, 2.8)
ax.set_ylim(0, 122)
ax.set_ylabel("WER (%)")
ax.set_title("(c) Kunz 2025 三臂（消融小模型，声学侧差距 ~30 点为主因）", fontsize=11)
ax.legend(fontsize=9, frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)

# (d) pool oracle
ax = axes[1][1]
labels = ["subset-100", "Card 全量\n(1426句)"]
old_o = [8.65, 16.66]
new_o = [3.92, 12.93]
x = np.arange(2)
for i, (v, c, lab) in enumerate([(old_o, C_OLD, "旧解码器（token 重组丢候选）"),
                                 (new_o, C_OPT, "官方 lattice 解码器")]):
    bars = ax.bar(x + (i - 0.5) * 0.32, v, 0.32, color=c, label=lab)
    for b, vv in zip(bars, v):
        ax.text(b.get_x() + b.get_width() / 2, vv + 0.25, f"{vv:.2f}",
                ha="center", va="bottom", fontsize=9.5)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("候选池 oracle WER (%)")
ax.set_title("(d) 根因证据：候选池 oracle（即使选对也降不到的下限）", fontsize=11)
ax.legend(fontsize=9, frameon=False)
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("实验一：官方解码栈逐行移植（wenet_lattice.cc，官方参数零调参）", fontsize=13, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT + r"\exp1_official_decoder.png", dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- fig 2
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))

# (a) 5g / +OPT / +Qwen / +Qwen-base
ax = axes[0]
groups = ["Willett", "Card", "Jude", "Kunz\nT12", "Kunz\nT15", "Kunz\nT16"]
g5_v = [24.85, 23.14, 47.39, 100.49, 79.55, 85.61]
opt_v = [23.22, 22.16, 46.01, 97.30, 78.81, 83.13]
qw_v = [35.55, 31.20, 55.07, 98.28, 83.09, 91.56]
qb_v = [35.93, 31.25, 55.33, np.nan, np.nan, np.nan]  # base 复核仅跑前三篇
x = np.arange(6)
for si, (v, c, lab) in enumerate([(g5_v, C_5G, "5-gram 基线"),
                                  (opt_v, C_OPT, "+ OPT-6.7B（base 模型）"),
                                  (qw_v, C_QWEN, "+ Qwen3.5-4B（指令微调）"),
                                  (qb_v, C_QWBASE, "+ Qwen3.5-4B-Base（base 复核）")]):
    bars = ax.bar(x + (si - 1.5) * 0.2, v, 0.2, color=c, label=lab)
    for gi, (b, vv) in enumerate(zip(bars, v)):
        if np.isnan(vv):
            continue
        # two-tier stagger by (group, series) parity so adjacent bar labels
        # within a group never sit at the same height
        ax.text(b.get_x() + b.get_width() / 2, vv + (1 if (gi + si) % 2 == 0 else 5.5),
                f"{vv:.1f}", ha="center", va="bottom", fontsize=7.5)
ax.set_xticks(x)
ax.set_xticklabels(groups, fontsize=10)
ax.set_ylabel("WER (%)")
ax.set_ylim(0, 130)
ax.set_title("(a) 重打分模型替换：Qwen3.5 两版本均全面劣化（α 扫描后最优）", fontsize=11)
ax.legend(fontsize=8.5, frameon=False, loc="upper left")
ax.spines[["top", "right"]].set_visible(False)
ax.text(0.70, 0.97, "Base 复核仅跑前三篇（35.93/31.25/55.33）\n与 instruct 版差 ≤0.4 点；Kunz 未跑",
        transform=ax.transAxes, fontsize=8, ha="center", va="top", color="#5f6368")

# (b) rank probe: matched 30-trial sample, search vs instruct vs base
ax = axes[1]
labels = ["搜索分数\n(0.325·声学 + 5-gram)", "Qwen3.5-4B\n（指令微调）",
          "Qwen3.5-4B-Base"]
vals = [12.0, 49.6, 49.0]
bars = ax.bar([0, 1, 2], vals, 0.45, color=[C_OPT, C_QWEN, C_QWBASE])
for b, vv in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, vv + 1, f"平均第 {vv:.0f} 名",
            ha="center", va="bottom", fontsize=11)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("最接近参考句的候选在池内排名（池约 100 句，越小越好）")
ax.set_ylim(0, 85)
ax.set_title("(b) 根因：两版 Qwen 给正确句的排名同样低（Jude 30 trial 同样本）", fontsize=11)
ax.text(0.98, 0.97, "两版几乎重合：49.6 vs 49.0，argmax 命中均 1/30\n"
                    "池内有逐字正确答案（ED=0）时：搜索排第 1，\n"
                    "instruct 排第 53–61 名、base 排第 70–88 名\n"
                    "→ 与指令微调无关：Qwen3.5 家族裸文本先验 ≠ 句子似然",
        transform=ax.transAxes, fontsize=8.5, ha="right", va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="#f8f9fa", ec="#dadce0"))
ax.spines[["top", "right"]].set_visible(False)

fig.suptitle("实验二：重打分模型 OPT-6.7B → Qwen3.5-4B / 4B-Base（官方公式不变）", fontsize=13, y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(OUT + r"\exp2_rescorer_swap.png", dpi=200)
plt.close(fig)
print("saved both figures to", OUT)
