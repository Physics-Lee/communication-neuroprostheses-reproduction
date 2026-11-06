# -*- coding: utf-8 -*-
"""Card 2024 WER scope ("口径") comparison: ours vs paper anchors.

Provenance:
  ours  = fixed-protocol OPT-6.7B rescoring (alpha=0.55), official decode
          params (bp90/acs0.325/NB100), OFFLINE mixed evaluation over the
          full 1426-trial val set (docs/20261007_tlg_vacation_results.md §4b)
  paper = online system with DAILY recalibration (Card 2024 NEJM):
          session-2 9.8 (CI 4.1-16.0); final five 2.5 (CI 2.0-3.1)
Plot: D:/apps/psychopy/python.exe code/plot_card24_scopes.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

OUT = (r".\20240815_card\result\figs"
       r"\summary\card24_scope_ours_vs_paper.png")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
fig.suptitle("Card 2024 (T15) WER：我们 vs 论文——口径拆解", fontsize=14, y=1.02)

# ---- Panel 1: session-2 anchor scope ----
ax = axes[0]
labels = ["我们\n(离线混评 + OPT)", "论文\n(在线逐日校准)"]
vals = [3.25, 9.8]
bars = ax.bar(labels, vals, color=["#2c7fb8", "#d95f0e"], width=0.55)
ax.errorbar(1, 9.8, yerr=[[9.8 - 4.1], [16.0 - 9.8]], fmt="none",
            ecolor="black", capsize=5, lw=1.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.2f}",
            ha="center", fontsize=11, fontweight="bold")
ax.axhline(4.1, ls=":", color="gray", lw=1)
ax.text(-0.38, 4.35, "CI 下缘 4.1", fontsize=8, color="gray")
ax.set_ylabel("WER (%)")
ax.set_ylim(0, 18)
ax.set_title("① session-2 口径（论文锚点）\n我们 3.25 已落入并低于论文 CI", fontsize=10)
ax.grid(alpha=0.3, axis="y")

# ---- Panel 2: final-five scope (daily adaptation gap) ----
ax = axes[1]
labels = ["我们末5 session\n(2025 追访, 离线无校准)", "论文 final five\n(在线+当日适配)"]
vals = [62.01, 2.5]
bars = ax.bar(labels, vals, color=["#2c7fb8", "#d95f0e"], width=0.55)
ax.errorbar(1, 2.5, yerr=[[2.5 - 2.0], [3.1 - 2.5]], fmt="none",
            ecolor="black", capsize=5, lw=1.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}",
            ha="center", fontsize=11, fontweight="bold")
ax.annotate("当日适配缺口\n(两种简单校准配方\n均已失败, 见报告)",
            xy=(0.5, 32), ha="center", fontsize=9, color="#7a0177")
ax.set_ylabel("WER (%)")
ax.set_ylim(0, 70)
ax.set_title("② final-five 口径：当日适配缺口", fontsize=10)
ax.grid(alpha=0.3, axis="y")

# ---- Panel 3: our scope ladder (all offline) ----
ax = axes[2]
scopes = ["session-2", "2023 年 26 session 池化", "全 41 session 池化\n(四联图面板B的数字)",
          "2024+ 晚期", "末 5 session\n(全为 2025)"]
wers = [3.25, 10.71, 21.57, 40.02, 62.01]
colors = ["#c7e9ff", "#7fcde5", "#41b6c4", "#1d91c0", "#0c2c84"]
y = range(len(scopes))[::-1]
ax.barh(list(y), wers, color=colors)
for yi, v in zip(y, wers):
    ax.text(v + 1, yi, f"{v:.2f}", va="center", fontsize=10)
ax.set_yticks(list(y))
ax.set_yticklabels(scopes, fontsize=9)
ax.set_xlabel("WER (%)")
ax.set_xlim(0, 72)
ax.set_title("③ 我们的口径阶梯（同一套离线输出，\n仅划分方式不同）", fontsize=10)
ax.grid(alpha=0.3, axis="x")

fig.text(0.5, -0.04,
         "我们 = 修复版 OPT-6.7B 重打分（α=0.55）· 官方解码参数 · 离线混合评估（1426 句）"
         "｜论文 = 在线系统 + 每日重校准",
         ha="center", fontsize=9, color="dimgray")
plt.tight_layout()
plt.savefig(OUT, dpi=150, bbox_inches="tight")
print("saved", OUT)
