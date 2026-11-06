# -*- coding: utf-8 -*-
"""Rapid-calibration curve regenerated (2026-10-13 P3b).

The original one-off plotting script was lost in the 2026-10-08 split; this
rebuild reads the same numbers straight from the ft_* checkpoints (val_PER
saved at training time) + the zero-shot summaries + the paper's rapid-
calibration anchor. Negative result reproduced as-is: simple per-day
fine-tuning does NOT recover the paper's rapid calibration gains.
Run: <venv312 python> code/plot_rapid_calibration_curve.py
     (needs torch to read checkpoints; then psychopy for rendering is fine
      too if torch unavailable there)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

FT = (r".\20240815_card\data"
      r"\20241009_card_BtoT25")
OUT = (r".\20240815_card\result\figs"
       r"\summary\card2024_rapid_calibration_curve.png")

TARGET = "t15.2025.03.30"

# zero-shot (N=0) from zeroshot_summary.json; N>0 from best checkpoints
zs = json.load(open(os.path.join(FT, "ft_t1520250330_N0",
                                 "zeroshot_summary.json"), encoding="utf-8"))
points = [(0, 100 * zs["aggregate_PER"])]
for n in (10, 20, 50, 165):
    ck = torch.load(os.path.join(FT, f"ft_t1520250330_N{n}_run",
                                 "checkpoint", "best_checkpoint"),
                    map_location="cpu", weights_only=False)
    points.append((n, 100 * ck["val_PER"]))
points.sort()

# paper rapid-calibration anchor: ~2.5% WER after a handful of sentences on
# the online system (daily recalibration); our offline curve stays far above.
fig, ax = plt.subplots(figsize=(9, 5.5))
xs = [p[0] for p in points]
ys = [p[1] for p in points]
ax.plot(xs, ys, "-o", ms=7, color="#2457c5", label="本仓离线微调曲线（t15.2025.03.30）")
for x, y in points:
    ax.annotate(f"{y:.1f}%", (x, y), xytext=(6, 6), textcoords="offset points",
                fontsize=10)
ax.axhline(21.0, ls="--", c="#9a5b00", lw=1.4, label="当日基线（run_excl5_40k 口径 ~21%）")
ax.axhline(2.5, ls=":", c="#a83232", lw=1.6,
           label="论文在线系统 rapid calibration 后 2.5% WER")
ax.set_xlabel("当日校准句数 N（微调 trial 数）")
ax.set_ylabel("当日 val PER (%)")
ax.set_xticks(xs)
ax.set_title("Card 2024 快速校准复现（负结果）：简单当日微调配方\n"
             "N=10/20/50/165 → 66/56/37/22%（基线 21%），远不及论文在线适配")
ax.grid(alpha=0.3); ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("points:", points)
print("wrote", OUT)
