# -*- coding: utf-8 -*-
"""Paper-Fig2-style Card 2024 WER panels, regenerated (2026-10-13 P3b).

The original one-off plotting script was not kept through the 2026-10-08
repo split; this rebuild draws the same panels from the byte-verified P2
decode outputs (repro_20261013_final_full.tsv == archive final_full.tsv):
  panel 1: per-session 5-gram WER across all 41 sessions (paper Fig 2 style)
  panel 2: our headline scopes vs paper anchors
Run: D:/apps/psychopy/python.exe code/plot_paper_fig2_panels.py
"""
import csv
import re
from collections import defaultdict

import editdistance
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#172033"; BLUE = "#2457c5"; RED = "#a83232"; AMBER = "#9a5b00"
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False
RUN = (r".\20240815_card\data"
       r"\20241009_card_BtoT25\run_120k")
FINAL = RUN + r"\repro_20261013_final_full.tsv"
GREEDY = RUN + r"\val_greedy_per_trial.csv"
OUT = (r".\20240815_card\result\figs"
       r"\paper_fig2_card2024_panels.png")


def norm(s):
    return re.sub(r"[^A-Za-z'< ]", " ", s).upper().split()


sessions = []
with open(GREEDY, encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        sessions.append(r["session"])

sess_ed = defaultdict(int)
sess_len = defaultdict(int)
with open(FINAL, encoding="utf-8") as f:
    lines = [ln for ln in f if not ln.startswith("key\t")]
for s, line in zip(sessions, lines):
    p = line.rstrip("\n").split("\t")
    r = norm(p[1])
    ed = editdistance.eval(r, p[2].split() if p[2] else [])
    sess_ed[s] += ed
    sess_len[s] += len(r)

order = sorted(sess_len)
dates = [s.replace("t15.", "") for s in order]
wers = [100 * sess_ed[s] / sess_len[s] for s in order]

fig, axes = plt.subplots(1, 2, figsize=(15, 5.5),
                         gridspec_kw={"width_ratios": [2.1, 1]})

ax = axes[0]
ax.plot(range(len(order)), wers, "-o", ms=4, color=BLUE, zorder=3)
ax.axhline(100 * sum(sess_ed.values()) / sum(sess_len.values()), ls="--",
           c=AMBER, lw=1.4,
           label=f"pooled 5-gram WER {100*sum(sess_ed.values())/sum(sess_len.values()):.2f}%")
s2 = order.index("t15.2023.08.18")
ax.scatter([s2], [wers[s2]], marker="*", s=260, color=RED, zorder=5,
           label=f"session-2 (t15.2023.08.18) {wers[s2]:.2f}% vs paper 9.8 (CI 4.1-16.0)")
ax.annotate("2025 追访崩塌\n(当日适配缺失,\n已知现象)", xy=(38.2, 80), fontsize=9,
            color="#7a0177", ha="center")
ax.set_xticks(range(0, len(order), 4))
ax.set_xticklabels([dates[i] for i in range(0, len(order), 4)], rotation=30,
                   ha="right", fontsize=8)
ax.set_ylabel("per-session 5-gram WER (%)")
ax.set_xlabel("recording session (chronological)")
ax.set_title("Card 2024 per-session WER, official decode stack (bp90/acs0.325/NB100)\n"
             "1,426 offline val trials; archive outputs byte-identical")
ax.legend(fontsize=9); ax.grid(alpha=0.3)

ax = axes[1]
scopes = ["session-2\n(5-gram)", "session-2\n(+OPT 修复版)", "2023 池化\n(26 sess, +OPT)",
          "全 41 池化\n(5-gram)", "末 5 session\n(2025, +OPT)"]
vals = [7.58, 3.25, 10.71, 23.14, 62.01]
cols = ["#41b6c4", "#c7e9ff", "#7fcde5", "#1d91c0", "#0c2c84"]
bars = ax.bar(scopes, vals, color=cols, width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.2f}", ha="center",
            fontsize=10, fontweight="bold")
ax.axhspan(4.1, 16.0, color=AMBER, alpha=0.12, label="paper session-2 CI 4.1-16.0")
ax.set_ylabel("WER (%)"); ax.set_ylim(0, 72)
ax.set_title("口径阶梯 (离线混评)")
ax.tick_params(axis="x", labelsize=7.5)
ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

fig.tight_layout()
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
