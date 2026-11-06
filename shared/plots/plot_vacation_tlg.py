# -*- coding: utf-8 -*-
"""Vacation-week TLG results figure (4 panels).

Data provenance (all numbers from docs/plans/20261001_tlg_vacation_plan.md §6):
  A. oracle depth: code/vac_oracle_depth.py runs 2026-10-02 (NB=400 sweeps)
  B. rescorer comparison: T5c fix chain 2026-10-03 (fixed opt_rescore scorer)
  C. error anatomy: code/vac_error_ops.py 2026-10-04
  D. day-adapt lambda: T6b 2026-10-03/04 (null result)
Plot with: D:/apps/psychopy/python.exe code/plot_vacation_tlg.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = (r".\shared\result\figs"
       r"\summary\tlg_vacation_4panel.png")

# ---- A. oracle WER vs n-best depth (NB=400 pools) ----
depths = [1, 5, 10, 25, 50, 100, 200, 400]
oracle = {
    "Willett (880)": [24.85, 17.67, 15.84, 14.04, 12.69, 11.44, 10.55, 9.67],
    "Card (1426)":   [23.14, 18.10, 16.74, 14.89, 13.77, 12.93, 12.22, 11.60],
    "Jude (417)":    [47.39, 37.17, 33.76, 29.71, 26.99, 25.14, 23.46, 22.21],
}
# ---- B. rescorer comparison (fixed scorer, 2026-10-03) ----
resc = {
    "5-gram":            {"Willett": 24.85, "Card": 23.14, "Jude": 47.39},
    "OPT-6.7B":          {"Willett": 22.36, "Card": 21.57, "Jude": 45.54},
    "Qwen3.5-4B":        {"Willett": 23.40, "Card": 22.34, "Jude": 46.36},
    "Qwen3-4B (text)":   {"Willett": 24.11, "Card": 23.12, "Jude": 46.66},
}
paper = {"Willett": 23.8, "Card": None, "Jude": 50.0}
# ---- C. error anatomy (2026-10-04) ----
anat = {  # dataset: (I%, D%, R%)
    "Willett": (2.7, 2.9, 94.4),
    "Card\n(early23)": (4.7, 2.0, 93.3),
    "Card\n(late24+)": (3.5, 9.4, 87.1),
    "Jude": (8.7, 9.3, 82.0),
    "Kunz T15": (5.0, 26.8, 68.2),
}
# ---- D. day-adapt lambda (null) ----
lams = [0, 0.25, 0.5, 0.75, 1.0]
adapt = {
    "pooled":  [23.14, 23.14, 23.17, 23.19, 23.08],
    "late 24+": [41.22, 41.17, 41.14, 41.37, 41.08],
    "last 5":  [62.21, 62.01, 62.31, 62.51, 62.31],
}

fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))
fig.suptitle("TLG vacation week 2026-10-01..07 — decode-stack results "
             "(fixed opt_rescore scorer)", fontsize=13, y=0.995)

ax = axes[0][0]
for name, ys in oracle.items():
    ax.plot(depths, ys, marker="o", label=name)
ax.set_xscale("log")
ax.set_xticks(depths); ax.set_xticklabels(depths)
ax.set_xlabel("n-best depth"); ax.set_ylabel("oracle WER (%)")
ax.set_title("A. Pool oracle vs depth (NB=400, official params)")
ax.grid(alpha=0.3); ax.legend()

ax = axes[0][1]
x = range(3); w = 0.2
for i, (m, vals) in enumerate(resc.items()):
    ax.bar([j + (i - 1.5) * w for j in x], [vals[k] for k in
            ("Willett", "Card", "Jude")], w, label=m)
ax.axhline(23.8, color="C0", ls=":", lw=1)
ax.text(2.45, 23.8, " Willett\n paper 23.8", fontsize=7, va="center", color="C0")
ax.set_xticks(list(x)); ax.set_xticklabels(["Willett", "Card", "Jude"])
ax.set_ylabel("pooled WER (%)")
ax.set_title("B. Rescorer comparison (lower=better)")
ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)

ax = axes[1][0]
names = list(anat)
import numpy as np
xs = np.arange(len(names))
ins = [anat[n][0] for n in names]
dels = [anat[n][1] for n in names]
subs = [anat[n][2] for n in names]
ax.bar(xs, ins, label="Insertion")
ax.bar(xs, dels, bottom=ins, label="Deletion")
ax.bar(xs, subs, bottom=[i + d for i, d in zip(ins, dels)], label="Substitution")
ax.set_xticks(xs); ax.set_xticklabels(names, fontsize=8)
ax.set_ylabel("share of errors (%)")
ax.set_title("C. Error anatomy (word-level I/D/R)")
ax.grid(alpha=0.3, axis="y"); ax.legend(fontsize=8)

ax = axes[1][1]
for name, ys in adapt.items():
    ax.plot(lams, ys, marker="s", label=name)
ax.set_xlabel("correction strength $\\lambda$")
ax.set_ylabel("WER (%)")
ax.set_title("D. Day-adaptation (label-free logit mean shift) — null result")
ax.grid(alpha=0.3); ax.legend()

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig(OUT, dpi=150)
print("saved", OUT)
