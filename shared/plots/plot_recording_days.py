# -*- coding: utf-8 -*-
"""Recording-day timelines of the four BrainGate speech datasets.

X axis = calendar date in both panels.
A  one marker per recording day, one row per participant (t12/t15/t16/t17),
   colour = source dataset; shows cadence, clustering and the 2024-08 gap.
B  cumulative count of distinct recording days per (participant, dataset),
   colour = same dataset; slope = recording cadence.

Dates are parsed from hdf5 session dir names (data/*/hdf5 and
nejm-brain-to-text/data/hdf5_data_final) and match the unique Date columns of
the *_description.csv files (checked 2026-09-17, see
docs/20260917_recording_days_comparison.md). kunz splits one recording day
into several condition dirs (attempted/imagined x train/eval) - counted once
per date. kunz t15 2024-03-03 coincides with a btot25 recording day; the
kunz t15 cumulative curve therefore only counts the two dates btot25 lacks.
Output: .\shared\result\figs\recording_days_timeline.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(r".\shared")
OUT = ROOT / r"result\figs\recording_days_timeline.png"

INK = "#172033"
BLUE = "#2457c5"    # btot24
AMBER = "#9a5b00"   # btot25
GREEN = "#17683c"   # jude
RED = "#a83232"     # kunz
MUTED = "#5c667a"

# (patient, dataset, hdf5 dir) -- dir names look like tXX.YYYY.MM.DD[_*]
SOURCES = [
    ("t12", "btot24", ROOT / r"data\20230616_willett_BtoT24\hdf5"),
    ("t15", "btot25", ROOT / r"nejm-brain-to-text\data\hdf5_data_final"),
    ("t17", "jude", ROOT / r"data\20260428_jude\hdf5"),
    ("t12", "kunz", ROOT / r"data\20250821_kunz\hdf5"),
    ("t15", "kunz", ROOT / r"data\20250821_kunz\hdf5"),
    ("t16", "kunz", ROOT / r"data\20250821_kunz\hdf5"),
]
COLOR = {"btot24": BLUE, "btot25": AMBER, "jude": GREEN, "kunz": RED}
PATIENTS = ["t12", "t15", "t16", "t17"]
ROW = {p: i for i, p in enumerate(reversed(PATIENTS))}  # t12 top row

def session_dates(folder: Path, patient: str) -> list:
    out = []
    for d in sorted(folder.iterdir()):
        parts = d.name.split("_")[0].split(".")
        if len(parts) == 4 and d.is_dir() and parts[0] == patient:
            out.append(f"{parts[1]}-{parts[2]}-{parts[3]}")
    return sorted(set(out))

import datetime as _dt  # noqa: E402
groups = {}
for pat, ds, folder in SOURCES:
    dates = session_dates(folder, pat)
    assert dates, f"no session dirs found under {folder} for {pat}"
    groups[(pat, ds)] = [_dt.datetime.strptime(s, "%Y-%m-%d") for s in dates]

b25_dates = {d.date() for d in groups[("t15", "btot25")]}
overlap = [d for d in groups[("t15", "kunz")] if d.date() in b25_dates]

fig, (axA, axB) = plt.subplots(2, 1, figsize=(14, 8.5), sharex=True,
                               height_ratios=[1.0, 1.0])
for ax in (axA, axB):
    ax.grid(alpha=0.3)

# ---------- A: timeline, one row per participant ----------
for (pat, ds), dates in groups.items():
    y = ROW[pat]
    c = COLOR[ds]
    axA.plot([min(dates), max(dates)], [y, y], color=c, alpha=0.30, lw=1.3,
             zorder=1)
    axA.scatter(dates, [y] * len(dates), s=46, color=c, edgecolors=INK,
                linewidths=0.4, zorder=3)
for d in overlap:
    axA.scatter([d], [ROW["t15"]], s=150, facecolors="none", edgecolors=RED,
                linewidths=1.5, zorder=4)
    axA.annotate("same day also in btot25", (d, ROW["t15"]), fontsize=8,
                 color=RED, xytext=(8, 10), textcoords="offset points")

totals = {p: len({d.date() for ds in COLOR for d in groups.get((p, ds), [])})
          for p in PATIENTS}
axA.set_yticks(range(len(PATIENTS)))
axA.set_yticklabels([f"{p}\n{totals[p]} distinct days" for p in reversed(PATIENTS)])
axA.set_ylim(-0.6, len(PATIENTS) - 0.4)
axA.set_ylabel("participant")
axA.set_title("A  one marker = one recording day (row = participant, colour = dataset)",
              loc="left", fontsize=11)
handles = [plt.Line2D([], [], marker="o", ls="", color=COLOR[ds], ms=7,
                      label=f"{ds} ({sum(len(v) for k, v in groups.items() if k[1] == ds)} days)")
           for ds in COLOR]
handles.append(plt.Line2D([], [], marker="o", ls="", mfc="none", mec=RED,
                          ms=9, label="ring: day shared with another dataset"))
axA.legend(handles=handles, fontsize=8, loc="lower left", framealpha=0.9)

# ---------- B: cumulative distinct recording days ----------
order = [("t12", "btot24"), ("t15", "btot25"), ("t17", "jude"),
         ("t12", "kunz"), ("t15", "kunz"), ("t16", "kunz")]
for pat, ds in order:
    dates = groups[(pat, ds)]
    if (pat, ds) == ("t15", "kunz"):
        dates = [d for d in dates if d.date() not in b25_dates]
        label = f"{pat} · {ds} (+{len(dates)} not in btot25)"
    else:
        label = f"{pat} · {ds} ({len(dates)})"
    c = COLOR[ds]
    ys = range(1, len(dates) + 1)
    axB.step(dates, ys, where="post", color=c,
             ls="-" if ds != "kunz" else "--", lw=1.6, label=label, zorder=3)
    axB.annotate(str(len(dates)), (dates[-1], len(dates)), fontsize=8.5,
                 color=c, fontweight="bold", xytext=(5, -2),
                 textcoords="offset points")
axB.set_ylabel("cumulative distinct recording days")
axB.set_ylim(0, 50)
axB.set_title("B  cumulative recording days (slope = cadence; kunz t15 counts only dates absent from btot25)",
              loc="left", fontsize=11)
axB.legend(fontsize=8, loc="upper left", framealpha=0.9)

for ax in (axA, axB):
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
plt.setp(axB.get_xticklabels(), rotation=30, ha="right")
axB.set_xlim(_dt.datetime(2022, 3, 1), _dt.datetime(2025, 7, 1))

fig.suptitle("Four BrainGate speech datasets: when each participant was recorded "
             "(2022-04 to 2025-04)", fontsize=14, y=0.99)
fig.text(0.01, 0.005,
         "dates parsed from hdf5 session dirs (data/*/hdf5, nejm-brain-to-text/data/hdf5_data_final); "
         "match unique Date in *_description.csv. btot24: 24 d · btot25: 45 d · jude: 12 d · kunz: 6 d "
         "(t12×1, t15×3, t16×2). One kunz day = several condition dirs, counted once.",
         fontsize=7.5, color=MUTED)
fig.tight_layout(rect=(0, 0.02, 1, 0.97))
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote:", OUT)
for pat in PATIENTS:
    days = sorted({d.date() for ds in COLOR for d in groups.get((pat, ds), [])})
    print(pat, len(days), "days:", days[0], "->", days[-1])
