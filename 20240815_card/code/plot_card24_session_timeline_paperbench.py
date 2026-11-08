# -*- coding: utf-8 -*-
"""Session timeline + paper-reported benchmarks overlay (2026-10-14).

Same base panel as plot_card24_figs.py panel D (per-session greedy PER,
validation partition, official pretrained RNN), overlaid with the paper's
own reported numbers so ours vs paper can be read off one chart:
  - red dotted hlines, labeled inline (kept out of the legend to avoid the
    bottom-edge label pile-up flagged in visual QA): paper word-level WER
    benchmarks (after LM):
      9.8  session-2 copy task, 125k vocab (CI 4.1-16.0, shown as a band)
      3.7  conversation-mode mean WER (CI 3.3-4.3)
      2.5  final-five-evaluation-sessions mean copy WER (CI 2.0-3.1)
  - red vline: paper session-2 date (t15.2023.08.18), where 9.8% was set
NOTE: y-axis is our offline greedy phoneme PER; the paper lines are
word-level WER after LM -> cross-unit reference lines, not direct equals.
(The same-unit anchor "PER after LM 2.66%, authors' online record" lives in
summary/card2024_btot25_lm_contribution.png instead.)
The validated original card2024_btot25_session_timeline.png is NOT
overwritten; this writes a separate *_paperbench.png.
Run: D:/apps/psychopy/python.exe code/plot_card24_session_timeline_paperbench.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

INK = "#172033"; BLUE = "#2457c5"; GREEN = "#17683c"; RED = "#a83232"
AMBER = "#9a5b00"; MUTED = "#5c667a"
FIGS = r".\20240815_card\result\figs\session_level"
CSV = r".\20240815_card\result\val_greedy_per_trial.csv"
OUT = FIGS + r"\card2024_btot25_session_timeline_paperbench.png"

S2_DATE = pd.Timestamp("2023-08-18")  # paper session-2 (first 125k-vocab test)

b25 = pd.read_csv(CSV)
b25["date"] = pd.to_datetime(b25.session.str.replace("t15.", "", regex=False),
                             format="%Y.%m.%d")
day25 = (b25.groupby("date")
           .apply(lambda g: 100 * g.edit_distance.sum() / g.true_len.sum(),
                  include_groups=False).sort_index().astype(float))
cnt25 = b25.groupby("date").size().reindex(day25.index)
agg25 = float(100 * b25.edit_distance.sum() / b25.true_len.sum())

fig, ax = plt.subplots(figsize=(11.5, 6), dpi=150)
ax.scatter(day25.index, day25.values, s=8 * cnt25.values, color=BLUE,
           edgecolors=INK, linewidths=0.4, zorder=3)
ax.axhline(agg25, ls="--", c=AMBER, lw=1.4,
           label=f"ours: aggregate greedy PER {agg25:.2f}%")
for d, v in day25.items():
    if v == day25.max() or v == day25.min():
        ax.annotate(f"{d:%Y-%m}\n{v:.1f}%", (d, v), fontsize=9,
                    xytext=(5, 6), textcoords="offset points")

# ---- paper-reported benchmarks (Card 2024 NEJM main text) ----
# hlines labeled inline; anchors auto-searched for a collision-free spot
# against the real transformed bubble positions (v2/v3 hand-placed labels
# grazed data points at these dense PER levels). Labels are placed AFTER
# all layout-affecting artists and tight_layout so their px bboxes are final.
ax.axhspan(4.1, 16.0, color=RED, alpha=0.05, zorder=1,
           label="paper session-2 WER 95% CI (4.1–16.0)")
for v in (9.8, 3.7, 2.5):
    ax.axhline(v, ls=":", c=RED, lw=1.3, zorder=2)
ax.axvline(S2_DATE, ls="--", c=RED, lw=1.0, alpha=0.7, zorder=2,
           label="paper session-2 date (2023-08-18)")


def place_label(text, candidates):
    """Place a (possibly multi-line) label at the first (y, va, x) that
    empirically misses every bubble: render, read the text's real bbox
    via get_window_extent, test against bubble circles, shift on hit.

    Parameters
    ----------
    text : str
        The label text to place (may contain newlines).
    candidates : list of tuple[float, str]
        (y_data, vertical_alignment) anchor candidates to try in order,
        e.g. ``(8.45, "top")``. Each candidate is scanned left-to-right
        across the x range until a collision-free position is found.

    Returns
    -------
    str
        Human-readable description of the accepted anchor (date, y value,
        vertical alignment), printed by the caller for QA.

    Raises
    ------
    RuntimeError
        If no candidate yields a position that misses every bubble.
    """
    pts = [ax.transData.transform((mdates.date2num(d), day25[d]))
           for d in day25.index]
    radii = [np.sqrt(8 * c) * 150 / 72 / 2 + 9 for c in cnt25.values]  # px
    x_lo, x_hi = ax.get_xlim()
    step = (x_hi - x_lo) / 60.0
    for y_data, va in candidates:
        xn = x_lo + step
        while xn <= x_hi - step:
            t = ax.text(mdates.num2date(xn), y_data, text,
                        color=RED, fontsize=8.5, va=va)
            fig.canvas.draw()
            bb = t.get_window_extent()
            hit = any(bb.x0 - r < cx < bb.x1 + r and bb.y0 - r < cy < bb.y1 + r
                      for (cx, cy), r in zip(pts, radii))
            if not hit:
                return f"{mdates.num2date(xn):%Y-%m-%d} y={y_data} va={va}"
            if bb.x1 + step > ax.get_window_extent().x1:
                t.remove()
                break
            t.remove()
            xn += step
    raise RuntimeError("no collision-free anchor for " + text.split("\n")[0])


ax.set_ylabel("per-session greedy PER (%)"); ax.set_xlabel("recording session")
ax.set_title("BtoT25: per-session greedy PER vs paper-reported benchmarks\n"
             "(blue = ours, offline greedy phoneme PER; red dotted = paper word-level WER, after LM)")
for n_ref in (10, 25, 50):
    ax.scatter([], [], s=8 * n_ref, color=MUTED, edgecolors=INK, linewidths=0.4,
               label=f"{n_ref} trials")
ax.legend(fontsize=8.5, labelspacing=1.1, borderpad=0.8, loc="upper center")
ax.grid(alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
ax.set_ylim(0, day25.max() * 1.22)
fig.tight_layout()

print("9.8 label at", place_label(
    "9.8  paper: session-2\ncopy WER (125k vocab)",
    [(8.45, "top"), (7.2, "top"), (9.0, "bottom"), (10.9, "bottom")]))
print("3.7 label at", place_label(
    "3.7  paper: conversation-mode mean WER",
    [(4.2, "bottom"), (5.2, "bottom"), (2.95, "top")]))
print("2.5 label at", place_label(
    "2.5  paper: final-5-sessions mean WER",
    [(0.85, "bottom"), (1.7, "top")]))
fig.savefig(OUT, dpi=150)
print("wrote", OUT)
