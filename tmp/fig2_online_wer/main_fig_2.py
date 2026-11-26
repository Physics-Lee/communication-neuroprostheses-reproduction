# -*- coding: utf-8 -*-
"""Main Figure 2 (Present -- vocabulary funnel to independent reproduction
of the 8 open-data communication neuroprostheses on a 16 GB consumer GPU).

2026-11-12 restoration round (user: keep the single-source repo's
``communication_neuroprostheses_reproduction.pdf`` 6-panel funnel structure,
"do not rewrite"): the 2026-11-11 8-card grid is retired; this module is a
print-ready port of the sister repo's ``render_figure2.py`` layout to the
journal double-column canvas. CNS double-column figures are 180 mm wide with
height free up to a full page, so the canvas is 7.09 x 9.05 in (taller, not
rewritten): every panel, letter, number, funnel drop line, and curly brace
of the 22.4 x 17.1 in original is preserved; only the type scale changes.

Print-ready text diet (user authorized deleting in-figure gray notes; all
deleted strings are archived in docs/20261107_fig345_panel_takeaways.md):
  * panel (a): the 4.6 pt in-bar "how open" notes deleted; runs of
    equal-height bars get ONE group label over a thin bracket line
    ("125k (17 studies)", "130k", "open") instead of 23 repeated labels;
  * panels (b)-(d): per-bar gray span/vocab/caliber tags and the corner
    dagger footnote deleted -- error bars carry the SD, the LaTeX caption
    carries the calibers; tick names compressed to "Author 'YY (IDs)";
  * panels (b)-(f), 2026-11-13 source-data round (user): per-bar value
    numbers deleted -- every bar value and SD whisker is recorded in
    ../latex/source_data/SourceData_Figure2.xlsx, so outside panel (a)
    the figure prints no per-bar numerics; reference-line labels (ASR
    WERs, able-bodied wpm), "n/r" tokens, group titles, and the legend
    remain.

Online WER is calculated from the sister repo's daily word-count CSV.
Published speed and reproduction inputs are retained in local snapshots.
Panel (f) reuses the online WERs and SDs from (c).
Run from figures/:  python main_fig_2.py
"""
import json
import os
import re
import sys

import numpy as np
import yaml
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
import matplotlib.ticker as mticker

from fig_common import *

REPRO_DIR = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "20261106_communication_neuroprostheses_reproduction",
))

# 2026-11-13 star sync (timeline repo 1246594 + sister repo render_figure2):
# criterion is now open data + online (closed-loop) + vocab > 71.4k + online
# WER < 12.8% + at least one vocal-paralysis (△) participant -> Card et al.,
# 2024 (T15) and Jude et al., 2026a (T17/T18); Fan et al., 2023 (T5, intact
# speech ▲) dropped when the vocal-paralysis condition was added.
STAR_KEYS = {"Card et al., 2024", "Jude et al., 2026a"}

# 2026-11-24 (user: drop the Chinese subtitles; retitle e/f): panel
# titles are English-only again and sit at the default 6 pt above the
# axes. The old pad of 13 reserved the line under each title for the
# CJK subtitle; with that line gone the freed ~7 pt per row flows into
# the two runtime-measured funnel corridors, so the canvas height (and
# with it the figure + caption one-page fit) is unchanged.
TITLE_PAD = 6

# Panel (e) tags each paper anchor as online or offline. Our bars are
# offline reruns throughout; panel (f) uses the online anchors from (c).
# Provenance verified against the paper texts / repro repo READMEs:
#   online  = closed-loop / real-time statistic, printed or recomputed
#             from the released real-time decode outputs:
#     Willett '21 (e) 2.78 = raw CER of the real-time decodes on the
#       release's 63-sentence HeldOutTrials subset (Table 1 prints the
#       all-data raw-online CER 5.9 / WER 25.1, which panel (c) carries);
#     Willett '23 = Table 1 "Online" row, PER 19.7;
#     Kunz '25 24/54/52 = the paper's printed closed-loop block values;
#     Jude '26a = raw CER 18.6 / 12.0 from online Fig. 3 Source Data.
#   offline = retrospective / offline benchmark:
#     Fan '23 6.16 = offline eval of the released CORP-v6 recalibration arm;
#     Card '24 9.54 = Brain-to-Text '25 greedy PER baseline;
#     Wairagkar '25 17.0 = offline mel-correlation benchmark
#       (r = 0.83 +- 0.04 on the 956 evaluation sentences);
#     Jude '26b (e) 34.0 = offline PER on day 68/75 (paper: "down to
#       34% ... offline on day 68").
CALIBER_E = {
    "Willett et al., 2021 (T5)": "online",
    "Willett et al., 2023 (T12)": "online",
    "Fan et al., 2023 (T5, CORP v6)": "offline",
    "Card et al., 2024 (T15)": "offline",
    "Wairagkar et al., 2025 (T15)": "offline",
    "Kunz et al., 2025 (T12)": "online",
    "Kunz et al., 2025 (T15)": "online",
    "Kunz et al., 2025 (T16)": "online",
    "Jude et al., 2026a (T17)": "online",
    "Jude et al., 2026a (T18)": "online",
    "Jude et al., 2026b (T17)": "offline",
}
# Online WER uses the common daily word-count calculation. Speed and
# reproduction bars retain their published inputs for this scoped update.
FIG2CD_SNAPSHOT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "data_fig2cd_online_wer_speed_pre20261123.json")
FIG2_REPRODUCTION = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "data_fig2_reproduction.json")
sys.path.insert(0, os.path.join(REPRO_DIR, "figures", "code"))
from build_figure2_data import apply_daily_metrics


def _load_figure2_data():
    """Load figure inputs with online WER calculated from daily word counts.

    Returns
    -------
    dict
        Benchmark rows with equal-day online WER means and sample standard
        deviations. Published speed and reproduction inputs are retained.

    Notes
    -----
    Daily WER is 100 times summed word edits divided by summed reference
    words. Single-day observations have no across-day standard deviation.
    The published listener-transcription median is a separate exception.
    """
    yaml_path = os.path.join(REPRO_DIR, "figures", "reproduction_benchmark.yaml")
    with open(yaml_path, encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    with open(FIG2CD_SNAPSHOT, encoding="utf-8") as stream:
        published = json.load(stream)["panel_cd_online_wer_speed"]
    rows = data["panel_cd_online_wer_speed"]
    if [r["label"] for r in rows] != [r["label"] for r in published]:
        raise ValueError("Published and current Figure 2 study rows differ")
    apply_daily_metrics(data, yaml_path)
    for row, old in zip(rows, published):
        for field in ("speed_wpm_mean", "speed_wpm_sd", "speed_caliber"):
            row[field] = old[field]
    with open(FIG2_REPRODUCTION, encoding="utf-8") as stream:
        reproduction = json.load(stream)
    for key in ("panel_e_repro_acoustic", "panel_f_repro_full_wer"):
        data[key] = reproduction[key]
    return data


def _caliber_tag(ax, x, row, caliber, pad=3.5):
    """Small gray origin tag centered above one ours/paper pair."""
    if caliber is None or (row.get("ours") is None
                           and row.get("paper") is None):
        return
    top = max((row.get(k) or 0.0) + (row.get(k + "_sd") or 0.0)
              for k in ("ours", "paper"))
    ax.text(x, top + pad, caliber, ha="center", va="bottom",
            fontsize=5.5, color="#777777", zorder=4)


def _short_label(label):
    """"Willett et al., 2021 (T5)" -> "Willett '21 (T5)" (vertical ticks)."""
    label = label.replace("\n", " ")
    m = re.match(r"(.+?) et al\., (\d{4})[ab]? \((.+)\)", label)
    if not m:
        return label
    name, year, ids = m.group(1), m.group(2)[2:], m.group(3)
    if name == "Nason-Tomaszewski":
        name = "Nason-T."
    ids = re.sub(r", CORP.*", "", ids)
    ids_map = {
        "T12/T15/T16/T17/T21/T22": "6 ptps",
        "T12/T15/T16/T17": "4 ptps",
        "T12/T15/T16": "T12/15/16",
        "T5/T12/T18": "T5/12/18",
        "T12/T15": "T12/15",
        "T17/T18": "T17/18",
        "V1/V2/V3": "V1-V3",
        "P1/P2": "P1/2",
        "B1r/B3/B6": "B1r-B6",
    }
    ids = ids_map.get(ids, ids)
    return f"{name} '{year} ({ids})"


def _curly_brace(fig, x0, x1, y_tip, y_arm, y_apex, color):
    """Right-pointing curly brace spanning [x0, x1] (geometry ported from
    the sister repo's render_figure2.py)."""
    dx = 0.012
    xm = 0.5 * (x0 + x1)
    verts = [
        (x0, y_tip), (x0, y_arm),
        (x0 + 0.35 * dx, y_arm), (x0 + dx, y_arm), (xm - dx, y_arm),
        (xm - 0.35 * dx, y_arm), (xm, y_arm), (xm, y_apex), (xm, y_arm),
        (xm + 0.35 * dx, y_arm), (xm + dx, y_arm), (x1 - dx, y_arm),
        (x1 - 0.35 * dx, y_arm), (x1, y_arm), (x1, y_tip),
    ]
    codes = [MplPath.MOVETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO,
             MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    fig.add_artist(mpatches.PathPatch(
        MplPath(verts, codes), transform=fig.transFigure,
        facecolor="none", edgecolor=color, linewidth=1.1,
        alpha=0.92, capstyle="round", joinstyle="round", zorder=5))


def main_fig_2():
    plt.rcParams["axes.linewidth"] = 0.4

    data = _load_figure2_data()
    rows_a = data["panel_a_vocab_funnel"]
    rows_b = data["panel_b_dataset_size"]
    rows_cd = data["panel_cd_online_wer_speed"]
    per = data["panel_e_repro_acoustic"]
    wer = {r["label"]: r for r in data["panel_f_repro_full_wer"]}
    assert wer.keys() == {r["label"] for r in rows_cd}
    assert len(rows_a) == 32 and len(rows_b) == len(rows_cd) == 11
    assert len(per) == len(wer) == 11

    W, H = 7.09, 8.51
    fig = plt.figure(figsize=(W, H))

    def _ax(x0_in, y0_in, w_in, h_in):
        return fig.add_axes([x0_in / W, y0_in / H, w_in / W, h_in / H])

    # ---- absolute-inch layout: 3 panel rows + 2 funnel corridors ---------
    # 2026-11-12 LaTeX-fit round: canvas 9.35 -> 8.51 in so that figure +
    # caption fit one letter text block at \linewidth (the previous height
    # overflowed the bottom margin by 70 pt); panel heights and tick rows
    # are untouched -- the savings come from the title band (0.56 -> 0.42),
    # the two funnel corridors (0.83 -> 0.54 each), and the bottom margin
    # (0.14 -> 0.02). Brace geometry stays runtime-measured, so it adapts.
    AX_L = 0.50                      # axes left edge (in)
    # row 1 (panel a): bottom 6.64, h 1.45 (ticks hang to ~5.64)
    ax_a = _ax(AX_L, 6.64, 6.45, 1.45)
    # row 2 (b, c, d): bottom 3.80, h 1.30 (ticks hang to ~2.82)
    pw, gap = 1.95, 0.30
    ax_b = _ax(AX_L, 3.80, pw, 1.30)
    ax_c = _ax(AX_L + pw + gap, 3.80, pw, 1.30)
    ax_d = _ax(AX_L + 2 * (pw + gap), 3.80, pw, 1.30)
    # row 3 (e, f): bottom 1.00, h 1.28 (ticks hang to ~0.02)
    ew = 3.02
    ax_e = _ax(AX_L, 1.00, ew, 1.28)
    ax_f = _ax(AX_L + ew + 0.36, 1.00, ew, 1.28)

    # 2026-11-12 tilt round (user: "why not 45/60/30 deg?"): study ticks
    # at 60 deg -- measured pitch 11.8-19.2 pt vs the >= 9.6 pt
    # perpendicular separation that pitch*sin(60) provides, so parallel
    # neighbors cannot touch; 45 deg fails panels (b)/(c)/(d) (needs
    # pitch*sin(45) >= 9.6 -> pitch >= 13.6 pt); hang depth only shrinks
    # (W*sin60 < W) so the row corridors stay safe
    tick_kw = dict(fontsize=6.2, rotation=60, ha="right", va="top",
                   rotation_mode="anchor")

    # =====================================================================
    # Panel (a): maximum vocabulary size, 32 studies (log scale)
    # =====================================================================
    ax_a.set_yscale("log")
    ax_a.set_ylim(2.2, 5.0e6)
    ax_a.set_yticks([10 ** k for k in range(0, 7)])
    ax_a.set_yticklabels(["1", "10", "100", "1,000", "10,000", "100,000",
                          "1M"], fontsize=6.2)
    ax_a.yaxis.set_minor_locator(mticker.LogLocator(base=10.0, subs="auto"))
    ax_a.tick_params(axis="y", which="minor", labelsize=0, length=1.4,
                     width=0.4)
    ax_a.tick_params(axis="y", which="major", length=2.2, width=0.4,
                     pad=1.5)

    for x, r in enumerate(rows_a):
        v = r["vocab_num"]
        state_c = OPEN_C[r["openness_code"]]
        if r["is_open_vocab"]:
            ax_a.bar(x, v, width=0.76, color=(*to_rgb(state_c), 0.18),
                     edgecolor=state_c, linewidth=0.6,
                     linestyle=(0, (3, 2)))
        else:
            ax_a.bar(x, v, width=0.76, color=(*to_rgb(state_c), 0.35),
                     edgecolor=state_c, linewidth=0.6)

    # equal-height runs -> one bracket line + one group label each
    runs, start = [], 0
    for i in range(1, len(rows_a) + 1):
        if (i == len(rows_a)
                or rows_a[i]["vocab_num"] != rows_a[start]["vocab_num"]):
            runs.append((start, i - 1))
            start = i
    for lo, hi in runs:
        v = rows_a[lo]["vocab_num"]
        if hi > lo:
            y_line = v * 4.0
            ax_a.plot([lo, hi], [y_line, y_line], color="#888888",
                      lw=0.55, zorder=3)
            txt = rows_a[lo]["vocab_display"]
            if v == 125000:
                txt = "125k (17 studies)"
            ax_a.text(0.5 * (lo + hi), y_line * 1.30, txt, ha="center",
                      va="bottom", fontsize=6.3, fontweight="bold",
                      color="#222222", zorder=3)
        else:
            ax_a.text(lo, v * 1.35, rows_a[lo]["vocab_display"], ha="center",
                      va="bottom", fontsize=6.3, zorder=3)

    # Brysbaert anchors: three dashed reference lines with ONE
    # left-aligned key stack in the empty upper-left corner (2026-11-12
    # user round: the labels must share one left edge). On-line badges
    # are physically impossible at a shared x -- the axis spans 6.36
    # decades over 1.45 in (a badge is ~0.61 decade tall) while the
    # 42k/71.4k lines sit only 0.23 decades apart -- so each row leads
    # with a short dashed swatch in the exact line style instead
    for anchor_v, anchor_lab, ky in (
            (11100, "11,100 word families", 2.6e6),
            (42000, "42,000 lemmas", 5.6e5),
            (71400, "71,400 word forms", 1.2e5)):
        ax_a.axhline(anchor_v, color=C_NORM, linewidth=0.6,
                     linestyle=(0, (4, 2)))
        ax_a.plot([-0.52, 0.30], [ky, ky], color=C_NORM, linewidth=0.9,
                  linestyle=(0, (4, 2)), zorder=4)
        ax_a.text(0.48, ky, anchor_lab, ha="left", va="center",
                  fontsize=6.2, color="#555555", zorder=4)

    ax_a.set_xticks(range(len(rows_a)))
    ax_a.set_xticklabels(
        [_short_label(r["label"]) for r in rows_a], **tick_kw)
    ax_a.set_xlim(-0.7, len(rows_a) - 0.3)
    ax_a.set_ylabel("vocabulary size", fontsize=7.0, labelpad=1.5)
    ax_a.set_title("(a) Maximum vocabulary size, 32 studies",
                   fontsize=7.5, fontweight="bold", pad=TITLE_PAD)

    # =====================================================================
    # Row 2: (b) dataset size, (c) online WER, (d) online speed
    # =====================================================================
    for x, r in enumerate(rows_b):
        state_c = OPEN_C[r["openness_code"]]
        ax_b.bar(x, r["released_sentences"], width=0.76,
                 color=(*to_rgb(state_c), 0.35),
                 edgecolor=state_c, linewidth=0.6)
    # per-bar numbers deleted (2026-11-13 source-data round); the
    # headroom that existed only for the staggered labels is reclaimed
    # (9,680-sentence tallest bar -> 10.8k axis top)
    ax_b.set_ylim(0, 10800)
    ax_b.set_yticks(range(0, 8001, 4000))
    ax_b.set_yticklabels([f"{v:,}" for v in range(0, 8001, 4000)],
                         fontsize=6.2)
    ax_b.tick_params(axis="y", pad=1.5)
    ax_b.set_xticks(range(len(rows_b)))
    ax_b.set_xticklabels(
        [("★ " if r["study_key"] in STAR_KEYS else "")
         + _short_label(r["label"]) for r in rows_b], **tick_kw)
    ax_b.set_xlim(-0.75, len(rows_b) + 0.15)
    ax_b.set_ylabel("released sentences", fontsize=7.0, labelpad=1.5)
    ax_b.set_title("(b) Open dataset size", fontsize=7.5, fontweight="bold",
                   pad=TITLE_PAD)

    # per-bar value labels deleted (2026-11-13 source-data round); the
    # error bars still carry the SD visually
    for x, r in enumerate(rows_cd):
        wmean, wsd = r["wer_mean"], r["wer_sd"]
        state_c = OPEN_C[r["openness_code"]]
        ax_c.bar(x, wmean, width=0.76, color=(*to_rgb(state_c), 0.35),
                 edgecolor=state_c, linewidth=0.6)
        if wsd is not None:
            ax_c.errorbar(x, wmean, yerr=wsd, color="#333333", capsize=1.6,
                          elinewidth=0.7, capthick=0.7, zorder=4)
    for asr_v in (2.5, 5.8, 12.8):
        ax_c.axhline(asr_v, color=C_NORM, linewidth=0.6, linestyle=(0, (4, 2)))
    for asr_v, asr_lab, ly, lx_lead in (
            (2.5, "2.5% LibriSpeech", 56.0, 4.55),
            (5.8, "5.8% Switchboard", 65.0, 3.44),
            (12.8, "12.8% 14-set avg.", 74.0, 2.50)):
        ax_c.plot([lx_lead, lx_lead], [asr_v, ly - 0.8], color="#999999",
                  lw=0.5, linestyle="-", zorder=2)
        ax_c.text(5.40, ly, asr_lab, ha="right", va="bottom", fontsize=6.2,
                  color="#666666", zorder=3,
                  bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                            alpha=0.88))
    ax_c.set_ylim(0, 102)
    ax_c.set_yticks(range(0, 101, 25))
    ax_c.tick_params(axis="y", labelsize=6.2, pad=1.5)
    ax_c.set_xticks(range(len(rows_cd)))
    ax_c.set_xticklabels(
        [("★ " if r["study_key"] in STAR_KEYS else "")
         + _short_label(r["label"]) for r in rows_cd], **tick_kw)
    ax_c.set_xlim(-0.75, len(rows_cd) - 0.15)
    ax_c.set_ylabel("online WER (%)", fontsize=7.0, labelpad=1.5)
    ax_c.set_title("(c) Online word error rate", fontsize=7.5,
                   fontweight="bold", pad=TITLE_PAD)

    ax_d.set_yscale("log")
    ax_d.set_ylim(1.5, 420)
    ax_d.set_yticks([2, 5, 10, 20, 50, 100, 200])
    ax_d.set_yticklabels(["2", "5", "10", "20", "50", "100", "200"],
                         fontsize=6.2)
    ax_d.yaxis.set_minor_locator(mticker.FixedLocator(
        [3, 4, 6, 7, 8, 9, 30, 40, 60, 70, 80, 90, 300, 400]))
    ax_d.tick_params(axis="y", which="minor", labelsize=0, length=1.4,
                     width=0.4)
    ax_d.tick_params(axis="y", which="major", pad=1.5)
    for x, r in enumerate(rows_cd):
        spd, spdsd = r["speed_wpm_mean"], r["speed_wpm_sd"]
        state_c = OPEN_C[r["openness_code"]]
        if spd is None:
            ax_d.text(x, 1.78, "n/r", ha="center", va="bottom", fontsize=6.2,
                      color="#777777")
            continue
        ax_d.bar(x, spd, width=0.76, color=(*to_rgb(state_c), 0.35),
                 edgecolor=state_c, linewidth=0.6)
        if spdsd is not None:
            ax_d.errorbar(x, spd, yerr=spdsd, color="#333333", capsize=1.6,
                          elinewidth=0.7, capthick=0.7, zorder=4)
    for spd_v, spd_lab in ((160, "160 conversation"),
                           (51.6, "51.6 keyboard"),
                           (25.1, "25.1 handwriting"),
                           (8, "8 p&c BCI")):
        ax_d.axhline(spd_v, color=C_NORM, linewidth=0.6, linestyle=(0, (4, 2)))
        ax_d.text(3.78, spd_v * 1.05, spd_lab, ha="left", va="bottom",
                  fontsize=6.2, color="#666666", zorder=3,
                  bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none",
                            alpha=0.88))
    ax_d.set_xticks(range(len(rows_cd)))
    ax_d.set_xticklabels(
        [("★ " if r["study_key"] in STAR_KEYS else "")
         + _short_label(r["label"]) for r in rows_cd], **tick_kw)
    ax_d.set_xlim(-0.75, len(rows_cd) - 0.15)
    ax_d.set_ylabel("online speed (wpm)", fontsize=7.0, labelpad=1.5)
    ax_d.set_title("(d) Online communication speed", fontsize=7.5,
                   fontweight="bold", pad=TITLE_PAD)

    # =====================================================================
    # Row 3: (e) reproduction no-LM stage, (f) full-pipeline WER
    # =====================================================================
    groups = ("per", "cer", "mel1r")
    group_titles = {"per": "greedy PER", "cer": "raw CER",
                    "mel1r": "1\u2212r mel (voice)"}
    x_e = np.empty(len(per))
    spans, cursor = [], 0.0
    for g in groups:
        idxs = [i for i, p in enumerate(per) if p["metric"] == g]
        for i in idxs:
            x_e[i] = cursor
            cursor += 1.0
        spans.append((cursor - len(idxs), cursor - 1.0))
        cursor += 0.55
    seps = [0.5 * (spans[k][1] + spans[k + 1][0]) for k in range(2)]

    w = 0.36
    fc_ours = (*to_rgb(C_SPEECH_I), 0.35)
    fc_pap = (*to_rgb(OPEN_C[2]), 0.35)
    # per-pair stacked value labels deleted (2026-11-13 source-data
    # round): ours/paper values and SD whiskers live only in
    # ../latex/source_data/SourceData_Figure2.xlsx

    for i, p in enumerate(per):
        xe = x_e[i]
        ax_e.bar(xe - w / 2, p["ours"], w, color=fc_ours,
                 edgecolor=C_SPEECH_I, linewidth=0.6)
        ax_e.bar(xe + w / 2, p["paper"], w, color=fc_pap,
                 edgecolor=OPEN_C[2], linewidth=0.6)
        if p.get("ours_sd") is not None:
            ax_e.errorbar(xe - w / 2, p["ours"],
                          yerr=[[min(p["ours"], p["ours_sd"])],
                                [p["ours_sd"]]],
                          color="#333333", capsize=1.4, elinewidth=0.6,
                          capthick=0.6, zorder=4)
        if p.get("paper_sd") is not None:
            ax_e.errorbar(xe + w / 2, p["paper"],
                          yerr=[[min(p["paper"], p["paper_sd"])],
                                [p["paper_sd"]]],
                          color="#333333", capsize=1.4, elinewidth=0.6,
                          capthick=0.6, zorder=4)
        _caliber_tag(ax_e, xe, p, CALIBER_E[p["label"]])
    for x_sep in seps:
        ax_e.axvline(x_sep, color="#999999", linewidth=0.6, linestyle=":",
                     ymax=107.0 / 120.0)
    for (g_lo, g_hi), g_key in zip(spans, groups):
        ax_e.text(0.5 * (g_lo + g_hi), 108.5, group_titles[g_key],
                  ha="center", va="bottom", fontsize=6.6, fontweight="bold",
                  color="#444444")
    ax_e.set_xticks(x_e)
    ax_e.set_xticklabels([_short_label(p["label"]) for p in per], **tick_kw)
    ax_e.set_xlim(-0.6, float(x_e.max()) + 0.7)
    ax_e.set_ylim(0, 120)
    ax_e.set_yticks(range(0, 101, 25))
    ax_e.tick_params(axis="y", labelsize=6.2, pad=1.5)
    ax_e.set_ylabel("PER / CER / 1\u2212r (%)", fontsize=7.0, labelpad=1.5)
    ax_e.set_title("(e) Reproduction: PER/CER/(1\u2212r)", fontsize=7.5,
                   fontweight="bold", pad=TITLE_PAD)
    ax_e.legend(handles=[
        mpatches.Patch(facecolor=fc_ours, edgecolor=C_SPEECH_I,
                       linewidth=0.6, label="ours (16 GB GPU)"),
        mpatches.Patch(facecolor=fc_pap, edgecolor=OPEN_C[2], linewidth=0.6,
                       label="paper anchor"),
    ], frameon=True, facecolor="white", edgecolor="none", framealpha=0.9,
        fontsize=6.2, loc="upper right",
        bbox_to_anchor=(1.0, 0.80), borderaxespad=0.1, labelspacing=0.3,
        handlelength=1.2, handletextpad=0.5)

    x_f = np.arange(len(rows_cd))
    for i, online in enumerate(rows_cd):
        v = wer[online["label"]]
        xo, xp = i - w / 2, i + w / 2
        if v["ours"] is None:
            ax_f.text(xo - 0.05, 6, "n/r", ha="center", va="bottom",
                      fontsize=6.2, color="#777777")
        else:
            ax_f.bar(xo, v["ours"], w, color=fc_ours,
                     edgecolor=C_SPEECH_I, linewidth=0.6)
        if v.get("ours_sd") is not None:
            ax_f.errorbar(xo, v["ours"],
                          yerr=[[min(v["ours"], v["ours_sd"])],
                                [v["ours_sd"]]],
                          color="#333333", capsize=1.4, elinewidth=0.6,
                          capthick=0.6, zorder=4)
        ax_f.bar(xp, online["wer_mean"], w, color=fc_pap,
                 edgecolor=OPEN_C[2], linewidth=0.6)
        if online["wer_sd"] is not None:
            ax_f.errorbar(xp, online["wer_mean"], yerr=online["wer_sd"],
                          color="#333333", capsize=1.6, elinewidth=0.7,
                          capthick=0.7, zorder=4)
    ax_f.set_xticks(x_f)
    ax_f.set_xticklabels([_short_label(r["label"]) for r in rows_cd],
                         **tick_kw)
    ax_f.set_xlim(-0.6, len(rows_cd) - 0.3)
    ax_f.set_ylim(0, 120)
    ax_f.set_yticks(range(0, 101, 25))
    ax_f.tick_params(axis="y", labelsize=6.2, pad=1.5)
    ax_f.set_ylabel("WER (%)", fontsize=7.0, labelpad=1.5)
    ax_f.set_title("(f) Reproduction: WER", fontsize=7.5,
                   fontweight="bold", pad=TITLE_PAD)

    # =====================================================================
    # Shared 6-color legend + the two measured funnel corridors
    # =====================================================================
    fig.legend(handles=make_color_legend_handles(), loc="upper center",
               bbox_to_anchor=(0.531, 1.002), ncol=6, frameon=False,
               fontsize=6.2, handlelength=1.0, columnspacing=0.7,
               handletextpad=0.3)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv_fig = fig.transFigure.inverted()

    def _title_y1(axes):
        return max(inv_fig.transform_bbox(
            ax.title.get_window_extent(renderer)).y1 for ax in axes)

    # --- green funnel: (a) selected bars -> bus -> brace over row 2 -------
    guide_c = OPEN_C[2]
    selected_idx = [i for i, r in enumerate(rows_a)
                    if r["vocab_num"] > 71400 and r["openness_code"] == 2]
    drop_points, min_lbl_y0 = [], 1.0
    for i, lbl in enumerate(ax_a.get_xticklabels()):
        bb = inv_fig.transform_bbox(lbl.get_window_extent(renderer))
        min_lbl_y0 = min(min_lbl_y0, bb.y0)
        if i in selected_idx:
            bar_x = inv_fig.transform(ax_a.transData.transform((i, 2.2)))[0]
            x_anchor = 0.5 * (bb.x0 + bb.x1)
            drop_points.append((0.65 * bar_x + 0.35 * x_anchor,
                                bb.y0 - 0.0025))
    title_y1 = _title_y1((ax_b, ax_c, ax_d))
    corridor = min_lbl_y0 - title_y1
    y_bus = min_lbl_y0 - 0.28 * corridor
    y_apex = title_y1 + 0.44 * corridor
    y_arm = title_y1 + 0.22 * corridor
    y_tip = title_y1 + 0.06 * corridor
    for x_pt, y_top in drop_points:
        fig.add_artist(Line2D([x_pt, x_pt], [y_top, y_bus],
                              transform=fig.transFigure, color=guide_c,
                              linewidth=0.9, alpha=0.85,
                              solid_capstyle="round", zorder=5))
        fig.add_artist(Line2D([x_pt], [y_top], transform=fig.transFigure,
                              marker="o", markersize=2.6, color=guide_c,
                              alpha=0.9, zorder=6))
    x_first, x_last = drop_points[0][0], drop_points[-1][0]
    fig.add_artist(Line2D([x_first, x_last], [y_bus, y_bus],
                          transform=fig.transFigure, color=guide_c,
                          linewidth=1.0, alpha=0.88, solid_capstyle="round",
                          zorder=5))
    brace_xm = 0.5 * (ax_b.get_position().x0 + ax_d.get_position().x1)
    fig.add_artist(Line2D([brace_xm, brace_xm], [y_bus, y_apex],
                          transform=fig.transFigure, color=guide_c,
                          linewidth=1.0, alpha=0.92, solid_capstyle="round",
                          zorder=5))
    _curly_brace(fig, ax_b.get_position().x0, ax_d.get_position().x1,
                 y_tip, y_arm, y_apex, guide_c)
    fig.text(brace_xm - 0.008, 0.5 * (y_bus + y_arm) + 0.001,
             "Filtered to 8 open-data studies "
             "(>71,400 word forms)",
             ha="right", va="center", fontsize=6.6, fontweight="bold",
             color=guide_c)

    # --- blue funnel: row-2 bars -> bus -> brace over row 3 ---------------
    rep_c = C_SPEECH_I
    title_y1_r = _title_y1((ax_e, ax_f))
    lbl_y0_r = min(inv_fig.transform_bbox(
        lbl.get_window_extent(renderer)).y0
        for lbl in (*ax_b.get_xticklabels(), *ax_c.get_xticklabels(),
                    *ax_d.get_xticklabels()))
    corridor_r = lbl_y0_r - title_y1_r
    y_bus_c = lbl_y0_r - 0.28 * corridor_r
    y_apex_r = title_y1_r + 0.44 * corridor_r
    y_arm_r = title_y1_r + 0.22 * corridor_r
    y_tip_r = title_y1_r + 0.06 * corridor_r
    drop_pts_c = []
    for i, lbl in enumerate(ax_c.get_xticklabels()):
        bb = inv_fig.transform_bbox(lbl.get_window_extent(renderer))
        x_anchor = 0.5 * (bb.x0 + bb.x1)
        bar_x = inv_fig.transform(ax_c.transData.transform((i, 0)))[0]
        drop_pts_c.append((rows_cd[i], 0.65 * bar_x + 0.35 * x_anchor,
                           bb.y0 - 0.0025))
    for r, x_pt, y_top in drop_pts_c:
        starred = r["study_key"] in STAR_KEYS
        fig.add_artist(Line2D(
            [x_pt, x_pt], [y_top, y_bus_c], transform=fig.transFigure,
            color=rep_c, linewidth=1.3 if starred else 0.7,
            alpha=0.98 if starred else 0.45,
            linestyle="-" if starred else (0, (2.3, 1.7)),
            solid_capstyle="round", zorder=5))
        fig.add_artist(Line2D([x_pt], [y_top], transform=fig.transFigure,
                              marker="o",
                              markersize=3.2 if starred else 2.0,
                              color=rep_c,
                              alpha=0.98 if starred else 0.48, zorder=6))
    fig.add_artist(Line2D([drop_pts_c[0][1], drop_pts_c[-1][1]],
                          [y_bus_c, y_bus_c], transform=fig.transFigure,
                          color=rep_c, linewidth=1.0, alpha=0.88,
                          solid_capstyle="round", zorder=5))
    brace_xmr = 0.5 * (ax_e.get_position().x0 + ax_f.get_position().x1)
    fig.add_artist(Line2D([brace_xmr, brace_xmr], [y_bus_c, y_apex_r],
                          transform=fig.transFigure, color=rep_c,
                          linewidth=1.0, alpha=0.92, solid_capstyle="round",
                          zorder=5))
    _curly_brace(fig, ax_e.get_position().x0, ax_f.get_position().x1,
                 y_tip_r, y_arm_r, y_apex_r, rep_c)
    fig.text(brace_xmr - 0.008, 0.5 * (y_bus_c + y_arm_r) + 0.001,
             "Our reproduction on a 64 GB RAM / RTX 5060 Ti (16 GB) home PC",
             ha="right", va="center", fontsize=6.6, fontweight="bold",
             color=rep_c)

    # 2026-11-12 column-alignment round: (a)/(b)/(e) share the 0.50 spine;
    # their ylabel left edges now share one line too (previously spread
    # 0.17 in -- (a)'s wide funnel-count ticks pushed its label further
    # left than (e)'s narrow ticks did)
    align_ylabel_left(fig, [[ax_a, ax_b, ax_e]])

    save_fig(fig, "main_fig_2")
    plt.close(fig)


if __name__ == "__main__":
    main_fig_2()
