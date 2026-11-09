#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone renderer for the 8-Study Consumer-GPU (16 GB VRAM / 64 GB RAM)
Communication Neuroprostheses Reproduction Benchmark (Figure 2).

Reads `reproduction_benchmark.yaml` and `session_per_vs_wer.json` in this
repository, syncs `reproduction_benchmark.json`, verifies zero text overlaps,
and outputs publication-ready `.png` (300 DPI) and vector `.pdf` files.

Usage:
    python render_figure2.py
    python render_figure2.py --yaml reproduction_benchmark.yaml --out communication_neuroprostheses_reproduction
"""
import argparse
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.text import Text
import numpy as np
import yaml

os.environ.setdefault("SOURCE_DATE_EPOCH", "946684800")

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
    "pdf.fonttype": 42,
})

OPEN_C = {
    2: "#00A087",
    1: "#c07f00",
    0: "#cc3324",
    "P": "#7F7F7F",
    "R": "#7B3294",
}
C_SPEECH_I = "#3C5488"
C_REANALYSIS = "#7B3294"
C_NORM = "#7F7F7F"

COLOR_LEGEND_ITEMS = (
    ("open", OPEN_C[2], "open data"),
    ("on_request", OPEN_C[1], "data on request"),
    ("closed", OPEN_C[0], "closed data"),
    ("preprint", OPEN_C["P"], "preprint"),
    ("reanalysis", C_REANALYSIS, "reanalysis on open data"),
    ("repro", C_SPEECH_I, "our reproduction"),
)


def make_color_legend_handles(keys=None):
    """Build matplotlib patch handles for the data-openness color legend.

    Parameters
    ----------
    keys : sequence of str, optional
        Restrict the legend to entries whose key (e.g. ``"open"``,
        ``"closed"``) appears in this sequence. If None, all entries of
        ``COLOR_LEGEND_ITEMS`` are used.

    Returns
    -------
    list of matplotlib.patches.Patch
        Semi-transparent patches carrying the legend labels.
    """
    items = COLOR_LEGEND_ITEMS if keys is None else [
        it for it in COLOR_LEGEND_ITEMS if it[0] in keys
    ]
    return [
        mpatches.Patch(
            facecolor=(*to_rgb(c), 0.35),
            edgecolor=c,
            linewidth=0.8,
            label=lab,
        )
        for _k, c, lab in items
    ]


def check_figure_overlaps(fig, tol_px=1.0):
    """Verify that no two visible text bounding boxes within an axes overlap.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure whose axes will be scanned for colliding text.
    tol_px : float, optional
        Overlap tolerance in pixels; overlaps smaller than this in either
        axis are ignored.

    Returns
    -------
    list of tuple of (str, str, float)
        One entry per overlapping pair: the two texts (newlines replaced
        by ``" | "``) and the overlap area as a percentage of the smaller
        bounding box. Empty when no overlaps are found.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    for ax in fig.axes:
        items = []
        for t in ax.findobj(Text):
            if not t.get_visible():
                continue
            s = t.get_text().strip()
            if not s:
                continue
            bb = t.get_window_extent(r)
            if bb.width <= 1 or bb.height <= 1:
                continue
            items.append((s.replace("\n", " | "), bb))
        for i in range(len(items)):
            s1, b1 = items[i]
            for j in range(i + 1, len(items)):
                s2, b2 = items[j]
                dx = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
                dy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
                if dx > tol_px and dy > tol_px:
                    inter = dx * dy
                    smaller = min(b1.width * b1.height, b2.width * b2.height)
                    bad.append((s1, s2, 100.0 * inter / smaller))
    return bad


def render_figure2(yaml_path=None, session_json_path=None, out_stem="communication_neuroprostheses_reproduction", verify_overlaps=True):
    """Render the full 8-study benchmark overview figure (Figure 2).

    Loads the benchmark YAML, syncs its JSON mirror, draws panels (a)
    vocabulary funnel through (i) session-level PER vs WER including the
    funnel and reproduction curly-brace annotations, optionally verifies
    that no text bounding boxes collide, and saves the figure as
    ``<out_stem>.png`` and ``<out_stem>.pdf``.

    Parameters
    ----------
    yaml_path : str, optional
        Path to ``reproduction_benchmark.yaml``. Defaults to the copy in
        this script's directory.
    session_json_path : str, optional
        Path to ``session_per_vs_wer.json`` used by panel (i). Defaults
        to the copy in this script's directory.
    out_stem : str, optional
        Output path stem; ``.png`` and ``.pdf`` are appended.
    verify_overlaps : bool, optional
        When True, run ``check_figure_overlaps`` before saving and raise
        on any collision.

    Returns
    -------
    tuple of (int, int)
        ``(n_open_pillars, n_repro_pairs)``: the number of dataset-size
        rows (panel b) and reproduction pairs (panel e).

    Raises
    ------
    RuntimeError
        If ``verify_overlaps`` is True and overlapping text is detected.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    if yaml_path is None:
        yaml_path = os.path.join(here, "reproduction_benchmark.yaml")
    if session_json_path is None:
        session_json_path = os.path.join(here, "session_per_vs_wer.json")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Auto-sync JSON mirror alongside YAML
    json_mirror = os.path.splitext(yaml_path)[0] + ".json"
    try:
        with open(json_mirror, "w", encoding="utf-8", newline="\n") as jf:
            json.dump(data, jf, indent=2, ensure_ascii=False)
    except OSError:
        pass

    rows_a = data["panel_a_vocab_funnel"]
    rows_b = data["panel_b_dataset_size"]
    rows_cd = data["panel_cd_online_wer_speed"]
    per = data["panel_e_repro_acoustic"]
    wer = data["panel_f_repro_full_wer"]
    walls = data["panel_g_host_ram_walls"]
    vram = data["panel_h_gpu_vram_walls"]

    fig = plt.figure(figsize=(22.4, 22.9))
    outer = fig.add_gridspec(
        5, 1, height_ratios=(1.02, 0.08, 0.96, 0.92, 0.98), hspace=0.72
    )

    # ---- Row 1: (a) Vocabulary size ----
    # 2026-11-07 open-mechanism round (user: "在fig2a的四个open vocab的
    # 柱子上加简短的灰色小字，介绍他们四个是怎么做到open的"; follow-up:
    # 统一写法; follow-ups: "加在柱子顶端" -> "横向文字"): the four
    # is_open_vocab bars all top out at the same 300k "open" value, so the
    # unified mechanism caption sits horizontally inside each hollow bar,
    # hanging from just below its top edge.
    # hanging from just below its top edge. 2026-11-07 correction (user:
    # Willett-2021's online decode is LM-free -- any LM, n-gram or LLM,
    # would impose a vocabulary ceiling): both brain-to-text bars are
    # "without language model".
    open_how = {
        "Willett et al., 2021": "brain-to-text\nwithout\nlanguage model",
        "Wairagkar et al., 2025": "brain-to-voice",
        "Levin et al., 2026": "brain-to-text\nwithout\nlanguage model",
        "Wairagkar et al., 2026": "brain-to-voice",
    }
    ax1 = fig.add_subplot(outer[0])
    for x, r in enumerate(rows_a):
        v = r["vocab_num"]
        end = r["vocab_display"]
        is_open = r["is_open_vocab"]
        state_c = OPEN_C[r["openness_code"]]
        if is_open:
            ax1.bar(x, v, width=0.76, color=(*to_rgb(state_c), 0.18),
                    edgecolor=state_c, linewidth=0.8,
                    linestyle=(0, (3, 2)))
        else:
            ax1.bar(x, v, width=0.76, color=(*to_rgb(state_c), 0.35),
                    edgecolor=state_c, linewidth=0.8)
        ax1.text(x, v * 1.15, end, ha="center", va="bottom", fontsize=7.6)
        how = open_how.get(r["study_key"]) if is_open else None
        if how is not None:
            ax1.text(x, v * 0.85, how, ha="center", va="top",
                     fontsize=4.6, color="#555555", zorder=3)
    for anchor_v, anchor_lab, above in (
            (11100, "11,100 word families", False),
            (42000, "42,000 lemmas", False),
            (71400, "71,400 word forms", True)):
        ax1.axhline(anchor_v, color=C_NORM, linewidth=0.9,
                    linestyle=(0, (4, 2)))
        if above:
            ax1.text(-0.55, anchor_v * 1.12, anchor_lab +
                     " (Brysbaert et al., 2016)", ha="left", va="bottom",
                     fontsize=7.0, color="#555555")
        else:
            ax1.text(-0.55, anchor_v * 0.72, anchor_lab +
                     " (Brysbaert et al., 2016)", ha="left", va="top",
                     fontsize=7.0, color="#555555")
    ax1.set_yscale("log")
    ax1.set_ylim(2.5, 1.5e6)
    ax1.set_yticks([10 ** k for k in range(0, 7)])
    ax1.set_yticklabels(["1", "10", "100", "1,000", "10,000", "100,000",
                         "1,000,000"])
    ax1.yaxis.set_minor_locator(
        matplotlib.ticker.FixedLocator(
            [m * 10 ** k for k in range(0, 6) for m in range(2, 10)
             if m * 10 ** k >= 2.5]
        )
    )
    ax1.set_xticks(range(len(rows_a)))
    ax1.set_xticklabels([r["label"] for r in rows_a], fontsize=6.7, rotation=75,
                        ha="right", rotation_mode="anchor")
    ax1.set_xlim(-0.7, len(rows_a) - 0.3)
    ax1.set_ylabel("vocabulary size")
    ax1.set_title(f"(a) Maximum vocabulary size used in the {len(rows_a)} "
                  "language-BCI studies",
                  fontsize=10, fontweight="bold")

    # ---- Row 2: (b) Dataset size, (c) WER, (d) Speed ----
    # 2026-11-06 equal-pillar round (user, review repo: "I want the
    # pillars in fig2 b c d have the same width ... calculate
    # [full-width / number-of-pillars]"), blank-space follow-up (user:
    # "there are a lot blank space between fig2c and fig2d"), and n/r
    # round (user: "add the kunz25 and wairagkar25 back to fig2d. leave
    # them as n/r"): with (d) back at the full 11-slot panel-(c) order,
    # all three panels carry 11 pillars over the same 11.5-unit x-span,
    # so the three columns are simply equal (1.176 each) with uniform
    # 0.12 gaps -- pillar = full-width / number-of-pillars is identical
    # everywhere (~0.43 in / 1.10 cm at bar width 0.76 of the slot).
    gs_r2 = outer[2].subgridspec(
        1, 5, width_ratios=(1.176, 0.12, 1.176, 0.12, 1.176), wspace=0.0
    )
    ax_b = fig.add_subplot(gs_r2[0, 0])
    ax2 = fig.add_subplot(gs_r2[0, 2])
    ax3 = fig.add_subplot(gs_r2[0, 4])

    for x, r in enumerate(rows_b):
        n_sent = r["released_sentences"]
        sent_str = r["sentences_display"]
        ds_note = r["span_note"]
        state_c = OPEN_C[r["openness_code"]]
        ax_b.bar(x, n_sent, width=0.76, color=(*to_rgb(state_c), 0.35),
                 edgecolor=state_c, linewidth=0.8)
        ax_b.text(x, n_sent + 220, sent_str, ha="center", va="bottom",
                  fontsize=7.3)
        ax_b.text(x, n_sent + 1150, ds_note, ha="center", va="bottom",
                  fontsize=4.6, color="#555555", linespacing="normal")
    ax_b.set_ylim(0, 14500)
    ax_b.set_yticks(range(0, 12001, 2000))
    ax_b.set_yticklabels([f"{v:,}" for v in range(0, 12001, 2000)])
    ax_b.set_xticks(range(len(rows_b)))
    ax_b.set_xticklabels([r["label"] for r in rows_b],
                         fontsize=6.4, rotation=75, ha="right",
                         rotation_mode="anchor")
    ax_b.set_xlim(-0.75, len(rows_b) - 0.25)
    ax_b.set_ylabel("released sentences")
    # 2026-11-06 same-height-title round (user: "for each row of fig2,
    # make the top of panels of that row have the same height"): the
    # axes tops already align in every row (same gridspec slots;
    # (i)'s equal-aspect shrink takes width, not height), but in row 2
    # the (c)/(d) 2-line titles (series / not-reported qualifiers) top
    # out ~0.18 in above (b)'s 1-line title. The trailing newline gives
    # (b) an empty second title line of identical height, so every
    # panel's title block -- and its first baseline -- sits at one
    # height per row (rows 3/4 are single-line and already aligned).
    ax_b.set_title("(b) Open-sourced dataset size\n", fontsize=9.8,
                   fontweight="bold")

    # (c) Online WER
    for x, r in enumerate(rows_cd):
        wmean = r["wer_mean"]
        wsd = r["wer_sd"]
        vocab = r["vocab_tag"]
        w_cal = r["wer_caliber"]
        state_c = OPEN_C[r["openness_code"]]
        ax2.bar(x, wmean, width=0.76, color=(*to_rgb(state_c), 0.35),
                edgecolor=state_c, linewidth=0.8)
        top = wmean + wsd if wsd is not None else wmean
        if wsd is not None:
            ax2.errorbar(x, wmean, yerr=wsd, color="#333333", capsize=2.0,
                         elinewidth=1.0, capthick=1.0, zorder=4)
        note = f"{wmean:.1f}\u00b1{wsd:.1f}" if wsd is not None else f"{wmean:.1f}"
        ax2.text(x, top + 1.6, note, ha="center", va="bottom", fontsize=6.6)
        ax2.text(x, top + 7.0, f"vocab {vocab}\n{w_cal}", ha="center",
                 va="bottom", fontsize=4.6, color="#555555",
                 linespacing="normal")
    # 2026-11-06 in-panel round (user, review repo: "write the annotations
    # of the grey lines inside the panel? like in fig3d and fig3e"): the
    # three ASR-ruler labels move from the outside-right gutter into the
    # axes as white-badged grey labels, fig3(d)/(e) style. No on-line or
    # right-end seat exists in this 11-bar panel (measured furniture:
    # the 2.5/5.8 lines sit 3.3 WER units apart vs ~3.2 units of 5.6-pt
    # label height, and every x-position at line height crosses a tall
    # bar's fill -- a badge there cuts the bar mid-height), so all three
    # lift as an ascending ladder into the empty band above the left
    # bars' vocab tags (y 62-77: below bar 6's value label at 86.0,
    # above bar 4's tag top at 61.0, right edge 5.5 clears bar 6's
    # column at 5.62). Follow-up (user: "the annotations of 3 grey
    # lines in fig2c is not closed to the arrows"): the 11-slot n/r
    # round narrowed this panel and widened the labels in data units,
    # detaching two leaders from their annotations -- leaders now drop
    # through the three measured corridors between value labels / vocab
    # tags (x = 2.50 / 3.44 / 4.55), each directly under its own label
    # and tucked into its badge (top at ly - 0.2); badges mask whatever
    # a leader crosses behind them -- the fig3(d) pattern.
    for asr_v in (2.5, 5.8, 12.8):
        ax2.axhline(asr_v, color=C_NORM, linewidth=0.9, linestyle=(0, (4, 2)))
    for asr_v, asr_lab, ly, lx_lead in (
            (2.5, "2.5% LibriSpeech (Radford et al., 2022)", 62.6, 4.55),
            (5.8, "5.8% Switchboard (Xiong et al., 2017)", 68.1, 3.44),
            (12.8, "12.8% 14-dataset average (Radford et al., 2022)",
             73.6, 2.50)):
        ax2.plot([lx_lead, lx_lead], [asr_v, ly - 0.2], color="#999999",
                 lw=0.55, linestyle="-", zorder=2)
        ax2.text(5.50, ly, asr_lab, ha="right", va="bottom",
                 fontsize=5.6, color="#666666", zorder=3,
                 bbox=dict(boxstyle="round,pad=0.12", fc="white",
                           ec="none", alpha=0.88))
    ax2.set_ylim(0, 125)
    ax2.set_yticks(range(0, 101, 25))
    ax2.set_xticks(range(len(rows_cd)))
    ax2.set_xticklabels([r["label"] for r in rows_cd],
                        fontsize=6.4, rotation=75, ha="right",
                        rotation_mode="anchor")
    ax2.set_xlim(-0.75, len(rows_cd) - 0.25)
    ax2.set_ylabel("word error rate (%)")
    ax2.set_title("(c) Word error rate (online)\n"
                  "no series printed: Wairagkar et al., 2025 "
                  "(listener median)",
                  fontsize=9.8, fontweight="bold")
    ax2.text(0.98, 0.985,
             "\u2020 recomputed from released data (decode outputs / trial "
             "timing / Source Data files)\n\u2021 digitized from the "
             "paper's figure markers\nerror bar is calculated across days",
             transform=ax2.transAxes, ha="right", va="top",
             fontsize=6.0, color="#555555", linespacing="normal")

    # (d) Communication speed
    # 2026-11-06 n/r round (user: "I think we had better add the kunz25
    # and wairagkar25 back to fig2d. leave them as n/r"): the panel
    # returns to the full 11-slot panel-(c) order (the one-to-one rule);
    # the four speed-None rows (Wairagkar et al., 2025; Kunz et al., 2025
    # T12/T15/T16) draw as explicit gray "n/r" slots at the axis base --
    # panel (f)'s Wairagkar convention -- instead of leaving the axis,
    # and the title's name list collapses to the abbreviation key.
    for x, r in enumerate(rows_cd):
        spd = r["speed_wpm_mean"]
        spdsd = r["speed_wpm_sd"]
        vocab = r["vocab_tag"]
        s_cal = r["speed_caliber"]
        state_c = OPEN_C[r["openness_code"]]
        if spd is None:
            ax3.text(x, 1.2, "n/r", ha="center", va="bottom",
                     fontsize=5.6, color="#777777")
            continue
        ax3.bar(x, spd, width=0.76,
                color=(*to_rgb(state_c), 0.35),
                edgecolor=state_c, linewidth=0.8)
        top = spd + spdsd if spdsd is not None else spd
        if spdsd is not None:
            ax3.errorbar(x, spd, yerr=spdsd, color="#333333", capsize=2.0,
                         elinewidth=1.0, capthick=1.0, zorder=4)
        note = f"{spd:.1f}\u00b1{spdsd:.1f}" if spdsd is not None else f"{spd:.1f}"
        ax3.text(x, top * 1.06 + 2.0, note, ha="center", va="bottom",
                 fontsize=6.8)
        ax3.text(x, max(top * 1.31, top * 1.06 + 8.0) + 2.0,
                 f"vocab {vocab}\n{s_cal}",
                 ha="center", va="bottom", fontsize=4.6, color="#555555",
                 linespacing="normal")
    # 2026-11-06 in-panel label round, re-seated for the 11-slot axis
    # (same user round as panel (c)); follow-up (user: "write 160 wpm
    # natural conversation (Card et al., 2024) on the top of the
    # y=160"): ALL FOUR line labels now sit directly ON their lines
    # (badge resting on the dashed line, +0.6 offset) -- the empty n/r
    # band x 4-7 and the clear strip above Jude-b's tag give every one
    # a clean on-line seat, 160 included (its badge tops at ~166, the
    # 3-line dagger legend's bottom sits at ~177); only 8 cannot (its
    # on-line band is taken by the 10-label above and the n/r texts
    # below), so it lifts to y = 20 with a short leader at x = 5.5,
    # dropping between the n/r slots. White badges (alpha 0.88) mask
    # whatever a leader crosses behind them, the fig3(d) pattern.
    # 2026-11-06 follow-up (user: "the annotations of 3 grey lines in
    # fig2d can be written in the same start x-coordinate of 5-10 wpm
    # eye typing"): all five line annotations left-align at one start
    # x = 3.92 (the measured left edge of the eye-typing label, mid-way
    # over the empty n/r band), replacing the staggered right-anchored
    # seats; each label keeps its on-line / lifted y seat, and every
    # shifted span re-clears its neighbors (51.6 now ends 2.4 units
    # clear of Jude-b's value label at 9.61; 36.2 stays clear of the
    # slot-8/9 vocab tags; the lifted 8-label ends 0.57 left of the
    # slot-8 tag with its leader still beneath it).
    for spd_v in (160, 51.6, 36.2, 10, 8):
        ax3.axhline(spd_v, color=C_NORM, linewidth=0.9, linestyle=(0, (4, 2)))
    LX = 3.92
    for spd_v, spd_lab in (
            (160, "160 wpm natural conversation (Card et al., 2024)"),
            (51.6, "51.6 wpm keyboard typing (Dhakal et al., 2018)"),
            (36.2, "36.2 wpm smartphone typing (Palin et al., 2019)"),
            (10, "5\u201310 wpm eye typing (Hamid & Kristensson, 2024)")):
        ax3.text(LX, spd_v + 0.6, spd_lab, ha="left", va="bottom",
                 fontsize=5.6, color="#666666", zorder=3,
                 bbox=dict(boxstyle="round,pad=0.12", fc="white",
                           ec="none", alpha=0.88))
    ax3.plot([5.5, 5.5], [8, 19.8], color="#999999",
             lw=0.55, linestyle="-", zorder=2)
    ax3.text(LX, 20.0, "8 wpm point-and-click BCI (Willett et al., 2021)",
             ha="left", va="bottom", fontsize=5.6, color="#666666",
             zorder=3,
             bbox=dict(boxstyle="round,pad=0.12", fc="white",
                       ec="none", alpha=0.88))
    ax3.set_ylim(0, 200)
    ax3.set_yticks(range(0, 201, 50))
    ax3.set_xticks(range(len(rows_cd)))
    ax3.set_xticklabels([r["label"] for r in rows_cd],
                        fontsize=6.4, rotation=75, ha="right",
                        rotation_mode="anchor")
    ax3.set_xlim(-0.75, len(rows_cd) - 0.25)
    ax3.set_ylabel("words per minute")
    # second line carries the n/r key (the 2026-10-29 name list retired
    # with the n/r slots); two lines keep row 2's title tops level with
    # (c)'s 2-line title and (b)'s padded 1-line title
    ax3.set_title("(d) Communication speed (online)\n"
                  "n/r: speed not reported",
                  fontsize=9.8, fontweight="bold")
    ax3.text(0.98, 0.985,
             "\u2020 recomputed from released data (decode outputs / trial "
             "timing / Source Data files)\n\u2021 digitized from the "
             "paper's figure markers\nerror bar is calculated across days",
             transform=ax3.transAxes, ha="right", va="top",
             fontsize=6.0, color="#555555", linespacing="normal")

    # ---- Row 3: (e) Reproduction greedy PER/CER, (f) Reproduction full WER
    gs_r4 = outer[3].subgridspec(1, 2, width_ratios=(1.0, 1.0), wspace=0.16)
    ax_h = fig.add_subplot(gs_r4[0, 0])
    ax_i = fig.add_subplot(gs_r4[0, 1])
    # ---- Row 4: (g) memory walls | (h) GPU VRAM | (i) PER vs WER
    gs_r5 = outer[4].subgridspec(
        1, 3, width_ratios=(0.84, 1.04, 0.52), wspace=0.16
    )
    ax_j = fig.add_subplot(gs_r5[0, 0])
    ax_v = fig.add_subplot(gs_r5[0, 1])
    ax_p = fig.add_subplot(gs_r5[0, 2])

    fc_ours = (*to_rgb(C_SPEECH_I), 0.35)
    c_ref = OPEN_C[2]
    fc_ref = (*to_rgb(c_ref), 0.35)

    x_h = np.arange(len(per))
    w = 0.36
    ax_h.bar(x_h - w / 2, [p["ours"] for p in per], w,
             label="our reproduction (16 GB home PC)",
             color=fc_ours, edgecolor=C_SPEECH_I, linewidth=0.8)
    ax_h.bar(x_h + w / 2, [p["paper"] for p in per], w,
             label="paper / benchmark anchor",
             color=fc_ref, edgecolor=c_ref, linewidth=0.8)
    for i, p in enumerate(per):
        o_val, o_sd = p["ours"], p.get("ours_sd")
        p_val, p_sd = p["paper"], p.get("paper_sd")
        top_o = o_val + o_sd if o_sd is not None else o_val
        top_p = p_val + p_sd if p_sd is not None else p_val
        if o_sd is not None:
            ax_h.errorbar(i - w / 2, o_val,
                          yerr=[[min(o_val, o_sd)], [o_sd]],
                          color="#333333", capsize=1.8,
                          elinewidth=0.9, capthick=0.9, zorder=4)
        if p_sd is not None:
            ax_h.errorbar(i + w / 2, p_val,
                          yerr=[[min(p_val, p_sd)], [p_sd]],
                          color="#333333", capsize=1.8,
                          elinewidth=0.9, capthick=0.9, zorder=4)
        o_txt = f"{o_val:.1f}\u00b1{o_sd:.1f}" if o_sd is not None else f"{o_val:.1f}"
        p_txt = f"{p_val:.1f}\u00b1{p_sd:.1f}" if p_sd is not None else f"{p_val:.1f}"
        ax_h.text(i - w / 2 - 0.02, top_o + 1.1, o_txt,
                  ha="center", va="bottom", fontsize=5.0)
        ax_h.text(i + w / 2 + 0.02, top_p + 1.1, p_txt,
                  ha="center", va="bottom", fontsize=5.0)
        if p["note"] is not None:
            ymax_h = max(top_o, top_p)
            ax_h.text(i, ymax_h + 6.2, p["note"], ha="center", va="bottom",
                      fontsize=4.8, color="#555555")
    ax_h.set_xticks(x_h)
    ax_h.set_xticklabels([p["label"] for p in per], fontsize=6.0, rotation=75,
                         ha="right", rotation_mode="anchor")
    ax_h.set_ylabel("Greedy PER / CER / 1\u2212r (%)")
    ax_h.set_ylim(0, 105)
    ax_h.set_yticks(range(0, 101, 25))
    ax_h.legend(frameon=False, fontsize=6.6, loc="upper left")
    ax_h.text(0.98, 0.97, "error bar is calculated across days",
              transform=ax_h.transAxes, ha="right", va="top",
              fontsize=6.0, color="#555555")
    ax_h.set_title("(e) Reproduction: acoustic / synthesis stage (no LM), "
                   "one pair per panel-(c) bar",
                   fontsize=9.3, fontweight="bold")

    # (f) Full-pipeline WER
    x_i = np.arange(len(wer))
    for i, v in enumerate(wer):
        if v["ours"] is None:
            ax_i.text(i, 8, v["note"], ha="center", va="bottom",
                      fontsize=5.2, color="#777777",
                      linespacing="normal")
            continue
        o_val, o_sd = v["ours"], v.get("ours_sd")
        p_val, p_sd = v["paper"], v.get("paper_sd")
        top_o = o_val + o_sd if o_sd is not None else o_val
        top_p = p_val + p_sd if p_sd is not None else p_val
        ax_i.bar(i - w / 2, o_val, w,
                 label="our reproduction (16 GB home PC)" if i == 0 else None,
                 color=fc_ours, edgecolor=C_SPEECH_I, linewidth=0.8)
        ax_i.bar(i + w / 2, p_val, w,
                 label="paper / benchmark anchor" if i == 0 else None,
                 color=fc_ref, edgecolor=c_ref, linewidth=0.8)
        if o_sd is not None:
            ax_i.errorbar(i - w / 2, o_val,
                          yerr=[[min(o_val, o_sd)], [o_sd]],
                          color="#333333", capsize=1.8,
                          elinewidth=0.9, capthick=0.9, zorder=4)
        if p_sd is not None:
            ax_i.errorbar(i + w / 2, p_val,
                          yerr=[[min(p_val, p_sd)], [p_sd]],
                          color="#333333", capsize=1.8,
                          elinewidth=0.9, capthick=0.9, zorder=4)
        o_txt = f"{o_val:.1f}\u00b1{o_sd:.1f}" if o_sd is not None else f"{o_val:.1f}"
        p_txt = f"{p_val:.1f}\u00b1{p_sd:.1f}" if p_sd is not None else f"{p_val:.1f}"
        ax_i.text(i - w / 2 - 0.02, top_o + 1.4, o_txt,
                  ha="center", va="bottom", fontsize=5.0)
        ax_i.text(i + w / 2 + 0.02, top_p + 1.4, p_txt,
                  ha="center", va="bottom", fontsize=5.0)
        ymax_i = max(top_o, top_p)
        ax_i.text(i, ymax_i + 7.8, v["note"], ha="center", va="bottom",
                  fontsize=4.8, color="#555555", linespacing="normal")
    ax_i.set_xticks(x_i)
    ax_i.set_xticklabels([v["label"] for v in wer], fontsize=6.0, rotation=75,
                         ha="right", rotation_mode="anchor")
    ax_i.set_ylabel("Word error rate (%)")
    ax_i.set_ylim(0, 140)
    ax_i.set_yticks(range(0, 126, 25))
    ax_i.legend(frameon=False, fontsize=6.6, loc="upper left")
    ax_i.text(0.98, 0.97, "error bar is calculated across days",
              transform=ax_i.transAxes, ha="right", va="top",
              fontsize=6.0, color="#555555")
    ax_i.set_title("(f) Reproduction: full-pipeline WER, one pair per "
                   "panel-(c) bar",
                   fontsize=9.3, fontweight="bold")

    # (g) Host RAM walls
    x_j = np.arange(len(walls))
    c_off = OPEN_C[2]
    fc_off = (*to_rgb(c_off), 0.35)
    ax_j.bar(x_j - w / 2, [r["official_gb"] for r in walls], w,
             label="official / naive stack",
             color=fc_off, edgecolor=c_off, linewidth=0.8)
    ax_j.bar(x_j + w / 2, [r["ours_gb"] for r in walls], w,
             label="our consumer-PC stack",
             color=fc_ours, edgecolor=C_SPEECH_I, linewidth=0.8)
    for i, r in enumerate(walls):
        ax_j.text(i, max(r["official_gb"], r["ours_gb"]) * 1.16, r["note"],
                  ha="center", va="bottom", fontsize=5.1,
                  color="#333333", linespacing="normal")
    ax_j.set_ylim(0, 500)
    ax_j.set_yticks(range(0, 501, 100))
    ax_j.axhline(64.0, color=C_NORM, linewidth=1.0,
                 linestyle=(0, (4, 2)), zorder=2.6)
    ax_j.set_xticks(x_j)
    ax_j.set_xticklabels([r["category"] for r in walls], fontsize=6.2, rotation=75,
                         ha="right", rotation_mode="anchor")
    ax_j.set_xlim(-0.62, len(walls) - 0.38)
    ax_j.set_ylabel("Peak host RAM (GB)")
    ax_j.legend(handles=[
        mpatches.Patch(facecolor=fc_off, edgecolor=c_off, linewidth=0.8,
                       label="official / naive stack"),
        mpatches.Patch(facecolor=fc_ours, edgecolor=C_SPEECH_I, linewidth=0.8,
                       label="our consumer-PC stack"),
        Line2D([0], [0], color=C_NORM, linewidth=1.0,
               linestyle=(0, (4, 2)), label="64 GB our host RAM"),
    ], frameon=False, fontsize=6.4, loc="upper right", ncol=1,
        columnspacing=1.0, handlelength=1.4, handletextpad=0.5)
    ax_j.set_title("(g) Host RAM walls: official stacks vs. our 64 GB PC",
                   fontsize=9.6, fontweight="bold")

    # (h) GPU VRAM walls
    x_v = np.arange(len(vram))
    ax_v.bar(x_v - w / 2, [r["official_gb"] for r in vram], w,
             color=fc_off, edgecolor=OPEN_C[2], linewidth=0.8)
    ax_v.bar(x_v + w / 2, [r["ours_gb"] for r in vram], w,
             color=fc_ours, edgecolor=C_SPEECH_I, linewidth=0.8)
    for i, r in enumerate(vram):
        ax_v.text(i, max(r["official_gb"], r["ours_gb"]) * 1.16, r["note"],
                  ha="center", va="bottom", fontsize=5.0, color="#333333",
                  linespacing="normal")
    ax_v.set_ylim(0, 130.0)
    ax_v.set_yticks(range(0, 131, 25))
    ax_v.axhline(16.0, color=C_NORM, linewidth=1.0,
                 linestyle=(0, (4, 2)), zorder=2.6)
    ax_v.set_xticks(x_v)
    ax_v.set_xticklabels([r["category"] for r in vram], fontsize=6.2, rotation=75,
                         ha="right", rotation_mode="anchor")
    ax_v.set_xlim(-0.62, len(vram) - 0.38)
    ax_v.set_ylabel("Peak GPU VRAM (GB)")
    ax_v.legend(handles=[
        mpatches.Patch(facecolor=fc_off, edgecolor=OPEN_C[2], linewidth=0.8,
                       label="official provisioned GPU VRAM"),
        mpatches.Patch(facecolor=fc_ours, edgecolor=C_SPEECH_I,
                       linewidth=0.8,
                       label="our peak VRAM (1\u00d7 RTX 5060 Ti)"),
        Line2D([0], [0], color=C_NORM, linewidth=1.0,
               linestyle=(0, (4, 2)), label="16 GB our GPU VRAM"),
    ], frameon=False, fontsize=6.4, loc="upper right", ncol=3,
        columnspacing=1.0, handlelength=1.4, handletextpad=0.5)
    ax_v.set_title("(h) GPU VRAM: official rigs vs. our single 16 GB GPU",
                   fontsize=9.6, fontweight="bold")

    # (i) Session-level PER vs post-LM WER phase diagram
    with open(session_json_path, "r", encoding="utf-8") as sf:
        _data = json.load(sf)
    _style = {
        "Willett 2023 (T12)":   ("#3C5488", "o", "Willett et al., 2023 (T12)"),
        "Card 2024 (T15)":      ("#4DBBD5", "s", "Card et al., 2024 (T15)"),
        "Jude 2026 (LIS, T17)": ("#1B6CA8", "D", "Jude et al., 2026b (T17)"),
    }
    _lim = 80
    _x_curve = np.linspace(0, 41.0, 200)
    _y_curve = 100.0 * (1.0 - (1.0 - _x_curve / 100.0) ** 3.4)
    ax_p.plot(_x_curve, _y_curve, ls="-.", lw=1.15, color="#888888", zorder=1.2)
    ax_p.text(1.5, 81.8,
              "No-LM baseline:\n1 \u2212 (1 \u2212 PER)$^{3.4}$",
              fontsize=5.9, color="#666666", ha="left", va="top",
              linespacing=1.12)
    ax_p.plot([0, _lim], [0, _lim], ls="--", lw=1.2, color="#999999", zorder=1)
    ax_p.axvline(25.0, color=C_NORM, lw=1.0, linestyle=(0, (4, 2)), zorder=1.5)
    ax_p.text(26.2, 23.4,
              "PER = 25%\n"
              "(\u226425%: 45/57 = 79% WER < PER;\n"
              " >25%: 4/20 = 20% WER < PER)\n"
              "This means that we must get PER < 25%.",
              fontsize=5.7, color="#555555", ha="left", va="top",
              linespacing=1.08)
    ax_p.text(60.0, 57.5, "WER = PER",
              fontsize=6.4, color="#777777", ha="left", va="top",
              rotation=45, rotation_mode="anchor")
    for _name, _v in _data.items():
        if _name not in _style:
            continue
        _c, _m, _lab = _style[_name]
        ax_p.scatter([s["per"] for s in _v["sessions"]],
                     [s["wer"] for s in _v["sessions"]],
                     s=26, marker=_m, facecolor="none", edgecolor=_c,
                     lw=1.4, alpha=0.85, zorder=3, label=_lab)
        ax_p.scatter([_v["pooled"]["per"]], [_v["pooled"]["wer"]], s=80,
                     marker="*", color=_c, edgecolor="white", lw=0.6,
                     zorder=4)
    ax_p.legend(fontsize=6.5, frameon=False, loc="lower right",
                title="per-session (open)   pooled (star)",
                title_fontsize=6.6, borderaxespad=0.2, labelspacing=0.35)
    ax_p.set_xlim(0, _lim)
    ax_p.set_ylim(0, _lim + 3)
    ax_p.set_aspect("equal", adjustable="box")
    ax_p.set_xlabel("greedy phoneme error rate per session (%)")
    ax_p.set_ylabel("word error rate after LM stage (%)")
    ax_p.set_title("(i) Session-level PER vs. post-LM WER",
                   fontsize=9.6, fontweight="bold")

    fig.legend(handles=make_color_legend_handles(), loc="upper center",
               bbox_to_anchor=(0.512, 0.992), ncol=6, frameon=False,
               fontsize=8.4, handlelength=1.4, columnspacing=1.5)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.952, bottom=0.085)

    for _ax in (ax_h, ax_i):
        _bb = _ax.get_position()
        _ax.set_position([_bb.x0, _bb.y0 - 0.010, _bb.width, _bb.height])

    # Visual funnel between Row 1 (panel a) and Row 2 (panels b-d):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv_fig = fig.transFigure.inverted()
    selected_idx = [
        i for i, r in enumerate(rows_a)
        if r["vocab_num"] > 71400 and r["openness_code"] == 2
    ]
    xtick_objs = ax1.get_xticklabels()
    drop_points = []
    min_lbl_y0 = 1.0
    for i, lbl in enumerate(xtick_objs):
        bb = inv_fig.transform_bbox(lbl.get_window_extent(renderer))
        if bb.y0 < min_lbl_y0:
            min_lbl_y0 = bb.y0
        if i in selected_idx:
            x_anchor = 0.5 * (bb.x0 + bb.x1)
            bar_x = inv_fig.transform(ax1.transData.transform((i, 2.5)))[0]
            x_pt = 0.65 * bar_x + 0.35 * x_anchor
            drop_points.append((x_pt, bb.y0 - 0.0025))

    title_y1 = max(
        inv_fig.transform_bbox(ax.title.get_window_extent(renderer)).y1
        for ax in (ax_b, ax2, ax3)
    )
    brace_x0 = ax_b.get_position().x0
    brace_x1 = ax3.get_position().x1
    brace_xm = 0.5 * (brace_x0 + brace_x1)

    corridor = min_lbl_y0 - title_y1
    y_bus = min_lbl_y0 - 0.28 * corridor
    y_apex = title_y1 + 0.42 * corridor
    y_arm = title_y1 + 0.22 * corridor
    y_tip = title_y1 + 0.06 * corridor

    guide_c = OPEN_C[2]
    for x_pt, y_top in drop_points:
        fig.add_artist(Line2D(
            [x_pt, x_pt], [y_top, y_bus],
            transform=fig.transFigure, color=guide_c,
            linewidth=1.15, alpha=0.85, solid_capstyle="round", zorder=5
        ))
        fig.add_artist(Line2D(
            [x_pt], [y_top],
            transform=fig.transFigure, marker="o", markersize=3.4,
            color=guide_c, alpha=0.9, zorder=6
        ))

    x_first = drop_points[0][0]
    x_last = drop_points[-1][0]
    fig.add_artist(Line2D(
        [x_first, x_last], [y_bus, y_bus],
        transform=fig.transFigure, color=guide_c,
        linewidth=1.25, alpha=0.88, solid_capstyle="round", zorder=5
    ))
    fig.add_artist(Line2D(
        [brace_xm, brace_xm], [y_bus, y_apex],
        transform=fig.transFigure, color=guide_c,
        linewidth=1.4, alpha=0.92, solid_capstyle="round", zorder=5
    ))

    dx_curl = 0.016
    verts = [
        (brace_x0, y_tip),
        (brace_x0, y_arm),
        (brace_x0 + 0.35 * dx_curl, y_arm),
        (brace_x0 + dx_curl, y_arm),
        (brace_xm - dx_curl, y_arm),
        (brace_xm - 0.35 * dx_curl, y_arm),
        (brace_xm, y_arm),
        (brace_xm, y_apex),
        (brace_xm, y_arm),
        (brace_xm + 0.35 * dx_curl, y_arm),
        (brace_xm + dx_curl, y_arm),
        (brace_x1 - dx_curl, y_arm),
        (brace_x1 - 0.35 * dx_curl, y_arm),
        (brace_x1, y_arm),
        (brace_x1, y_tip),
    ]
    codes = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
    ]
    fig.add_artist(mpatches.PathPatch(
        MplPath(verts, codes), transform=fig.transFigure,
        facecolor="none", edgecolor=guide_c, linewidth=1.45,
        alpha=0.92, capstyle="round", joinstyle="round", zorder=5
    ))
    fig.text(
        brace_xm - 0.008, 0.5 * (y_bus + y_arm) + 0.001,
        "Filtered to 8 open-data studies exceeding native-speaker vocabulary (>71,400 word forms)",
        ha="right", va="center", fontsize=7.4, fontweight="bold",
        color=guide_c
    )

    # Second curly brace gathering reproduction panels (e-f)
    title_y1_rep = max(
        inv_fig.transform_bbox(ax.title.get_window_extent(renderer)).y1
        for ax in (ax_h, ax_i)
    )
    lbl_y0_rep = min(
        inv_fig.transform_bbox(lbl.get_window_extent(renderer)).y0
        for lbl in (*ax_b.get_xticklabels(), *ax2.get_xticklabels(),
                    *ax3.get_xticklabels())
    )
    corridor_rep = lbl_y0_rep - title_y1_rep
    y_tip_r = title_y1_rep + 0.05 * corridor_rep
    y_arm_r = title_y1_rep + 0.30 * corridor_rep
    y_apex_r = title_y1_rep + 0.58 * corridor_rep
    brace_x0r = ax_h.get_position().x0
    brace_x1r = ax_i.get_position().x1
    brace_xmr = 0.5 * (brace_x0r + brace_x1r)
    verts_r = [
        (brace_x0r, y_tip_r),
        (brace_x0r, y_arm_r),
        (brace_x0r + 0.35 * dx_curl, y_arm_r),
        (brace_x0r + dx_curl, y_arm_r),
        (brace_xmr - dx_curl, y_arm_r),
        (brace_xmr - 0.35 * dx_curl, y_arm_r),
        (brace_xmr, y_arm_r),
        (brace_xmr, y_apex_r),
        (brace_xmr, y_arm_r),
        (brace_xmr + 0.35 * dx_curl, y_arm_r),
        (brace_xmr + dx_curl, y_arm_r),
        (brace_x1r - dx_curl, y_arm_r),
        (brace_x1r - 0.35 * dx_curl, y_arm_r),
        (brace_x1r, y_arm_r),
        (brace_x1r, y_tip_r),
    ]
    codes_r = [
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
    ]
    rep_c = C_SPEECH_I
    fig.add_artist(mpatches.PathPatch(
        MplPath(verts_r, codes_r), transform=fig.transFigure,
        facecolor="none", edgecolor=rep_c, linewidth=1.45,
        alpha=0.92, capstyle="round", joinstyle="round", zorder=5
    ))
    drop_pts_c = []
    for i, lbl in enumerate(ax2.get_xticklabels()):
        bb = inv_fig.transform_bbox(lbl.get_window_extent(renderer))
        x_anchor = 0.5 * (bb.x0 + bb.x1)
        bar_x = inv_fig.transform(ax2.transData.transform((i, 0)))[0]
        x_pt = 0.65 * bar_x + 0.35 * x_anchor
        drop_pts_c.append((x_pt, bb.y0 - 0.0025))
    y_bus_c = lbl_y0_rep - 0.28 * corridor_rep
    for x_pt, y_top in drop_pts_c:
        fig.add_artist(Line2D(
            [x_pt, x_pt], [y_top, y_bus_c],
            transform=fig.transFigure, color=rep_c,
            linewidth=1.15, alpha=0.85, solid_capstyle="round", zorder=5
        ))
        fig.add_artist(Line2D(
            [x_pt], [y_top],
            transform=fig.transFigure, marker="o", markersize=3.0,
            color=rep_c, alpha=0.9, zorder=6
        ))
    fig.add_artist(Line2D(
        [drop_pts_c[0][0], drop_pts_c[-1][0]], [y_bus_c, y_bus_c],
        transform=fig.transFigure, color=rep_c,
        linewidth=1.25, alpha=0.88, solid_capstyle="round", zorder=5
    ))
    fig.add_artist(Line2D(
        [brace_xmr, brace_xmr], [y_bus_c, y_apex_r],
        transform=fig.transFigure, color=rep_c,
        linewidth=1.4, alpha=0.92, solid_capstyle="round", zorder=5
    ))
    # 2026-11-06 (user: "is below the bracket, I want it above of the
    # bracket"): the label leaves the below-arm band (2026-10-30 v4 had
    # moved it under the arm so the drop lines owned the band above;
    # the drops end AT the bus, so the bus-to-arm band is free) and
    # takes the funnel label's mirrored seat: between bus and arm,
    # right-aligned just left of the riser
    fig.text(
        brace_xmr - 0.012, 0.5 * (y_bus_c + y_arm_r),
        "Our reproduction on a 64 GB RAM / RTX 5060 Ti (16 GB) home PC"
        " (8 open-data studies)",
        ha="right", va="center", fontsize=7.4, fontweight="bold",
        color=rep_c
    )

    if verify_overlaps:
        bad = check_figure_overlaps(fig)
        if bad:
            plt.close(fig)
            msg = "\n".join(f"  {pct:5.1f}%: {a!r} <-> {b!r}" for a, b, pct in bad)
            raise RuntimeError(f"Overlap check failed:\n{msg}")

    fig.savefig(out_stem + ".png", bbox_inches="tight")
    fig.savefig(out_stem + ".pdf", bbox_inches="tight")
    plt.close(fig)
    return len(rows_b), len(per)


def main():
    """Parse CLI arguments and run ``render_figure2``.

    Supports overriding the benchmark YAML, session JSON, output stem,
    and the text-overlap check via command-line flags.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--yaml", default=os.path.join(here, "reproduction_benchmark.yaml"),
                    help="Path to reproduction_benchmark.yaml")
    ap.add_argument("--session-json", default=os.path.join(here, "session_per_vs_wer.json"),
                    help="Path to session_per_vs_wer.json")
    ap.add_argument("--out",
                    default=os.path.join(here, "communication_neuroprostheses_reproduction"),
                    help="Output stem for .png and .pdf")
    ap.add_argument("--check-overlaps", action="store_true", default=True,
                    help="Run bounding-box overlap check before saving (default: True)")
    args = ap.parse_args()

    n_open, n_pillars = render_figure2(
        yaml_path=args.yaml,
        session_json_path=args.session_json,
        out_stem=args.out,
        verify_overlaps=args.check_overlaps,
    )
    print(f"[ok] Rendered Figure 2 ({n_open} open-data pillars, {n_pillars} reproduction pairs) -> {args.out}.{{png,pdf}}")
    print("     Overlap check: ALL CLEAN")


if __name__ == "__main__":
    main()
