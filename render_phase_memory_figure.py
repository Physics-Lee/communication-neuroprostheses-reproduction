#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone 4-panel companion figure: error-rate phase diagrams + the
memory wall.

Panels
------
(a) Session-level greedy PER vs. post-LM WER across the phoneme-level
    speech BCIs (the phase diagram retired from Figure 2 in the
    2026-11-11 phase-diagram round; reads ``session_per_vs_wer.json``).
(b) Session-level raw CER (no LM) vs. post-LM WER across the four
    character-level arms we reproduced (Willett 2021 handwriting bigram,
    Fan 2023 CORP recalibration, Jude 2026a typing T17/T18; reads
    ``session_cer_vs_wer.json`` built by ``build_session_cer_vs_wer.py``).
    The no-LM baseline uses the pooled reference characters-per-word of
    the plotted corpora (spaces included, matching the CER denominators).
(c) Peak host RAM per study: our 64 GB consumer PC stack vs. the
    official / naive stacks (moved out of Figure 2 in the 2026-11-11
    split).
(d) Peak GPU VRAM per study: our 16 GB single GPU vs. the official rigs
    (moved out of Figure 2 in the 2026-11-11 split).

2026-11-11 split round (user: "fig2 is too long to publish -- move RAM
and VRAM out of fig2 into a new 4-panel figure with WER vs PER and WER
vs CER"): panels (c)/(d) moved here from ``render_figure2.py``;
``reproduction_benchmark.yaml`` stays the single source of truth for all
four panels.

Usage:
    python render_phase_memory_figure.py

``build_phase_memory_svg`` additionally serialises the same figure as a
self-contained inline SVG consumed by ``render_figure2.export_interactive_html``
(the interactive HTML shows it in its own "Insights" section, capped at the
figure's natural 1210 px CSS width — the ``.pdf``/``.png`` outputs are
identical either way).
"""
import io
import json
import os
import re

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D

from render_figure2 import C_NORM, C_SPEECH_I, OPEN_C, check_figure_overlaps, to_rgb

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_STEM = os.path.join(HERE, "communication_neuroprostheses_phase_memory")


def _draw_per_panel(ax, session_path):
    """Draw panel (a): session-level PER vs. post-LM WER phase diagram.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes (equal aspect is applied here).
    session_path : str
        Path to ``session_per_vs_wer.json``.

    Returns
    -------
    None
    """
    with open(session_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    style = {
        "Willett 2023 (T12)":   ("#3C5488", "o", "Willett et al., 2023 (T12)"),
        "Card 2024 (T15)":      ("#4DBBD5", "s", "Card et al., 2024 (T15)"),
        "Jude 2026 (LIS, T17)": ("#1B6CA8", "D", "Jude et al., 2026b (T17)"),
    }
    lim = 80
    x_curve = np.linspace(0, 41.0, 200)
    y_curve = 100.0 * (1.0 - (1.0 - x_curve / 100.0) ** 3.4)
    ax.plot(x_curve, y_curve, ls="-.", lw=1.3, color="#888888", zorder=1.2)
    ax.text(1.5, 82.5, "No-LM baseline:\n1 \u2212 (1 \u2212 PER)$^{3.4}$",
            fontsize=8.0, color="#666666", ha="left", va="top",
            linespacing=1.12)
    ax.plot([0, lim], [0, lim], ls="--", lw=1.3, color="#999999", zorder=1)
    ax.axvline(25.0, color=C_NORM, lw=1.1, linestyle=(0, (4, 2)), zorder=1.5)
    ax.text(26.4, 24.0,
            "PER = 25%\n"
            "(\u226425%: 45/57 = 79% WER < PER;\n"
            " >25%: 4/20 = 20% WER < PER)\n"
            "This means that we must get PER < 25%.",
            fontsize=7.6, color="#555555", ha="left", va="top",
            linespacing=1.1)
    ax.text(58.5, 54.0, "WER = PER", fontsize=8.4, color="#777777",
            ha="left", va="top", rotation=45, rotation_mode="anchor")
    for name, v in data.items():
        if name not in style:
            continue
        c, m, lab = style[name]
        ax.scatter([s["per"] for s in v["sessions"]],
                   [s["wer"] for s in v["sessions"]],
                   s=34, marker=m, facecolor="none", edgecolor=c,
                   lw=1.5, alpha=0.85, zorder=3, label=lab)
        ax.scatter([v["pooled"]["per"]], [v["pooled"]["wer"]], s=120,
                   marker="*", color=c, edgecolor="white", lw=0.6, zorder=4)
    ax.legend(fontsize=8.2, frameon=False, loc="lower right",
              title="per-session (open)   pooled (star)",
              title_fontsize=8.2, borderaxespad=0.3, labelspacing=0.4)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim + 3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("greedy phoneme error rate per session (%)")
    ax.set_ylabel("word error rate after LM stage (%)")
    ax.set_title("(a) Session-level PER vs. post-LM WER",
                 fontsize=11.5, fontweight="bold")


def _draw_cer_panel(ax, session_path):
    """Draw panel (b): session-level raw CER vs. post-LM WER phase diagram.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes (equal aspect is applied here).
    session_path : str
        Path to ``session_cer_vs_wer.json``.

    Returns
    -------
    None
    """
    with open(session_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    meta = data.pop("_meta")
    chars_per_word = meta["chars_per_word_pooled"]
    style = {
        "Willett 2021 (T5)":  ("#3C5488", "o", "Willett et al., 2021 (T5, bigram)"),
        "Fan 2023 (T5)":      ("#4DBBD5", "s", "Fan et al., 2023 (T5)"),
        "Jude 2026a (T17)":   ("#1B6CA8", "D", "Jude et al., 2026a (T17)"),
        "Jude 2026a (T18)":   ("#7B9ACC", "^", "Jude et al., 2026a (T18)"),
    }
    lim = 40
    x_curve = np.linspace(0, 100.0 * (1.0 - (0.001) ** (1.0 / chars_per_word)), 200)
    y_curve = 100.0 * (1.0 - (1.0 - x_curve / 100.0) ** chars_per_word)
    ax.plot(x_curve, y_curve, ls="-.", lw=1.3, color="#888888", zorder=1.2)
    # The steep 5.1-exponent curve exits the top edge near x=9, so the
    # curve label tucks against its left side.
    ax.text(8.3, 39.6,
            f"no-LM:\n1 \u2212 (1 \u2212 CER)$^{{{chars_per_word:.1f}}}$",
            fontsize=7.8, color="#666666", ha="right", va="top",
            linespacing=1.12)
    ax.plot([0, lim], [0, lim], ls="--", lw=1.3, color="#999999", zorder=1)
    ax.text(30.0, 27.0, "WER = CER", fontsize=8.4, color="#777777",
            ha="left", va="top", rotation=45, rotation_mode="anchor")
    # 2026-11-11 removal round (user: "把这段话删了，太让人迷惑了"):
    # the three-line >=5-gram / bigram / no-LM crossover annotation is
    # retired; the diagonal WER = CER line and the no-LM curve label
    # carry the reading on their own.
    for name, v in data.items():
        if name not in style:
            continue
        c, m, lab = style[name]
        ax.scatter([s["cer"] for s in v["sessions"]],
                   [s["wer"] for s in v["sessions"]],
                   s=34, marker=m, facecolor="none", edgecolor=c,
                   lw=1.5, alpha=0.85, zorder=3, label=lab)
        ax.scatter([v["pooled"]["cer"]], [v["pooled"]["wer"]], s=120,
                   marker="*", color=c, edgecolor="white", lw=0.6, zorder=4)
    ax.legend(fontsize=8.2, frameon=False, loc="lower right",
              title="per-session (open)   pooled (star)",
              title_fontsize=8.2, borderaxespad=0.3, labelspacing=0.4)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim + 2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("raw character error rate per session (%)")
    ax.set_ylabel("word error rate after LM stage (%)")
    ax.set_title("(b) Session-level CER vs. post-LM WER",
                 fontsize=11.5, fontweight="bold")


def _draw_memory_panel(ax, rows, ceiling, title, ylabel, y_max, y_ticks,
                       label_pad, note_pad):
    """Draw panels (c)/(d): ours-vs-official memory bar pairs.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    rows : list of dict
        ``panel_g_host_ram_walls`` / ``panel_h_gpu_vram_walls`` rows with
        ``category``, ``ours_gb``, ``official_gb``, and ``note`` keys.
    ceiling : float
        Consumer-PC ceiling drawn as a dashed line (64 RAM / 16 VRAM).
    title, ylabel : str
        Panel title and y-axis label.
    y_max : float
        Upper y limit.
    y_ticks : list of int
        Y tick positions.
    label_pad, note_pad : float
        Data-coordinate offsets for the value labels and grey notes.

    Returns
    -------
    None
    """
    c_off = OPEN_C[2]
    fc_ours = (*to_rgb(C_SPEECH_I), 0.35)
    fc_off = (*to_rgb(c_off), 0.35)
    x = np.arange(len(rows))
    w = 0.36
    ax.bar(x - w / 2, [r["ours_gb"] for r in rows], w,
           color=fc_ours, edgecolor=C_SPEECH_I, linewidth=0.9)
    ax.bar(x + w / 2, [r["official_gb"] for r in rows], w,
           color=fc_off, edgecolor=c_off, linewidth=0.9)
    for i, r in enumerate(rows):
        o_gb, p_gb = r["ours_gb"], r["official_gb"]
        o_str = f"{o_gb:.0f}" if o_gb == int(o_gb) else f"{o_gb:.1f}"
        p_str = f"{p_gb:.0f}" if p_gb == int(p_gb) else f"{p_gb:.1f}"
        ax.text(i - w / 2 - 0.02, o_gb + label_pad, o_str,
                ha="center", va="bottom", fontsize=6.8)
        ax.text(i + w / 2 + 0.02, p_gb + label_pad, p_str,
                ha="center", va="bottom", fontsize=6.8)
        ax.text(i, max(o_gb, p_gb) + note_pad, r["note"],
                ha="center", va="bottom", fontsize=6.4,
                color="#555555", linespacing="normal")
    ax.set_ylim(0, y_max)
    ax.set_yticks(y_ticks)
    ax.axhline(ceiling, color=C_NORM, linewidth=1.1,
               linestyle=(0, (4, 2)), zorder=2.6)
    ax.set_xticks(x)
    ax.set_xticklabels([r["category"] for r in rows], fontsize=7.4,
                       rotation=75, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.65, len(rows) - 0.35)
    ax.set_ylabel(ylabel)
    ax.legend(handles=[
        mpatches.Patch(facecolor=fc_ours, edgecolor=C_SPEECH_I, linewidth=0.9,
                       label="our reproduction (64 GB RAM PC)"),
        mpatches.Patch(facecolor=fc_off, edgecolor=c_off, linewidth=0.9,
                       label="official / naive stack"),
        Line2D([0], [0], color=C_NORM, linewidth=1.1, linestyle=(0, (4, 2)),
               label=f"{ceiling:.0f} GB consumer PC "
                     f"{'RAM' if ceiling > 32 else 'GPU VRAM'} ceiling"),
    ], frameon=False, fontsize=7.6, loc="upper left", ncol=1,
        columnspacing=1.0, handlelength=1.4, handletextpad=0.5)
    ax.set_title(title, fontsize=11.0, fontweight="bold")


def _build_figure():
    """Assemble the live 4-panel Matplotlib figure.

    Returns
    -------
    matplotlib.figure.Figure
        Unchecked (no overlap verification yet) figure instance; the
        caller is responsible for ``check_figure_overlaps`` and closing.
    """
    with open(os.path.join(HERE, "reproduction_benchmark.yaml"), "r",
              encoding="utf-8") as f:
        data = yaml.safe_load(f)

    fig = plt.figure(figsize=(12.6, 12.2))
    gs = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.86), hspace=0.34,
                          wspace=0.24)

    ax_a = fig.add_subplot(gs[0, 0])
    _draw_per_panel(ax_a, os.path.join(HERE, "session_per_vs_wer.json"))
    ax_b = fig.add_subplot(gs[0, 1])
    _draw_cer_panel(ax_b, os.path.join(HERE, "session_cer_vs_wer.json"))
    ax_c = fig.add_subplot(gs[1, 0])
    _draw_memory_panel(
        ax_c, data["panel_g_host_ram_walls"], 64.0,
        "(c) Host RAM across all 8 studies: our 64 GB PC vs. official stacks",
        "Peak host RAM (GB)", 445, list(range(0, 401, 100)), 4.5, 23.0)
    ax_d = fig.add_subplot(gs[1, 1])
    _draw_memory_panel(
        ax_d, data["panel_h_gpu_vram_walls"], 16.0,
        "(d) GPU VRAM across all 8 studies: our 16 GB GPU vs. official rigs",
        "Peak GPU VRAM (GB)", 118.0, list(range(0, 101, 25)), 1.3, 6.8)

    fig.subplots_adjust(left=0.075, right=0.985, top=0.965, bottom=0.09)
    return fig


def build_phase_memory_svg():
    """Render the 4-panel companion figure as a self-contained inline SVG.

    2026-11-12 HTML rounds (user: "把 phase_memory 图加入 html 的 Overview
    之后"; then "算了，还是新开一个 section 吧，就叫 Insights"; "四个 panel
    在网页中打开看上去很大，和其他图片不太适应，在 html 中处理一下"):
    the interactive HTML shows this figure in its own "Insights" section,
    so the exporter needs an in-process SVG serialisation that does not
    depend on the ``.png``/``.pdf`` files (CI runs ``render_figure2.py``
    before this script's ``main()``); page CSS caps the rendered width.

    Returns
    -------
    tuple of (str, int, int)
        ``(svg_element, vw, vh)`` where ``svg_element`` is a complete
        ``<svg id="pm-svg" viewBox="...">...</svg>`` element (fixed
        ``width``/``height`` stripped so page CSS scales it, Matplotlib's
        internal ``<style>`` block scoped under ``#pm-svg`` so its
        universal selectors cannot leak into the page-wide CSS).
    """
    fig = _build_figure()
    bad = check_figure_overlaps(fig)
    if bad:
        plt.close(fig)
        msg = "\n".join(f"  {pct:5.1f}%: {a!r} <-> {b!r}" for a, b, pct in bad)
        raise RuntimeError(f"Overlap check failed:\n{msg}")
    buf = io.StringIO()
    fig.savefig(buf, format="svg", bbox_inches="tight")
    plt.close(fig)
    svg = buf.getvalue()

    # HTML cannot host the <?xml?> declaration; the generator comment
    # would render as a bogus comment, so both headers are dropped.
    svg = re.sub(r"^<\?xml[^>]*\?>\s*", "", svg)
    svg = re.sub(r"^<!--.*?-->\s*", "", svg, flags=re.DOTALL)

    # Scope Matplotlib's <style> block (e.g. ``*{stroke-linejoin: round}``)
    # under #pm-svg: a bare universal selector inside an inline SVG would
    # apply document-wide and silently restyle the hand-built f2/f6 SVGs.
    def _scope_style(match):
        rules = []
        for rule in match.group(1).split("}"):
            if "{" not in rule:
                continue
            sel, body = rule.split("{", 1)
            scoped = ", ".join(
                f"#pm-svg {s.strip()}" for s in sel.split(",") if s.strip()
            )
            rules.append(f"{scoped}{{{body.strip()}}}")
        return ('<style type="text/css">' + " ".join(rules) + "</style>")

    svg = re.sub(r'<style type="text/css">\s*(.*?)\s*</style>',
                 _scope_style, svg, count=1, flags=re.DOTALL)

    # Drop the fixed pt width/height on the root tag only and stamp the id
    # the page CSS (#pm-svg { width: 100%; height: auto; }) targets.
    root = re.search(r"<svg\b[^>]*>", svg)
    if root is None:
        raise RuntimeError("Matplotlib SVG output has no root <svg> tag")
    root_tag = root.group(0)
    vw_m = re.search(r'viewBox="([\d.]+) ([\d.]+)', root_tag)
    vw, vh = (int(round(float(g))) for g in vw_m.groups()) if vw_m else (0, 0)
    slim = re.sub(r'\s+(?:width|height)="[^"]*"', "", root_tag)
    slim = slim.replace("<svg", '<svg id="pm-svg"', 1)
    svg = svg.replace(root_tag, slim, 1)
    return svg, vw, vh


def main():
    """Render the 4-panel figure and save PNG + PDF.

    Returns
    -------
    None
        Writes ``communication_neuroprostheses_phase_memory.{png,pdf}``.
    """
    fig = _build_figure()
    bad = check_figure_overlaps(fig)
    if bad:
        plt.close(fig)
        msg = "\n".join(f"  {pct:5.1f}%: {a!r} <-> {b!r}" for a, b, pct in bad)
        raise RuntimeError(f"Overlap check failed:\n{msg}")
    fig.savefig(OUT_STEM + ".png", bbox_inches="tight")
    fig.savefig(OUT_STEM + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUT_STEM}.png / .pdf")


if __name__ == "__main__":
    main()
