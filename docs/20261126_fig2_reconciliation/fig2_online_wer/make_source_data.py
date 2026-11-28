# -*- coding: utf-8 -*-
"""make_source_data.py -- Nature-style Source Data xlsx for the 8 main figures.

User request (2026-11-12): the main figures carry too much printed text; the
bar values and error bars must move to a ``source_data/`` package (template:
``reference/source_data_of_Jude26`` -- one xlsx per figure, one sheet per
panel).  A second agent will then delete from the figures the numbers this
package records.

Method -- render-time interception, NOT transcription:
  * ``fig_common.save_fig`` is patched BEFORE the ``main_fig_N`` modules are
    imported (they bind it via ``from fig_common import *``), so the real
    render functions run against a recorder that writes NO figure files;
  * ``Axes.bar / text / errorbar / annotate / axhline / axvline / axhspan``
    are wrapped to record every call's arguments verbatim;
  * at save time each axes contributes: title, numeric xticks + tick labels,
    its recorded calls, and unmatched in-axes texts.
Every number below therefore IS the number drawn on the figure (same code
path), and every ``value_display`` string IS the exact string printed on it.

Scope: Figures 2-7 (all panels with bars; Figure 3a/b scatter tables are read
from the sister repo JSONs; per-figure ``annotations`` sheets inventory every
remaining in-axes text).  Figures 1 (timeline) and 8 (flowcharts + tier maps)
contain no bar-chart numerics and get no file -- see README.

Run from figures/:  python make_source_data.py
Outputs: ../latex/source_data/SourceData_Figure2..7.xlsx (+ README.md is
written by a separate DLP-safe step, NOT by this script).
"""
import importlib
import json
import os
import re
import sys

import openpyxl
from openpyxl.styles import Alignment, Font

FIG_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(FIG_DIR)                       # figure modules must run from figures/
sys.path.insert(0, FIG_DIR)

import matplotlib                       # noqa: E402
matplotlib.use("Agg")
from matplotlib.axes import Axes        # noqa: E402

REPRO_DIR = os.path.normpath(os.path.join(
    FIG_DIR, "..", "..", "20261106_communication_neuroprostheses_reproduction"))
OUT_DIR = os.path.normpath(os.path.join(FIG_DIR, "..", "latex", "source_data"))

# ---------------------------------------------------------------------------
# 1. render-time interception
# ---------------------------------------------------------------------------
_REC = []          # dicts: method, ax, args, kwargs (args hold plain values)


def _patch_axes_method(name):
    orig = getattr(Axes, name)

    def hooked(self, *args, **kwargs):
        _REC.append({"method": name, "ax": self,
                     "args": args, "kwargs": dict(kwargs)})
        return orig(self, *args, **kwargs)

    setattr(Axes, name, hooked)


for _m in ("bar", "text", "errorbar", "annotate", "axhline", "axvline",
           "axhspan"):
    _patch_axes_method(_m)

import fig_common                      # noqa: E402  (before main_fig_*!)

CAPTURES = {}      # stem -> {"panels": {letter: axes-data}, "fig_texts": [...]}


def _panel_letter(title):
    m = re.match(r"^\(([a-z])\)", title.strip())
    return m.group(1) if m else None


def _fake_save_fig(fig, stem):
    panels, lettered = {}, []          # letter -> axes data
    for ax in fig.axes:
        title = ax.get_title()
        data = {
            "title": title,
            "pos": ax.get_position(),
            "xticks": [float(t) for t in ax.get_xticks()],
            "xticklabels": [t.get_text()
                            for t in ax.get_xticklabels()],
            "calls": [r for r in _REC if r["ax"] is ax],
        }
        letter = _panel_letter(title)
        if letter:
            panels.setdefault(letter, []).append(data)
            lettered.append(data)
        else:
            # unlettered axes join the lettered panel whose x-range overlaps
            # most (e.g. Figure 7's tone-curve axes and bar axes both live
            # under the one "(c)" block title)
            host = None
            for cand in lettered:
                if (data["pos"].x1 > cand["pos"].x0
                        and data["pos"].x0 < cand["pos"].x1):
                    host = cand
                    break
            if host is not None:
                panels[_panel_letter(host["title"])].append(data)
            else:
                panels.setdefault("~", []).append(data)
    CAPTURES[stem] = {
        "panels": panels,
        "fig_texts": [(t.get_text(), t.get_position()) for t in fig.texts],
    }


fig_common.save_fig = _fake_save_fig


def _run_figure(n):
    _REC.clear()
    mod = importlib.import_module(f"main_fig_{n}")
    getattr(mod, f"main_fig_{n}")()
    return CAPTURES[f"main_fig_{n}"]


# ---------------------------------------------------------------------------
# 2. small helpers over one panel's recorded calls
# ---------------------------------------------------------------------------
class Panel:
    """All recorded calls of one panel letter (merges its axes)."""

    def __init__(self, cap, letter):
        self.letter = letter
        self.axes_list = cap["panels"][letter]
        self.calls = [c for axd in self.axes_list for c in axd["calls"]]
        # primary axes = the one with bars if any, else the first
        bar_axes = [axd for axd in self.axes_list
                    if any(c["method"] == "bar" for c in axd["calls"])]
        self.ax = (bar_axes or self.axes_list)[0]
        self.ticks = list(zip(self.ax["xticks"], self.ax["xticklabels"]))

    def bars(self):
        """[(x, height, bottom, kwargs), ...] in draw order."""
        out = []
        for c in self.calls:
            if c["method"] != "bar":
                continue
            x, h = c["args"][0], c["args"][1]
            kw = c["kwargs"]
            bottom = float(kw.get("bottom", 0.0) or 0.0)
            xs = list(x) if hasattr(x, "__iter__") else [x]
            hs = list(h) if hasattr(h, "__iter__") else [h]
            for xi, hi in zip(xs, hs):
                out.append((float(xi), float(hi), bottom, kw))
        return out

    def errorbars(self):
        """[(x, y, yerr_low, yerr_high, kwargs), ...] from ax.errorbar calls."""
        out = []
        for c in self.calls:
            if c["method"] != "errorbar":
                continue
            x, y = c["args"][0], c["args"][1]
            yerr = c["args"][2] if len(c["args"]) > 2 else None
            yerr = c["kwargs"].get("yerr", yerr)
            lo = hi = None
            if isinstance(yerr, (list, tuple)) and yerr and isinstance(
                    yerr[0], (list, tuple)):
                lo, hi = float(yerr[0][0]), float(yerr[1][0])
            elif yerr is not None:
                lo = hi = float(yerr)
            out.append((float(x), float(y), lo, hi, c["kwargs"]))
        return out

    def bar_yerr(self):
        """{x: (low, high)} from bar(..., yerr=...) calls (Figure 4c)."""
        out = {}
        for c in self.calls:
            if c["method"] == "bar" and c["kwargs"].get("yerr") is not None:
                x = float(c["args"][0])
                yerr = c["kwargs"]["yerr"]
                out[x] = (float(yerr), float(yerr))
        return out

    def texts(self, include_transformed=False):
        """[(x, y, s), ...] from ax.text calls in DATA coordinates by
        default (axes-transform texts like the CJK subtitles and
        transAxes corner notes are excluded unless asked for)."""
        out = []
        for c in self.calls:
            if c["method"] == "text":
                if "transform" in c["kwargs"] and not include_transformed:
                    continue
                out.append((float(c["args"][0]), float(c["args"][1]),
                            c["args"][2]))
            elif c["method"] == "annotate" and c["args"]:
                s = c["args"][0]
                if s:
                    xy = c["kwargs"].get("xy", (0, 0))
                    out.append((float(xy[0]), float(xy[1]), s))
        return out

    def hlines(self):
        return [float(c["args"][0]) for c in self.calls
                if c["method"] == "axhline"]

    def hspans(self):
        return [(float(c["args"][0]), float(c["args"][1]))
                for c in self.calls if c["method"] == "axhspan"]

    def category(self, x):
        """Tick label of the nearest numeric tick (empty when far)."""
        if not self.ticks:
            return ""
        best = min(self.ticks, key=lambda t: abs(t[0] - x))
        return best[1].replace("\n", " ") if abs(best[0] - x) < 1.0 else ""


def nearest_text_above(texts, x, value, used, max_ratio=8.0):
    """The closest text above a bar top at ~same x (value_display matcher).
    x-distance is weighted heavily so a neighbour bar's label can never
    win over a bar's own (staggered) label."""
    best, best_score = None, None
    for i, (tx, ty, s) in enumerate(texts):
        if i in used:
            continue
        dx = abs(tx - x)
        if value is not None and value > 0:
            if ty < value * 0.8 or ty > value * max_ratio:
                continue
            score = (ty - value) / value + 10.0 * dx
        else:
            score = ty + 10.0 * dx
        if best_score is None or score < best_score:
            best, best_score = i, score
    return best


def cluster_bars(bars, gap=0.6):
    """Group bars into x-clusters (a bar pair spans 0.36-0.66; adjacent
    studies sit >= 0.95 apart in every paired panel)."""
    bs = sorted(bars, key=lambda b: b[0])
    clusters = []
    for b in bs:
        if clusters and abs(b[0] - clusters[-1][-1][0]) < gap:
            clusters[-1].append(b)
        else:
            clusters.append([b])
    return clusters


def pair_cluster(p, series_names, text_lower_idx, extra_top=None,
                 note_of=None):
    """Grouped-pair panels: gap-cluster bars; assign the two stacked centre
    texts (lower->text_lower_idx series, upper->the other series).

    ``series_names``: [left-bar series, right-bar series] order.
    Returns rows (bar order: left pair row, right pair row per cluster).
    """
    bars, texts = p.bars(), p.texts()
    rows = []
    for members in cluster_bars(bars):
        members.sort(key=lambda m: m[0])
        (xl, hl, _, _), (xr, hr, _, _) = members[0], members[-1]
        cat = p.category(0.5 * (xl + xr))
        cands = sorted(
            [(i, t) for i, t in enumerate(texts)
             if abs(t[0] - 0.5 * (xl + xr)) < 0.45],
            key=lambda it: it[1][1])
        disp = ["", ""]
        for slot, (ti, t) in enumerate(cands[:2]):
            disp[slot] = t[2]
        note = note_of(members) if note_of else ""
        rows.append(dict(kind="bar", x=round(xl, 3), category=cat,
                         series=series_names[0], value=hl,
                         value_display=disp[text_lower_idx], notes=note))
        rows.append(dict(kind="bar", x=round(xr, 3), category=cat,
                         series=series_names[1], value=hr,
                         value_display=disp[1 - text_lower_idx], notes=note))
        if extra_top and len(cands) > 2:      # e.g. "-22.8%" reduction tag
            rel = cands[2][1][2]
            for r in rows[-2:]:
                r["notes"] = (r["notes"] + "; " if r["notes"] else "") + \
                    f"third label above pair: {rel}"
    return rows


# ---------------------------------------------------------------------------
# 3. per-figure sheet builders
#     Every builder returns {sheet_name: [header, *rows]} where a row is a
#     dict keyed by header.  Numbers always come from the capture / JSON /
#     YAML -- never retyped literals.
# ---------------------------------------------------------------------------
HEAD = ["kind", "x", "category", "series", "value", "error_low",
        "error_high", "value_display", "notes"]

OPEN_LABEL = {2: "open", 1: "on request", 0: "closed", "P": "preprint",
              "R": "reanalysis on open data"}

_wb_cache = {}


def _row(**kw):
    base = dict(kind="bar", x="", category="", series="", value="",
                error_low="", error_high="", value_display="", notes="")
    base.update(kw)
    if isinstance(base["notes"], str):
        # keep cells single-line (value_display stays verbatim)
        base["notes"] = base["notes"].replace("\n", "; ")
    return base


# ---------- Figure 2 --------------------------------------------------------
def build_fig2():
    from main_fig_2 import _load_figure2_data

    bench = _load_figure2_data()
    cap = CAPTURES["main_fig_2"]
    sheets = {}

    # (a) vocabulary funnel, 32 bars; equal-height runs share one label
    p = Panel(cap, "a")
    rows_a = bench["panel_a_vocab_funnel"]
    bars = p.bars()
    assert len(bars) == len(rows_a) == 32
    texts = p.texts()
    used = set()
    rows = []
    # runs of equal consecutive heights share the group label at run centre
    runs, start = [], 0
    for i in range(1, 33):
        if i == 32 or bars[i][1] != bars[start][1]:
            runs.append((start, i - 1))
            start = i
    for lo, hi in runs:
        v = bars[lo][1]
        cx = 0.5 * (lo + hi)
        best, best_d = None, None
        for ti, (tx, ty, s) in enumerate(texts):
            if ti in used or tx < lo - 0.6 or tx > hi + 0.6:
                continue
            if not (v * 1.1 < ty < v * 9.0):
                continue
            d = abs(tx - cx)
            if best_d is None or d < best_d:
                best, best_d = ti, d
        disp = texts[best][2] if best is not None else ""
        if best is not None:
            used.add(best)
        for i in range(lo, hi + 1):
            r = rows_a[i]
            rows.append(_row(
                x=i, category=p.category(i), value=r["vocab_num"],
                value_display=disp if hi > lo else disp,
                notes="open-vocab definition (dashed bar)"
                if r["is_open_vocab"] else
                f"data {OPEN_LABEL[r['openness_code']]}"))
    # Brysbaert anchors (dashed hlines with boxed labels)
    anchor_lab = {}
    for ti, (tx, ty, s) in enumerate(texts):
        if ti in used and False:
            continue
        for av, alab in ((11100, "11,100 word families"),
                         (42000, "42,000 lemmas"),
                         (71400, "71,400 word forms")):
            if s == alab:
                anchor_lab[av] = alab
    for av in (11100, 42000, 71400):
        rows.append(_row(kind="ref", value=av,
                         value_display=anchor_lab.get(av, ""),
                         notes="dashed reference line (Brysbaert 2016)"))
    sheets["a_vocab_32_studies"] = rows

    # (b) open dataset size
    # 2026-11-13 number-diet round: panels (b)-(f) print no per-bar
    # values, so their bar rows keep value_display empty -- matching the
    # nearest in-axes text here would misattribute reference-line labels
    # (e.g. "2.5% LibriSpeech") to bars.  Panel (a) still prints its
    # group labels and keeps the run-centre text matching above.
    p = Panel(cap, "b")
    rows_b = bench["panel_b_dataset_size"]
    bars = p.bars()
    assert len(bars) == len(rows_b) == 11
    rows = []
    for i, ((x, h, _, _), r) in enumerate(zip(bars, rows_b)):
        assert abs(h - r["released_sentences"]) < 1e-6
        rows.append(_row(
            x=i, category=p.category(i), value=h,
            notes=(r.get("span_note") or "") + "; "
            f"data {OPEN_LABEL[r['openness_code']]}"))
    sheets["b_open_dataset_size"] = rows

    # (c) online WER + SD error bars + ASR reference lines
    p = Panel(cap, "c")
    rows_cd = bench["panel_cd_online_wer_speed"]
    bars = p.bars()
    errs = {x: (lo, hi) for x, y, lo, hi, _ in p.errorbars()}
    asr_lab = {2.5: "2.5% LibriSpeech", 5.8: "5.8% Switchboard",
               12.8: "12.8% 14-set avg."}
    rows = []
    for i, ((x, h, _, _), r) in enumerate(zip(bars, rows_cd)):
        assert abs(h - r["wer_mean"]) < 1e-6
        lo, hi = errs.get(x, ("", ""))
        note = r["wer_caliber"]
        if r["wer_sd"] is not None:
            note = f"SD across sessions; {note}"
        rows.append(_row(
            x=i, category=p.category(i), value=h,
            error_low=lo, error_high=hi, notes=note))
    for v in (2.5, 5.8, 12.8):
        rows.append(_row(kind="ref", value=v, value_display=asr_lab[v],
                         notes="dashed ASR reference"))
    sheets["c_online_WER"] = rows

    # (d) online speed (log axis; n/r bars have no bar patch)
    p = Panel(cap, "d")
    bars = p.bars()
    errs = {x: (lo, hi) for x, y, lo, hi, _ in p.errorbars()}
    texts = p.texts()
    rows = []
    for i, r in enumerate(rows_cd):
        match = [b for b in bars if abs(b[0] - i) < 0.1]
        spd, sd = r["speed_wpm_mean"], r["speed_wpm_sd"]
        if match:
            x, h, _, _ = match[0]
            assert abs(h - spd) < 1e-6
            lo, hi = errs.get(x, ("", ""))
            rows.append(_row(
                x=i, category=p.category(i), value=h, error_low=lo,
                error_high=hi,
                notes=(r.get("speed_caliber") or "")))
        else:
            nr = [s for (tx, ty, s) in texts
                  if abs(tx - i) < 0.45 and s == "n/r"]
            rows.append(_row(
                x=i, category=p.category(i), value="",
                value_display=nr[0] if nr else "n/r",
                notes="no quantitative online-speed number reported"))
    for v, lab in ((160, "160 conversation"), (51.6, "51.6 keyboard"),
                   (25.1, "25.1 handwriting"), (8, "8 p&c BCI")):
        rows.append(_row(kind="ref", value=v, value_display=lab,
                         notes="dashed able-bodied reference (wpm)"))
    sheets["d_online_speed"] = rows

    # (e)/(f) reproduction pairs: ours left + paper right (no printed
    # value labels since the 2026-11-13 number-diet round; "n/r" rows
    # keep their token because the figure still prints it)
    online_by_label = {r["label"]: r for r in rows_cd}
    wer_by_label = {r["label"]: r for r in bench["panel_f_repro_full_wer"]}
    for letter, rows_yaml, sheet in (
            ("e", bench["panel_e_repro_acoustic"], "e_repro_noLM"),
            ("f", [wer_by_label[r["label"]] for r in rows_cd],
             "f_repro_fullWER")):
        p = Panel(cap, letter)
        bars = p.bars()
        errs = {round(x, 3): (lo, hi) for x, y, lo, hi, _ in p.errorbars()}
        clusters = cluster_bars(bars)
        assert len(clusters) == len(rows_yaml)
        rows = []
        for members in clusters:
            members.sort(key=lambda m: m[0])
            xc = 0.5 * (members[0][0] + members[-1][0])
            i = min(range(len(p.ticks)), key=lambda j: abs(p.ticks[j][0] - xc))
            r = rows_yaml[i]  # panel (e) groups bars by metric
            ours_note = r.get("note") or ""
            if letter == "e":
                ours_note = f"metric group: {r['metric']}; {ours_note}".rstrip("; ")
                paper_note = ours_note
                paper_series = "paper anchor"
            else:
                online = online_by_label[r["label"]]
                paper_note = f"Online WER from panel (c); {online['wer_caliber']}"
                paper_series = "online WER from panel (c)"
            cat = p.category(xc)
            if len(members) == 2:
                xo, vo, _, _ = members[0]
                rows.append(_row(
                    x=round(xo, 3), category=cat,
                    series="ours (16 GB GPU)", value=vo, notes=ours_note))
            else:
                xo = next(x for x, _, text in p.texts()
                          if text == "n/r" and p.category(x) == cat)
                rows.append(_row(
                    x=round(xo, 3), category=cat,
                    series="ours (16 GB GPU)", value="",
                    value_display="n/r", notes="no reproduction WER for the voice arm"))
            xp, vp, _, _ = members[-1]
            rows.append(_row(
                x=round(xp, 3), category=cat, series=paper_series,
                value=vp, notes=paper_note))
        # error bars per bar (lower whisker is clipped at 0 by the renderer
        # when it would cross the axis)
        for row in rows:
            if row["value"] == "":
                continue
            lohi = errs.get(row["x"])
            if lohi and lohi[0] is not None:
                row["error_low"], row["error_high"] = lohi
                if row["value"] < lohi[0]:
                    row["notes"] = (row["notes"] + "; "
                                    "lower whisker clipped at 0").strip("; ")
        sheets[sheet] = rows
    return sheets, cap


# ---------- Figure 3 --------------------------------------------------------
def build_fig3():
    import yaml
    with open(os.path.join(REPRO_DIR, "reproduction_benchmark.yaml"),
              "r", encoding="utf-8") as f:
        bench = yaml.safe_load(f)
    cap = CAPTURES["main_fig_3"]
    sheets = {}

    # (a)/(b) session-level scatter tables straight from the sister repo
    # (only the studies the renderer draws -- the JSON also carries the
    # Kunz inner-speech arms used by Figure 6g, not by these panels)
    for letter, fname, xkey, sheet, keep in (
            ("a", "session_per_vs_wer.json", "per", "a_session_PER_vs_WER",
             ("Willett 2023 (T12)", "Card 2024 (T15)",
              "Jude 2026 (LIS, T17)")),
            ("b", "session_cer_vs_wer.json", "cer", "b_session_CER_vs_WER",
             ("Willett 2021 (T5)", "Fan 2023 (T5)", "Jude 2026a (T17)",
              "Jude 2026a (T18)"))):
        with open(os.path.join(REPRO_DIR, fname), encoding="utf-8") as f:
            data = json.load(f)
        meta = data.pop("_meta", {})
        rows = []
        for study in keep:
            v = data[study]
            for s in v["sessions"]:
                rows.append({"study": study, "level": "session",
                             "session": s["session"], "n": s.get("n", ""),
                             f"x_{xkey} (%)": s[xkey],
                             "WER after LM (%)": s["wer"]})
            rows.append({"study": study, "level": "pooled (star)",
                         "session": "", "n": "",
                         f"x_{xkey} (%)": v["pooled"][xkey],
                         "WER after LM (%)": v["pooled"]["wer"]})
        sheets[sheet] = (rows, meta)

    # (c)/(d) memory walls: ours left / official right; texts
    # lower = official (teal), upper = ours (blue)
    for letter, key, sheet, ylabel, ceiling, ceil_lab in (
            ("c", "panel_g_host_ram_walls", "c_host_RAM",
             "Peak host RAM (GB)", 64.0, "64 GB RAM ceiling"),
            ("d", "panel_h_gpu_vram_walls", "d_GPU_VRAM",
             "Peak GPU VRAM (GB)", 16.0, "16 GB GPU ceiling")):
        p = Panel(cap, letter)
        rows_yaml = bench[key]
        clusters = cluster_bars(p.bars())
        assert len(clusters) == len(rows_yaml)
        rows = []
        for members, r in zip(clusters, rows_yaml):
            members.sort(key=lambda m: m[0])
            xc = 0.5 * (members[0][0] + members[-1][0])
            cat = p.category(xc)
            cands = sorted(
                [(ti, t) for ti, t in enumerate(p.texts())
                 if abs(t[0] - xc) < 0.45],
                key=lambda it: it[1][1])
            disp = [cands[0][1][2] if cands else "",
                    cands[1][1][2] if len(cands) > 1 else ""]
            note = f"{ylabel}: " + (r.get("note") or "")
            rows.append(_row(
                x=round(members[0][0], 3), category=cat,
                series="ours (64 GB PC)" if letter == "c"
                else "ours (16 GB GPU)",
                value=members[0][1], value_display=disp[1], notes=note))
            rows.append(_row(
                x=round(members[-1][0], 3), category=cat,
                series="official / naive",
                value=members[-1][1], value_display=disp[0], notes=note))
        rows.append(_row(kind="ref", value=ceiling,
                         value_display=ceil_lab,
                         notes="dashed ceiling of our hardware"))
        sheets[sheet] = rows
    return sheets, cap


# ---------- Figure 4 --------------------------------------------------------
def build_fig4():
    cap = CAPTURES["main_fig_4"]
    sheets = {}

    def single(letter, sheet, series="", note_of=None, yerr_in_bar=False):
        p = Panel(cap, letter)
        bars = p.bars()
        texts = p.texts()
        byerr = p.bar_yerr()
        rows = []
        used = set()
        for (x, h, _, kw) in bars:
            ti = nearest_text_above(texts, x, h, used)
            disp = texts[ti][2] if ti is not None else ""
            if ti is not None:
                used.add(ti)
            lo = hi = ""
            if yerr_in_bar and x in byerr:
                lo, hi = byerr[x]
            note = note_of(kw) if note_of else ""
            rows.append(_row(x=round(x, 3), category=p.category(x),
                             series=series, value=h, error_low=lo,
                             error_high=hi, value_display=disp, notes=note))
        sheets[sheet] = rows
        return p, texts, used

    # (a) daily calibration
    single("a", "a_daily_calibration")
    for r in sheets["a_daily_calibration"]:
        if "\u2013" in (r["value_display"] or ""):
            r["notes"] = (f"reported range {r['value_display']};"
                          " bar drawn at the upper bound")

    # (b) T12 online->offline
    single("b", "b_T12_online_offline")

    # (c) speech vs typing (yerr inside bar kwargs; in-bar n = notes)
    p = Panel(cap, "c")
    bars = p.bars()
    texts = p.texts()
    rows = []
    for (x, h, _, kw) in bars:
        ti = nearest_text_above(texts, x, h, set())
        n_note = [s for (tx, ty, s) in texts
                  if abs(tx - x) < 0.3 and s.startswith("n =")]
        rows.append(_row(
            x=round(x, 3), category=p.category(x), value=h,
            error_low=float(kw["yerr"]), error_high=float(kw["yerr"]),
            value_display=texts[ti][2] if ti is not None else "",
            notes=(n_note[0] + "; SD across sessions") if n_note
            else "SD across sessions"))
    sheets["c_speech_vs_typing"] = rows

    # (d) self-gen vs copy
    single("d", "d_selfgen_vs_copy")

    # (e) ECoG vs Utah (hatched first bar = ECoG B3)
    single("e", "e_ECoG_vs_Utah",
           note_of=lambda kw: "ECoG grid (hatched bar)"
           if kw.get("hatch") else "Utah array")

    # (f) Broca: 6v left / Area 44 right, texts share each bar's x
    p = Panel(cap, "f")
    bars = p.bars()
    texts = p.texts()
    rows = []
    for (x, h, _, kw) in bars:
        is_44 = kw.get("hatch") is not None
        cands = [(ti, t) for ti, t in enumerate(texts)
                 if abs(t[0] - x) < 0.25 and t[1] >= h]
        disp = min(cands, key=lambda it: it[1][1])[1][2] if cands else ""
        rows.append(_row(
            x=round(x, 3), category=p.category(x),
            series="Area 44 x2 (Broca's)" if is_44 else "6v x2 (decoded)",
            value=h, value_display=disp,
            notes="chance-level control (white/hatched bar)" if is_44 else ""))
    sheets["f_Broca_fails"] = rows

    # (g) PEF & hand knob: 3 bar styles interleaved
    p = Panel(cap, "g")
    bars = p.bars()
    texts = p.texts()
    rows = []
    for (x, h, _, kw) in bars:
        fc = kw.get("facecolor")
        if isinstance(fc, str) and fc.lower() == "white":
            series = "PEF / 6d (chance)"
        elif kw.get("color") is not None:
            col = kw["color"]
            alpha = col[3] if isinstance(col, (tuple, list)) and len(col) > 3 \
                else 1.0
            series = "55b (decoded)" if (not isinstance(alpha, float)
                                         or alpha < 0.4) else "6v / i6v (decoded)"
        else:
            series = "6v / i6v (decoded)"
        ti = nearest_text_above(texts, x, h, set())
        rows.append(_row(
            x=round(x, 3), category=p.category(x), series=series, value=h,
            value_display=texts[ti][2] if ti is not None else ""))
    sheets["g_PEF_handknob"] = rows

    # (h) channel dose T15 + chance ref
    p, texts, used = single("h", "h_channel_dose_T15")
    sheets["h_channel_dose_T15"].append(
        _row(kind="ref", value=80.2, value_display="chance (80.2%)",
             notes="chance line: 80.2%"))

    # (i) channel dose B3
    single("i", "i_channel_dose_B3")

    # (j) stacked channels: one row per segment + a total row per stack
    p = Panel(cap, "j")
    bars = p.bars()
    texts = p.texts()
    rows = []
    stacks = {}
    for (x, h, bottom, kw) in bars:
        stacks.setdefault(round(x, 3), []).append((bottom, h, kw))
    for x, members in stacks.items():
        members.sort(key=lambda m: m[0])
        cat = p.category(x)
        total = sum(m[1] for m in members)
        seg_names = []
        for (bottom, h, kw) in members:
            inside = [t for t in texts
                      if abs(t[0] - x) < 0.25 and abs(t[1] - (bottom + h / 2))
                      < h * 0.9]
            seg = inside[0][2] if inside else ""
            seg_names.append(seg)
            rows.append(_row(
                x=x, category=cat, series=seg, value=h,
                notes=f"stack segment {bottom:.0f}-{bottom + h:.0f} ch"
                + ("; untuned (hatched)" if kw.get("hatch") else "")))
        tot_txt = [t for t in texts
                   if abs(t[0] - x) < 0.25 and t[1] > total
                   and t[1] - total < 120]
        rows.append(_row(
            kind="total", x=x, category=cat, value=total,
            value_display=tot_txt[0][2] if tot_txt else "",
            notes=" + ".join(
                f"{n} {m[1]:.0f}" for n, m in zip(seg_names, members))))
    sheets["j_two_craniotomies"] = rows

    # (k) retention: join the Hahn 2026 JSON for provenance columns
    p = Panel(cap, "k")
    with open("data_fig_hahn_retention.json", encoding="utf-8") as f:
        hahn = json.load(f)["participants"]
    parts = sorted(hahn.items(),
                   key=lambda kv: (re.match(r"[A-Za-z]+", kv[0]).group(0),
                                   int(re.search(r"\d+", kv[0]).group())))
    bars = p.bars()
    texts = p.texts()
    assert len(bars) == len(parts) == 14
    rows = []
    for (x, h, _, kw), (pid, v) in zip(bars, parts):
        assert abs(h - v["retention"]) < 1e-9
        ti = nearest_text_above(texts, x, h, set())
        rows.append(_row(
            x=round(x, 3), category=pid, value=h,
            value_display=texts[ti][2] if ti is not None else "",
            notes=(f"last_day {v['last_day']} d"
                   f" ({v['last_day'] / 365.25:.2f} y);"
                   f" best10 n={v['best10_n']}, last10 n={v['last10_n']};"
                   " hollow bar = < 1.5 y"
                   if kw.get("facecolor") == "white" else
                   f"last_day {v['last_day']} d"
                   f" ({v['last_day'] / 365.25:.2f} y);"
                   f" best10 n={v['best10_n']}, last10 n={v['last10_n']}")))
    rows.append(_row(kind="ref", value=0.5, value_display="",
                     notes="dashed 50% line; panel note: 7 of 14 <= 50%"))
    sheets["k_retention"] = rows

    # (l) stroke vs ALS
    single("l", "l_stroke_vs_ALS")
    for r in sheets["l_stroke_vs_ALS"]:
        if (r["value_display"] or "").startswith("~"):
            r["notes"] = (f"printed as '{r['value_display']}' (approximate;"
                          f" plotted value {r['value']})")
    return sheets, cap


# ---------- Figure 5 --------------------------------------------------------
def build_fig5():
    cap = CAPTURES["main_fig_5"]
    sheets = {}

    def leaderboard(letter, sheet, note_of=None):
        p = Panel(cap, letter)
        bars = p.bars()
        texts = p.texts()
        used = set()
        rows = []
        for (x, h, _, kw) in bars:
            ti = nearest_text_above(texts, x, h, used)
            disp = texts[ti][2] if ti is not None else ""
            if ti is not None:
                used.add(ti)
            rows.append(_row(
                x=round(x, 3), category=p.category(x), value=h,
                value_display=disp,
                notes=note_of(kw) if note_of else ""))
        return rows

    sheets["a_benchmark_splits"] = leaderboard(
        "a", "a_benchmark_splits",
        note_of=lambda kw: "test split (hatched)"
        if kw.get("hatch") else "train+val split")

    def kind_of(kw):
        if kw.get("hatch"):
            return "single model (hatched)"
        if kw.get("linestyle"):
            return "unverified (dashed outline)"
        return "ensemble"

    sheets["b_B2T24_public_board"] = leaderboard(
        "b", "b_B2T24_public_board")
    sheets["c_B2T25_private_board"] = leaderboard(
        "c", "c_B2T25_private_board")
    sheets["d_B2T24_entries"] = leaderboard(
        "d", "d_B2T24_entries", note_of=kind_of)

    # (e) 1-RNN vs 10-RNN pairs: base left lower-text, ens right upper-text;
    # third text above = relative reduction
    p = Panel(cap, "e")
    sheets["e_1RNN_vs_10RNN"] = pair_cluster(
        p, ["1-RNN baseline", "10-RNN ensemble"], 0, extra_top=True)

    # (f) ensemble compute
    p = Panel(cap, "f")
    bars = p.bars()
    texts = p.texts()
    rows = []
    for (x, h, _, kw) in bars:
        ti = nearest_text_above(texts, x, h, set())
        rows.append(_row(
            x=round(x, 3), category=p.category(x), value=h,
            value_display=texts[ti][2] if ti is not None else "",
            notes="total VRAM = n_gpu x GB per GPU (tick token)"))
    sheets["f_ensemble_compute"] = rows
    return sheets, cap


# ---------- Figure 6 --------------------------------------------------------
def build_fig6():
    cap = CAPTURES["main_fig_6"]
    sheets = {}

    # (b) ASR corpora (log axis; dashed T15 estimate carries no printed tag)
    p = Panel(cap, "b")
    bars = p.bars()
    texts = p.texts()
    rows = []
    for (x, h, _, kw) in bars:
        ls = kw.get("linestyle")
        est = ls == "--" or isinstance(ls, (tuple, list))  # dashed bar
        ti = nearest_text_above(texts, x, h, set()) if not est else None
        rows.append(_row(
            x=round(x, 3), category=p.category(x), value=h,
            value_display=texts[ti][2] if ti is not None else "",
            notes="patient-side estimate (dashed bar; value 586 h appears "
                  "only in the panel note text)"
            if est else ""))
    for v, lab in ((960, "960 h LibriSpeech"), (309, "309 h Switchboard")):
        rows.append(_row(kind="ref", value=v, value_display=lab,
                         notes="dashed ASR reference (hours)"))
    sheets["b_ASR_corpora"] = rows

    # (c) home vs lab + lab-ceiling band
    p = Panel(cap, "c")
    bars = p.bars()
    texts = p.texts()
    rows = []
    for (x, h, _, kw) in bars:
        ti = nearest_text_above(texts, x, h, set())
        rows.append(_row(
            x=round(x, 3), category=p.category(x), value=h,
            value_display=texts[ti][2] if ti is not None else ""))
    lo, hi = p.hspans()[0]
    rows.append(_row(kind="ref", value="", value_display=f"{lo:.0f}-{hi:.0f}",
                     notes="shaded band 'lab ceiling ~ 20k' sentences"))
    sheets["c_home_vs_lab"] = rows

    # (d) cross-subject: SU left lower-text, MU right upper-text
    p = Panel(cap, "d")
    rows = pair_cluster(p, ["single-user", "6-user pretrain"], 0)
    for v in (2.5, 5.8, 12.8):
        rows.append(_row(kind="ref", value=v, value_display="",
                         notes="dashed ASR WER reference (2.5% LibriSpeech /"
                               " 5.8% Switchboard / 12.8% 14-set avg.)"))
    sheets["d_cross_subject"] = rows

    # (e) voice vs text + B2V 2.0 ref
    p = Panel(cap, "e")
    bars = p.bars()
    texts = p.texts()
    rows = []
    for (x, h, _, kw) in bars:
        ti = nearest_text_above(texts, x, h, set())
        rows.append(_row(
            x=round(x, 3), category=p.category(x), value=h,
            value_display=texts[ti][2] if ti is not None else ""))
    rows.append(_row(kind="ref", value=5.24,
                     value_display="offline B2V 2.0: 5.24%",
                     notes="dashed reference"))
    sheets["e_voice_vs_text"] = rows

    # (f) non-invasive: MEG left / EEG right; lower text = EEG, upper = MEG
    p = Panel(cap, "f")
    rows = pair_cluster(p, ["MEG (solid)", "EEG (hatched)"], 1)
    lo, hi = p.hspans()[0]
    rows.append(_row(kind="ref", value="",
                     value_display=f"{lo}-{hi}",
                     notes="shaded noise-floor band"))
    sheets["f_noninvasive"] = rows

    # (g) attempted vs inner
    p = Panel(cap, "g")
    bars = p.bars()
    texts = p.texts()
    rows = []
    for (x, h, _, kw) in bars:
        ti = nearest_text_above(texts, x, h, set())
        rows.append(_row(
            x=round(x, 3), category=p.category(x), value=h,
            value_display=texts[ti][2] if ti is not None else ""))
    sheets["g_attempted_vs_inner"] = rows

    # (h) 160-wpm wall
    p = Panel(cap, "h")
    bars = p.bars()
    texts = p.texts()
    rows = []
    for (x, h, _, kw) in bars:
        ti = nearest_text_above(texts, x, h, set())
        rows.append(_row(
            x=round(x, 3), category=p.category(x), value=h,
            value_display=texts[ti][2] if ti is not None else ""))
    rows.append(_row(kind="ref", value=160,
                     value_display="able-bodied 160 wpm",
                     notes="dashed ceiling"))
    sheets["h_160wpm_wall"] = rows
    return sheets, cap


# ---------- Figure 7 --------------------------------------------------------
def build_fig7():
    cap = CAPTURES["main_fig_7"]
    sheets = {}

    def two_label(letter, sheet, sub_is_note=True):
        p = Panel(cap, letter)
        bars = p.bars()
        texts = p.texts()
        rows = []
        for (x, h, _, kw) in bars:
            above = sorted([t for t in texts
                            if abs(t[0] - x) < 0.3 and t[1] > h],
                           key=lambda t: t[1])
            disp = above[0][2] if above else ""
            note = above[1][2] if len(above) > 1 and sub_is_note else ""
            rows.append(_row(
                x=round(x, 3), category=p.category(x), value=h,
                value_display=disp, notes=note))
        return rows

    sheets["a_chars_vs_words"] = two_label("a", "a_chars_vs_words")
    sheets["b_phonemes"] = two_label("b", "b_phonemes")

    # (c) tone decoding bars (the tone-contour sketches are qualitative)
    p = Panel(cap, "c")
    bars = p.bars()
    texts = p.texts()
    rows = []
    for (x, h, _, kw) in bars:
        ti = nearest_text_above(texts, x, h, set())
        rows.append(_row(
            x=round(x, 3), category=p.category(x), value=h,
            value_display=texts[ti][2] if ti is not None else "",
            notes="data on request"))
    rows.append(_row(kind="ref", value=25.0, value_display="25% chance",
                     notes="dashed chance line"))
    sheets["c_tone_decoding"] = rows

    # (e) survivors: printed range label, bar drawn at range midpoint
    p = Panel(cap, "e")
    bars = p.bars()
    texts = p.texts()
    rows = []
    for (x, h, _, kw) in bars:
        above = sorted([t for t in texts if abs(t[0] - x) < 0.3 and t[1] > h],
                       key=lambda t: t[1])
        disp = above[0][2] if above else ""
        note = above[1][2] if len(above) > 1 else ""
        rows.append(_row(
            x=round(x, 3), category=p.category(x), value=h,
            value_display=disp,
            notes=note + "; bar height = printed range midpoint"))
    sheets["e_survivors_per_year"] = rows
    return sheets, cap


# ---------------------------------------------------------------------------
# 4. annotations sheets: every in-axes text that is NOT a bar value label
# ---------------------------------------------------------------------------
def build_annotations(cap, matched_texts):
    """matched_texts: set of text strings already recorded as displays."""
    rows = []
    for letter in sorted(cap["panels"]):
        if letter == "~":
            continue
        p = Panel(cap, letter)
        for (x, y, s) in p.texts():
            if s in matched_texts:
                continue
            rows.append(dict(panel=letter, text=s,
                             x=round(x, 3), y=round(y, 3)))
    for (s, pos) in cap["fig_texts"]:
        rows.append(dict(panel="figure", text=s,
                         x=round(pos[0], 3), y=round(pos[1], 3)))
    return rows


def collect_matched(sheets):
    used = set()
    for rows in sheets.values():
        if isinstance(rows, tuple):
            continue
        for r in rows:
            if isinstance(r, dict) and r.get("value_display"):
                used.add(r["value_display"])
    return used


# ---------------------------------------------------------------------------
# 5. xlsx writing
# ---------------------------------------------------------------------------
HDR_FONT = Font(bold=True)


def write_sheet(ws, header, rows):
    ws.append(header)
    for c in range(1, len(header) + 1):
        ws.cell(row=1, column=c).font = HDR_FONT
    for r in rows:
        ws.append([r.get(h, "") for h in header])
    ws.freeze_panes = "A2"
    for i, h in enumerate(header, 1):
        width = max(len(str(h)),
                    *(len(str(r.get(h, ""))) for r in rows)) if rows \
            else len(str(h))
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = \
            min(max(width + 2, 8), 60)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    builders = {2: build_fig2, 3: build_fig3, 4: build_fig4,
                5: build_fig5, 6: build_fig6, 7: build_fig7}
    audit = []
    for n, builder in builders.items():
        _run_figure(n)                 # render + intercept BEFORE assembling
        sheets, cap = builder()
        # annotations = every in-axes text minus strings already recorded
        matched = collect_matched(sheets)
        ann_rows = build_annotations(cap, matched)
        wb = openpyxl.Workbook()
        wb.remove(wb.active)
        for name, rows in sheets.items():
            if isinstance(rows, tuple):        # fig3 scatter sheets
                rows, meta = rows
                header = list(rows[0].keys())
                ws = wb.create_sheet(name)
                write_sheet(ws, header, rows)
                if meta:
                    ws.append([])
                    ws.append(["_meta"] + [f"{k} = {v}"
                                           for k, v in meta.items()])
            else:
                ws = wb.create_sheet(name)
                write_sheet(ws, HEAD, rows)
        ws = wb.create_sheet("annotations")
        write_sheet(ws, ["panel", "text", "x", "y"], ann_rows)
        out = os.path.join(OUT_DIR, f"SourceData_Figure{n}.xlsx")
        wb.save(out)
        # audit
        audit.append(f"=== Figure {n} -> {out}")
        for name, rows in sheets.items():
            k = len(rows[0]) if isinstance(rows, tuple) else len(rows)
            audit.append(f"  {name}: {k} rows")
        audit.append(f"  annotations: {len(ann_rows)} texts")
    report = "\n".join(audit)
    print(report)
    with open(os.path.join(FIG_DIR, "..", "tmp", "source_data_audit.txt"),
              "w", encoding="utf-8") as f:
        f.write(report + "\n\n==== FULL DUMP ====\n")
        for n in builders:
            f.write(f"\n##### main_fig_{n}\n")
            f.write(json.dumps(
                {k: (v if k != "panels" else
                     {L: [f"{c['method']}:{c['args'][:3]}"
                          for axd in A for c in axd["calls"]]
                      for L, A in v.items()})
                 for k, v in CAPTURES[f"main_fig_{n}"].items()},
                ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    main()
