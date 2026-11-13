#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Standalone renderer for the 8-Study Consumer-GPU (16 GB VRAM / 64 GB RAM)
Communication Neuroprostheses Reproduction Benchmark (Figure 2) and its
interactive HTML edition (Figure 2 + the phase-diagram companion panels in
an "Insights" section and the memory-wall panels in a "Memory Savings"
section, both at native size + Figure 6: The Limit of CTC-TLG).

Reads `reproduction_benchmark.yaml` in this repository, syncs
`reproduction_benchmark.json`, verifies zero text overlaps,
and outputs publication-ready `.png` (300 DPI), vector `.pdf`, and interactive
single-file `.html` (`communication_neuroprostheses_reproduction.html` +
`index.html`) artifacts.

Usage:
    python render_figure2.py
    python render_figure2.py --yaml reproduction_benchmark.yaml --out communication_neuroprostheses_reproduction
"""
import argparse
import base64
import html as html_lib
import json
import math
import os
import re
import zlib
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

PANEL_A_IDS = [
    "angrick2024", "wandelt2024", "brosler2026", "moses2021", "silva2024b",
    "metzger2023", "littlejohn2025", "metzger2022", "feghhi2026", "willett2023",
    "card2024", "feng2024", "kunz2025", "li2025", "feghhi2025",
    "levin2026_speech", "yzhang2026_bit", "nason2026", "boccato2026", "jude2026a",
    "jude2026b", "yoon2026", "card2026", "fogg2026", "zzhang2026_align",
    "pouratian2026", "fan2023", "karpowicz2024", "willett2021", "wairagkar2025",
    "levin2026_handwriting", "wairagkar2026",
]

PANEL_BCD_IDS = [
    "willett2021", "willett2023", "fan2023", "card2024", "wairagkar2025",
    "kunz2025", "kunz2025", "kunz2025", "jude2026a", "jude2026a", "jude2026b",
]

# 2026-11-11 split round: PANEL_GH_IDS (study-id slots for the former
# RAM/VRAM panels) was removed together with those panels -- they now
# live in render_phase_memory_figure.py, reading the same YAML rows.

# 2026-11-11 metric-split round (user: "fig2e，其实应该拆分成多个部分，
# 第一部分画PER，第二部分画CER，第三部分画1-r"): panel (e)'s 11 pairs
# regroup by acoustic metric -- greedy PER bars first, CER bars second,
# the single 1-mel-r voice bar last -- with dotted separators rising to
# bold group headers in the axes top band (headers cannot hang below the
# tick labels: row-4 axes patches start right under the label tips and
# paint white over them). YAML row order stays parallel to panel (c) so
# PANEL_BCD_IDS keeps mapping each pair to its study in the interactive
# edition; only the drawn x positions regroup.
PANEL_E_GROUPS = ("per", "cer", "mel1r")
PANEL_E_GROUP_TITLES = {
    "per": "greedy phoneme error rate (PER)",
    "cer": "character error rate (CER)",
    "mel1r": "1 \u2212 mel r (voice)",
}


def _panel_e_group_geometry(per_rows):
    """Compute metric-grouped x positions for panel (e) bar pairs.

    Parameters
    ----------
    per_rows : list of dict
        Rows of ``panel_e_repro_acoustic`` in YAML order (parallel to
        the panel-(c) rows and ``PANEL_BCD_IDS``).

    Returns
    -------
    tuple of (numpy.ndarray, list of tuple, list of float)
        ``(x_pos, group_spans, separators)``: one x position per input
        row with bars regrouped PER -> CER -> 1-mel-r (unit intra-group
        pitch, +0.8-unit inter-group gaps), the ``(first_x, last_x)``
        span of each group in ``PANEL_E_GROUPS`` order, and the
        separator x positions midway between adjacent groups.

    Raises
    ------
    ValueError
        If any row lacks a ``metric`` key in ``PANEL_E_GROUPS`` or any
        group ends up empty.
    """
    bad = [p["label"] for p in per_rows if p.get("metric") not in PANEL_E_GROUPS]
    if bad:
        raise ValueError(
            f"panel_e_repro_acoustic rows missing valid metric key: {bad}"
        )
    x_pos = np.empty(len(per_rows))
    spans = []
    cursor = 0.0
    for g in PANEL_E_GROUPS:
        idxs = [i for i, p in enumerate(per_rows) if p["metric"] == g]
        if not idxs:
            raise ValueError(f"panel_e_repro_acoustic group {g!r} is empty")
        for i in idxs:
            x_pos[i] = cursor
            cursor += 1.0
        spans.append((cursor - len(idxs), cursor - 1.0))
        cursor += 0.8
    separators = [
        0.5 * (spans[k][1] + spans[k + 1][0]) for k in range(len(spans) - 1)
    ]
    return x_pos, spans, separators


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


_STUDIES_B64 = (
    "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"
    "BMe0ubmbstw//D/wqQjc"
)

_WORLD_110M_B64 = (
    "eNqUvduyJcdxLPgrML7My17bKu+Z5w28U6JIGkBK51Cjh02yRfYIaMgagHTEsfn3U+HukZVZqyXZvAA7umpVZeUlMjLCw+Mf/98f"
    "vKUf/I8f/PRvfvGDlx98ePv6nQnv/5/3p/Tx/Yc/f/uD//GP//iPoR8vj1Bfj/ZPL1Mo1YQ2XlOF3CH215ZOsb0eQTKv1qy7A8Sc"
    "JGbKqS9PPl9zSvhx4MOKnpUaxJT9XRFyTZJLOeXzV7y/vY4EOQ7JqfN6lRwHft+i5Krn+7fM11t7HtbeNthE+wX+YUT2BuS1ax5b"
    "r91+/E//9P+9qOt/+/vPr67/7duHv759eP+2dX9Kr+cPj9dhL0n5fOD5mNfDXpLOxr880mvQ3xCqNT+N17NlWcLZVfnlUV5HoHT2"
    "wzkIui9nk3qWdD6j8YHnaA0Tmp539m7nwOpSf82DUjmfcf6brg0bhwMPycdripBSfBbtlxQHG5atxwNH7PygLrFRtLadYoFYX1un"
    "OCiWABGjmcpropgze61U3hwpxs4mohfPDkb766DUIGXcGl9bNSkmSvzuASngqWfnVkoBEttz4Cmdk/VJitavhw1APF8bbQDQsmiv"
    "xUhlSgVS1rWUbUwLhVZMyIVPDBjuQ++2vkjqqAN3pnOhXi1Jr0fSnWcXx9ceKFkPx3NVfkqyfogamuO127i96gUZQtAjGydoYKd0"
    "SkkdrZm8rIAvP//5tQL+4fWzL9/+8vZxXwJnz51rM54rFEvLJfS0Sd2kLGG8xPLaO1ZdeB3NRIwXxPwSz67gmozWuFOMvDlZJ8Zz"
    "wP3qqT9ieE387blEyiKew1dNPFyKJuXo4iJlG4BTLJIqHturxIiXpqA29GFNwlS0Npy/zFeLjvOnlZ3JuW5ikJSGSfryw7ThKY4h"
    "MfAq3jpsEizSqYbPDjzUnaeeOqUQKZ0D/dTxp7QM4I8+/9U1gD96+/D2p9vohfP7en7J+qZog5KHuu78vBFMHGz4OVrngiiH956N"
    "bDa5J5fHeCla6JDPgSnRB+ZcGafeLpGrEvKpCkpytczVUTJnMuVscv+UGK1dL6XMOXKcE9vkwbefE7yd8rl8ObqmL863Nx/AZOvu"
    "pXSuolM+39tM1vCf8jkly6COO+VzsXeTm35fJWfdf/ZGsd8PPc90il33rzu7Nb9Udm1GY+vhjc/ndDVRc+28fPZkra/HKp6vDp+U"
    "TelHk3PzppxTrHafYqaCIfqHnPP8vDtF/9Bzlp1yTd5RA9dL8I7MkKH50PHnPGw2b+ew4vdNA9OtY9tBPQM5ZJNzn9OmmJzmtDoV"
    "mLW++7Q7tVbTFgA54mv1+mTru85+PSftqWVOWeNmq7tB5uKzduNrJZ8TwEQ17lQeMVnfdD49FNtN6xzFYIrxFOfV8+amjjyvFZM0"
    "oc6V3E3s/OWB3dZu1rI/34HrasjRrBvt/jGvB3uV5sAph2Zy5fw9Kp7XX5uLAaKeVmy2naJ65ciSZRAdybTZKevhwcbA2o6Hj/E6"
    "rqaPbuv6FDmgwx46+2DAGLMX8WLF1DtNuewieohLeJxrBt1HBTay7QEmVokRE4uG2EDnaq+3J53DaLPIxVOJtqBJd94bg4lc7APK"
    "+RSpWUa0rj5FzvBhxsU1YUfUfOdboY5tOi1S10rqA712/pdK1roTQxIlniaTdRtvPocWfcqL1ZSbDT/FcnYYRq+7iAdzT+j5FS1i"
    "//doetZ6gm8NZhOdYpgiG9EkDowO9nhcrRhYSQkDG/jghElRpfLOtzaI+rpi9kF1K7mbgQwdVPV15xZXrd/9Y89WZP+ebsrwFIO6"
    "7dw4Tonb4dmp54POnTO4dIqnkqkaHNM2UQvwHMnTVjtFzlubIqY0X+f8SSbxc0zs0LhFPz07pngXD9iHxfYgidD+mk7nE02i+u1Q"
    "zaXqVhtmiLldn1qkjNkt5zbTfFy56yTv4Io9KWsgI+4NRWLGBqa3RmyHbuScY5Vtc+QoN3ThuZeyu+18lk0seA9Ob+e9xe891+D5"
    "3JAlntb5KXKCnPeeq8wawavnZDCJ3d0q9tTqz63Y4TWF7Vxm/XkUPaege7tfw+7GLms44tRDC9BOaMFGMfmDMkRuxa28YgLwu1uG"
    "Qjk/nGIyg6RGHg8eLWIqRQ35uaZN2c9+CHbyPEV+m63iYSKnbIUBb5tslHjuIzZdeBWnVv8eqEKT2N+2KDCV0IUVm4J9LMWMtXzw"
    "zPA4ZzbtAEpmGBbT3ZAwqKfmdqmbpN/BjLeZhBaYpQARjS0N1k3W3D7FxDHFd9sTs00d/bTaojlFavOCDdUmVpNoyyRqFtq8xszi"
    "t51PtlnoJjG+2UR14WGr/jTxOKErDNtTPLp/ecXVoW5KEJO6EMZhTOrfaHamjArbP2BaUvVX7JX5nERZw3pePJcjZ89hKi27wYwd"
    "8SX3+ZZTYech9Xc24TQO8vC3ZttDc9dzz6vnQjnFw8exL8/NNhynGH1U7afFf4o9JPvx5ewXE+VNwVAW/NaHstu9VP02H7OJHI7z"
    "wedkyr48z9dSjO3qw5y0VCq211PkjlMx0DlLM1mTE548dPWgWNXlAe/V4oC9YR+0SM27Akoi2w4m0Sz/ph3TVk42MRcNyGEfy43a"
    "DLmArw0SazWRZvE5lAf6Ivn6zeFq4ikWiOzGc+2fO+glZjsFmiitcarh89s1K8w3ZWJwPWbC4dc6ey25csqLCPfUJQ4MdFKfNjMN"
    "TYyuPcd5c9QKNjGbSGOzHxi8qGMrd+pTjOHS9znQ1YNtfSxiwmgFnZ5OMeDm4VcD3nv4b4dJOfmDExo1NxJ8X2kST9Vsk8Q3oYJO"
    "5dCeDy6YBuzGnjDrixZBt5lmYxv86mlXXaK5Ma5pblZLNrG4OZEgTusCq1atyLbR2L0uNqy96mLnUtSeWqENoltSuZrIJdFx6jxF"
    "HkfMCOtTHZjBhgXPiXCKFZJbHufGeIoc2tPyqNlEro/TQKjBxOzWRMXN1W2NRLG56YHfymrJmHDnIcOt0wAtRFPfRZnPttD8iH0c"
    "+NThNvtRlmuNoh8uFiHNv+r6zzDR/Cx/LKd6HvHhKz2H1M6OUZ/ZoY3MGguawXZIiRqI8Ipbm1uPBzfqKrEt+/b6XLzI3KqnER65"
    "v54rxUzyJF9Ax1A096ac4rnwT3GaD6cWb76f2VLPJvpSP1tsz/XFfaqf5ivSfNEDYtdqzhCbm0beJnYGnJ5mbQ43umCLJhcLriY3"
    "s+Ly/c3sczM2qn9ReKnu+7AHL2bs+h682GbX2eqsFXzOn3OGnGLx+cOrmmxQZ3ZzknjaMaYO/be9mFh93p6GQ5NBur7HX3xOhrPT"
    "itvXyaTsh6xTFdrF6kcw3MueHjyXN2mO80kNYvU3zQfzTdWca81tqPObzlPkKdLOOEWbAPMoyFGcxno37WdX5wkUj+KecooVv6U5"
    "NmCKmujdM9/LQba4g/UBR/XUCY19Hf00iM7trgXOWW3d52fFjr7mdNI5s/rBstjZq7mOGw2ttvcvDWmaLzytNHrw7UmntWf3dtdF"
    "YRHhC7YH+6l04ObDT6U2H4ofV+DytwngJzMbiOK7AfbJVuRdMcWLm32raJg7zY8gp2oxMWjW2jIssld5rrC55HM6ox/nKXWgl6Xv"
    "k+k562U/TGIaamtYRoQBnIBja1OECM6TjkE8XPvZJzb5Pszdip7N00lTMI1b8uuN04dPW5+u9x0Yuek9wVq2U9X05HTMt+DPK/km"
    "YwKW+Xw0thf/uT8cbzMD3ExqTpoCF8lpGFP12anTdD6P/nYRVnNwSz7DIcu9zo5t2GaoJgo8lLbNBIm8mV1+2v8ZT45Fohl8XdP+"
    "FA+IaoUdx7EXVoncgblbF2h+2639yQVGAo1SO3hi89YnJDzK9ULJsh3b/CLc6wcXWJKcUXb2gU2avGdiWR6EI5C1qcnGtp1ejo6C"
    "6EyWZ84Oddjni9+Zli/HDjFPDPZYHFqaj07DQZsmtfU4jkNqQ9HZqfhVH1jfW233LO5uifDweZCgw94wm90XjLmakhtxCNyYz6Re"
    "20nSjO50+CZfP1E/rb6Hh3A5XzJ/qeVPR1PSUuj4Hvtllq6IuCqrrMKDdbbBxYgmdXd18Ou4l3ZYNaeo9W97DPxDbipmXJVlvPQM"
    "rQOcM8+dOLn/wFRa1Cxp3ACjzmINjsLmR9ZmX9g83oO7TCx+pICvT2evLl918VPoea4xT6AfpwZcxcFPlvTANr+a4Qu8TovwrHGV"
    "22BEc49xU7ajfTGRZgishBcc13TzeYqoOslYNxSTsh9hG56UvBVHvLxyFd5z6+ahLzjX7ZxiNhrJhogzoRZ55WgaVLpWkj8Zvqqz"
    "aTp5VvgVo1wM8E1iyhW9KMHa0YG32uqbTpsKS6EG//ouszG5W+aIl1HJEbIO08WOuVzd35PoORx+5uMHuXOIq0IWWbOYjh2Z/SRn"
    "numsuW64Akw5txMP9LnaBLVknV58SuHm7ifPAVdo8hZXunK7O6Hg1W7+qBSuiQHbAR5xt1Y7HOT067cqJ3fwic5gRvPzZGCwwj9o"
    "IHQS2mW9Vvc7mTeXjms/NDGyMdxNeeC301dt5sbhKsM8LyY2v2re9eCrdch/TvG0N+Ky/DqQCGZ/J/9tMLG4HqAp36UzCkz3sPiU"
    "W9T667RVkywKO6lGmP1FWq5CB7h90WDnH27YT4VB+zK/woZgo6YZ5ye9giZ2jwcc9lN2lPmJ8Ro/yUU893CTLofrrQMuFfuphzcO"
    "NFgGX3kty1uuBtHIsINXMbMtxHkQ22TgEprrDAtVwQZW3M3wLbTGZ6QK5vYUE0zEPuNaARZkkQ0EJ3arHlWzOBhMyOLvhuE3nlrG"
    "tlu0FLFYbgWIphb4mKMHCQ/EdhXdMowFnYxTLpscdb1ml88trOhMZ8HYA2KecoSDUybVKQe4Q0ObwVxvHht8INRkqCePd5mJNvwH"
    "x6HzRnfxgIkmQ36YojT9McNjPJwoeBZh2wokgncF2IceystYL11KC22B/alQ9NI2zYwEv1f3oY4ZTqc+4Q8FTrQ+I+sF/tHhvRkB"
    "eTBTb0becfL32GPscoMqABy7IvUlLyFaM3qa/94i8YcbtOfzGwL9S4DW/bomZliho0yYAGSP3ybJR3V5fqw+P8CeFkTGfBWYu1n6"
    "BjMfR4swvRwFzoPhUViubp/aFaohSXvZVI9Y4SX64+2kEmTHXHKaDT5gJAwXa8Tl+fkD1kefgwO9JJTC+fkdrdFkidhKmu9v6p42"
    "UQ3rx2vqtu08h0hvwloNPn06jnCjudyuE53dbmfnMm8fdjnP00hEb2RHbQSgqOZx2OSO+4XaMPAKTogKv4f5+Dk4DbL3NoxG2+Z8"
    "cDa1E83daJolz6MVfl3mQe/gxzS/jofl5Jdt5KuDLg56AupcaR0rz1++9qT6tiL2m+RJwFK0wavNV27DPjjFjD3UsTnRzUfXGgN7"
    "quuFAGNsKBaO29OMJZvIy2GqkQysQStTbeQFa3DAnVDnKj+aYB3++3EFkHHmxMOE2ghulGiJ27CWFehQPJQ9Fw1RG1KQQG/CmpjD"
    "Xhl/b/578w0cc43S/TNRICEKtSEVFjKmxfE6H5fZk9FfRzH7bpXx9BGW1tkanHth5ZL1Fb6Y/PbrAsvhurtjT2/z2819kOZ1hIXM"
    "SIkuV9xf55LJmCXeFwH+BveZAtoRl+sHYgb2+zShGzCnHLphGBEYFc2vQ0Mcwy9zjqpr1zmrWXyoQW3uTwVPSNUnZoLVJSfWgGHh"
    "EY4BfO30rspN5QGP0fVxw32ABa+aPsAGZVjd9ZYwDnKhDxhwYWnHgWFr2Wc8Hq1tzhZAv1y322fN5cqG+4KIFLMvGNhbeW6rCY8T"
    "nuyo0rx5ynoYDcZO1UI9N5pcbNE7BWqIDTf/o3m+9I2dLtRpSEAZH/OTO5xkdf3kMnE4UX4+Nx0C1oEfNAZAbKbP3O9JdSc/8NVi"
    "d6me5mfXmho47HVfM3DlmuRe4fPzuuMHiHkBOETuxoCbm5v/50m3Hx7IMSDQSxcSGAeL+NImHKWgP4bb8A1zfbjPAFCK1h0AZJgo"
    "k+KCy3E7ylzV9NROz3WGyeYeUnMSjsvNbT7QIf1iXdFN5IqwaYomh3rZ6egodpx5bOzzu7fqvHnG1gw0bGLwcNP5ou4nXXOgmCRX"
    "S7YZaaI7XvpLd5hDhwrsMzhoUezudkED2LH7IYpRDBObQyIibq46cJ7WKnyoHlNt182m5CG5C4D3EmyDg7U1Ylxuie6aqeLs3T2i"
    "URHU6vMUD4f1KTZ3lpw6twsdWwF/NCno4nno7bKZAD1Ef0eP32Nw/IQfMa7VT88dF7N3xOB0iuoI2u0eyI2YeodftOBOv9AqdPwH"
    "h6R0OImHB4LqdZpq9Pq3C09T4aof/tOBo9cRlye5N94WBs5pM9IV2uWc71UHr6TlwIjAXFh40IST6a0eyWw4dQSfh7HO9ncAUkxq"
    "l3uttcVTt3z6eTM9zsldAYELy1dsQo8fjljiYcSbX9D/zR88oBninO+QPAKYuco8Vgitkev0O1pX9dkP3bf7bsvWRJ2UDWa/KiO7"
    "GGQSrmuVrkJaLs0RQXRhNneoVLhGFVZpgJlUB6Db8RhGUXC3Ikwo6iYzcenGCS7il13SfKnr3kiU6gzRwbwKrn8inD5ziyWC7/Bd"
    "k4bfmDvKxLMOAAwmlnd0zPjDg9INfkQHtw54VwzX5ODL2SY00lxhOHpNoFJg+MCdkh2nQr7KlAHCB3GqCkBypFYinPFDRkolbECN"
    "Xt/jLz4AGDkcPFMR/BeKJSJQ4avC4qDhggYAMZUdO1+jQB9jfgJ+mhykk4BAaI7vma9dYO+/+3LJ3Pndh/ffvfvTZ19+9/bdu28/"
    "++afP/v863cf3//xvwbCH5+Mqe/R9vTJ2PwWtb9H9Jdo/w4E2CACN/TADVvwhDzYcAk31MIN03BDPOx4iB0scUdS7DCLGwbjhtC4"
    "4Td2cMcT8uOGC9lBIzdEyQ1vsqFRbliVJyTLDeeyoWBuGJknBM0NX3ND36zQnBtu5wnVs2F+boigG17ohibasUY3JNIdp3RDMd0w"
    "TjcE1A0fdUNP3aFVK+zqhsnaAVs3NNcN6/WEBNtwYjcU2RPGbAGg3dBpd+zaDdl2w73dQXEbYq4ikpnlVbOdIpnIgAK0u3WTTIjD"
    "kXou2hrwM16D28NGq7mIKaTIujk9bFb4x3WT2nwu5he3MEDGMaEc02oZO0Gryd6KGUTvi9lhEeLwyEs1UaaWIaKzH+UAmYEYNXin"
    "QW2JJkuQLQetpoZ0resqcOWnOC2801bZxGJim5GXaqLmIsPFh5uDCCYn36MaUpBSl6IBkMWuujkYIZTLckxyvnBmJgf72nmtmdh8"
    "dz/HLimlAVAak6aVeX5ccvXVEAIwsV6WenLVx/BOckdygzMsjcs4ObVb8ohiQxAtOayFpkvy3Am7igdP0/dsokOH4f2DeNlAqS6A"
    "6vOn5YpAnaZv8nlrOAmTZAjjuG/5fjOEBrE7qMk6JvmMsVzVNIM5hzXQT/Ow5kw8PPHgnAMpuM7EOddSDtMVurIUrezx7P4S3dPV"
    "kdkVZ9YFch/Nh50vGMrMnzMRCXNTLZ47cZya3NKKTJxXK8RjAlyQdRi9Ufhp9SafU96e5Go+IhWvua5uyMXzHeL8+OjJOzgfWouL"
    "X7UExfHqVrP90kEcsJOT57V0GKGWaurBtNPqip4Ew/BZcjiI7bMm+cHb5vvhu7chq0xMntJg62ie4YGMT34+7kwsHN58O4yZmMeV"
    "9RLdMJw3z982jF3yUNt51ojupBxwj0RfAGZx4MnVczJwb5kJG7jYJrgKV1u4koWihxXM4LYebh6IO78nerLUgOKOE6mG2HX0VCv6"
    "v6LvFwNAkjmB9NtyockK55Ob55wFsuQH0kmreyHOXsDMdLdbQUqmnJoHwhqW/hamu4vfML3JFR3Z3TVU69Ud8paZXD2kxcd5GAD+"
    "kjgTFOm8tt69fNuYaWG6yw5MnzqjWOfYpDBd48Ae2QqeXsjwgrjfdLXnT8nu2RrQk1MOAAGkmZZrDmqoi+w+13NHTHHGKzP0apxh"
    "iALlGGXJIr5J7TPDEOklpQnxalDaDpSHQzm8IGN8uoijacFjepDPXdbkGR7N0JJphqTOKZTWEFXF79WcCIRemlGSCOWR6pRhUaY2"
    "I2oRvdlmkiRCimnOjZigwvsMqiEqk8aM0OVX7LBpBgwDdmuFVSIikGZ2ulhgX1y/pgXbrngj7LZ6xazM7vGAWYYJMgPTfHmbeb1Y"
    "mbnPaKQBMHA2mPE0pVnk+e2wuYKLCY9LV1etT9vgygGAsegZJyaeuinMyGbBTh5m9mfBtAozebUAT2KjW/bfy9d+yufWg9ng8vlt"
    "YYZuC4yBMIOZJuP5muUF6WBByYQQ8XgFYg0/VtDc+fjTTI8O8MD1anKcsuV4u/tl/Xp1hyHmosM4IHaT62yO6erDdUChnjx82hm8"
    "zcQ+f95wu7LwT/k8L8bgQaMCIyQeszeut8/mNCTbh+v5Aw/Q+6lrg4beMsuQcB/n8+bP9UAY2XHmkBes6zhDKQVuseg2rMkNLY51"
    "yh2yGgSukDgDSZbhxZT/PGU0uMz75/vVIviuoh/DIOMJPqQDajc4EAPJP3HGegroIGL0lVrslB3jnH/L0/m+isTXOtOjDfoFWfHj"
    "ikzSMjxUZlCsgbz04nJHsprStc2HVfH7ef98vr4Q+3BxV2NAYiFkFw2T6j41u91SHp2HwJbYgUR7oW8LnENlEhwgE9HuV1jGkKhM"
    "zIwuz9ezQf9dvvs9Pf4peX7Lrd8S75+y8u9Z+/es/nvW/50V4M4acGcVuLMO3FkJbqwFn2A1+C84EO4i4tGW6jnlgZTCyYBgaYxu"
    "vFhbCwbWb4fatxUT/dstD7Z7iPfsm5quHEnrO9xevJsNKjt8ZWWcca3nOC0yImF1avJMQN7hgwiURXVvcsjAM9XDl1WGY7QeHrmH"
    "ZwrTfF4vbVkW5ubk67VsD1NtxeNHoTCV0a1aky2X0f3gJme832c1UlRqcAOiwCCol6JBrsx1e1SO6DH3JTZX9oVN+rHMmwKIa5kA"
    "+AKanDIh7aVKzlNz2hqtUwshY8LUf3U5I+vXF6GpneJJAQHqxGTtGxVJ5fZS/rzi8GObK19XgQC3lSwlhDQCQ5y7ODAP/fEw5cvE"
    "jNCpW9y/FpgIW5z6oB76GG071m0NHxP9YyyD1cMz+DhokHhtAkhbLnPXKVjUMj3tl5jIY/4+oPPrlJna3GZnHVzEcerwgN8nlwdn"
    "/uzNhLkjnV+DlECd8mirTPDcVCI1aOkIP2Aecc7tMbsTcvfertTozXcEm5oevnV5bsJMIq1OcyPM8NyhKgBKdaIXKiiZLP+1+Nsb"
    "0OZxnQvVY5WYK/UCkYdKZMoMw1cm/Cc3RivOPQbijfN+YHzzvB2AYA1GJYeBMw0EpJgB9Dtf3/A4WWTWkO161uvy3B8zEcZj2S8N"
    "Gy2ZuetlDhZsJNts5tTPpEno/r5adrlxc+refktZn6hPaz+ut9nehkiTX0dSrwVf4xxNkmbk2R4QPvhSNVwoANjrz+dcZx5znXqv"
    "OruLFI/NTQCDnUzpAPjsmAYzA/qHr1UqGpMvRdQXCEiBp808j8UVW8Lvm+u9sT4tAr81KYQKGBMmaqgwD8qdg1Dq8YLz2yZR8XPt"
    "SBnW+8T3h5wVXNNCynD7N/en2J50EH5d/luiHTHNBSwez7w3OTDdgm+wthLdrtkyMPrJv6CCBssg+S5ybYR5e8HalFpv0NPXdSIB"
    "7Ofz9cwEkTncAN6biSJbc5cA2d9+/vsrQPa3b399+5e/fPvd24ctJNaRGGPhQXtUrwwOwWyhWyorH6/DI+r+d6aFmIPdBAAk7Khn"
    "n2vxDTj1oSiYMGYef2t6oyNbyWXmW0S8ApaJufgQCPFr5j9W5ogF8hEGkRCumEgDfiELpdWQaiSUE+A6uNGe33h45pgg63Q67g1M"
    "NObjGTCNnC3gosEZ2wToNIvVSEpjBoFAcAO3OyQkL2bZMbXLGQ5sTQWBlp/qqXhMapQKJHxAxTlzkdB+63JTdzjp28dg4iAoYS1W"
    "3DXhY8CSgRiXDa9tJoh4YVoXTNJMu9G2Ofxtbyqw0hThKUXBBxjedlRFdAEnsXPZN3QItFahQ0FAmRIVugqSygx3FCj3LKdyYXxM"
    "XCC8MXJfKHA3Z8FpjINsIEglieE7mGaFsSN5bgqDavKGI3cLQajCF5DyAV1ix5w8A1QF5DIWbEzXNRHGlOgSHxkYYCp8ucWMKl9u"
    "wXWEM+3vDqeI2HYye1zMg5nwbdFhZPDPuPsl4/hmzpbBawdi+FBoGQAeCxVXXjMrSJo8A1rm0ACjxEAOYmAjyfkBnQW+tsniVuBJ"
    "KjJ7QXuD3+EaiB7tWuY8SXUyhxSsS0eUFyRW2yhlTr1E3jgKBaRxWDkFasBs6cpZOUAYhy+o2AyKPISWaj3zDyriCUaAUnmJPDsw"
    "J2tR6gD0BU/TZshy9QUIOPhynRal/rZDJitWHxFNRdkmDZDqojSrlkRbh24wjQFWH/jSDWaNhqEbLLJKq3lQKuB0gZJrOEwxR8pC"
    "HOygSqmgT9Auqk27FqhgE4egURr8He5EvnbRubiD+6Roa+xwe2XFGU7l3pk+ADW9KP4VVPH7Hy6gir/+4d2/vH/aM540wqIsNjWy"
    "Kphd99z00qaydm22KbpVA9514643V52669tdF+96etPhu36/6f6g7SloJ8hoCzyrDQk//j5kFOCZ2mkOvL3rKbbDylCx4DiCsXAa"
    "mQ0AIeoFkIqu4FdcPQPQEB1J69AzsK8BJ45gaWI/wNcMCxaYJ4Qw1SkBsVH9KiFgGbRbdQQseWODQ11e3QoI8yrBGc51Bhesgq+E"
    "KCVly1VkFicpMMsnxOcc2rAYoD5cbZQZvSaVkgevq0fQYXPxsGbDEzXz6BmHXmL4X4NVgH1zdIb5RWawfJ3Yy5r4za9+dq2J37z9"
    "6/dvn/3q3b9/9rPv3394twOM6KR6EDhqdie5XkmAnEEo9VAQNoAqyDhjYbKbaGy7mcYKHCUd1LO8CofoQ8cNE08V89CGEzLArA+B"
    "nuFUAXFw4nstEg3q4CzROKIZtgzIKzeJ5NIZeKCHHOtmmR9k5u26SmpezF67ShZffh+S3SFjhgXuiiZjJePXvH+2a5B8OPpHjUBW"
    "4OSfQSLgoY88FeRDIAETDxIiN3WuNc3JqDOYbhVlNymCNjgWicZ73bmr2DBVdEhKEg90AWmqbUTZXdFFUAL3eZXjDcpsZL6BAxpX"
    "zYCIJvJFBTAhEQjbvRzw1iRWXAxdYgaHcGPfYj946LRtYsJU4pAW4Gge4j8x0fiUo95jOBFc7BI72nToNZFTNC+NQPv5QZYi91CY"
    "FAe1ZRaCMMzEFH18wyJ2zdEjSUzouFYkghE7Z5+FJhWfZudZ3cQ5CU3ic2xvwy+PpQmFGg4zMuCx3k3HMDEEiRWczc2vFva4DwBH"
    "J/jwkO05BRfB8KypH3x5eqee+vmhfIC119iLMJMfothQNOshNx2CU+hVdBtjSw9B5RA6Alt4jxKrScOl0JZ7M1Y20z3hoEdfBL9Y"
    "0anHvNebtKi6X/x4oRL+xYc/ffPh3bfv/0sd97we1tXCwSKeEAS2yST0aEI2+UPxRHMzVyzK7FcPrmdJFbzonFoJYd+HzHT76cB3"
    "U68kxJRtmRWJxgsuQgZzh9takR8Lnnxo4YNPjhjXrBkj9nGlLkMklTevBrwoamwSPGZG2O0valh3nCImgr/bHxzTIuLM+Qi0uM3h"
    "PyCSpz+Bhs54u7Pf3ExsepKpAmVk2WsGOL652hNCwg/5aUACHKELmrom4/P0W1AOPaJ/HkwEkYwjpJIuCcCTR+SmbeJIk6v8rhUj"
    "kjoe2v0ts5IK9pjigPqlmPlWYaORV6pdo0q2NgeZSbhe7OdDko2fQC3bm9GUBCfAgwAr9A4kviqBLeBRrr7DgsnedVxcze9tWKYa"
    "wOu5eI/lpp3miB4ckHirPA/kbJcXZZbqTlkjACy8MN0CV8KLj+QpnWqcoHykzmJG8FIlu742ylBFKp/9aoDWY0kIQzdfO1QA3cWc"
    "7wFezwcNcST/YXpQSZ9ihMi1HbhPJn9N4PweQ9KBlcL5YaOFeTeSRBZRwDEB6Z8QOe8MYoIpzHV0GFriRXTkyiQV/ESMPlGfdoC6"
    "yK8FsAMo+dkadGoB5U2Z1Gc3YwEFTd0AyFeYHWAwA9qUyB+ML74omSnpaz+AgEWpOyaZza/vITn9IvnM4OJAaPhBArUg0n/5pUw0"
    "G8FNRgsbDnR5kRjqrCxg6d6FV9FA42nHxaGLATYBd/LlrWwF6CymuWGk9rAvaF/GovfSNrE8dEj+nriYJuuT+GhoZxnfyFl/EdGO"
    "WqwovUkGyvDHdjNFxHOB5seXqQMalrXwE8hz75fOs3uhLXUr72yS8otwDybYSHcXvJVsNRzoiksKKiOAhBSY0LgmRXso5yfV1+Fj"
    "YB7il6mqQFg1r+EILAAAkuP7dSftbiXWmKj9IOiXGd87XCMm3MzNI3KZ6gQFoA03jyQxDSy9qEehbIc6C365h/zQJkZqA/+Y0GZN"
    "E4iLURNpRRcf2ijDLPbr0+cmbMgLGFe9+Adxx24SK25WMw5O5OoCJq5eA7ffVCwRR/YHcetgAEiXcWh4HTSwSbIVIH49E2kbsoEG"
    "LcKcHq6V+yLCne9LNvhWyC02DFZYoSMKL7V5xIsc18O/E/5MV3v21e2anNf843w0VC52vaomDJ7TvLncQuvy3UNfClwQdsgpxnUD"
    "XZ7MV8Fx9+i+LOGceXQfOkuEeHTfvGkgib7efsoqNcN73E54XWtsijkvDenejuW13EPhqXkoBGYi7cU4ux3GZHO1yp1+uBjxzbR2"
    "Q9XBlWeI0HQi1mA3TF3lzzBp3a3UtRFolWV141ij/airfIx2aUA1HkIxm9hgr/MUH5Co81BmoIlmrAhyg80Xoh5VcJxStp9dVbEb"
    "35oy2+WP4tXk+1uJ0+62RvGLfAu16SG/M74AZ+XqYmQr8NoD9FgPJVaYGPt1ALHcbBxAul89yjyAHA3qo+hrj45Nv/qTln5kx2bW"
    "l5LeOpAy9FDEwMRO2zTrxRUGBXXpgezMh4LVoNKo19ntAKHeo3grgZh9CApsANveroPRAUjHQ3FrE7nHcie0xO658g8oJreBD551"
    "5O0YVk3A1r3dObBa5UE3XssXsZQgH12Ob0Mo9xedLYyU0+wJNHEgc0H+6wHsuA6eA7hFnaIHLERFOQZ4z8U3PRAxU/TSWHDGNFUO"
    "5JLEV/86mlPde+KFOGHrJGyAVDcHWEBlix4JBvuhhXgAJ+YFsrZRXc6Zn3+xuNQ+//jndx++e//hlqxH3+xjsglVgPoeziZk7kyK"
    "vVxEbuBH9FTmg7JTbpukpAC/Wb2lJCBwLerZhWL3XGw05S6jaWR/pA5JTkNfGs2gIDRLof01kdFELD5SuuRUIBPiSGSPyUQ6wbH+"
    "sPOb8xna+TE5KTj5DU0e/SI4fCRPqiNi8+GZJYV9lxyOS/LEx0xhMaBQhjw8XTNR7i6j9ePK9bTDYXbEq10vkJkRUHlwzuFKX7cd"
    "1D30luHZ+Pt+Mbjjds9gx6k8XnKCmJx3z7SshyFB2ke5zNv58+5TwXZsxRtAp2ePy1O000h2WBNDC4vMunXZ83CQTHzKdVLV27ub"
    "f2mhW6854bycE17JwxAhDbLnppmqunJ76awtzvdWB71fjkCrnarNSRORBI2JOSne7WNM9tIJkIKT9JnHtUTP1uMwluBpWhHWVfFh"
    "bNywgNfwjDC2LjkHQGbrZ+4ZjCqHrej5eeZC0d5cKwKYgTCJQ4l7QFd7+qD5bnL2dCng2R4eVEKhCI5k9uxDm/UWI57ZiBUD38b6"
    "PgdSNE3b6MlmgdM2enEM3e9AFtapsGXQl2xHLJt45QOaXMqVqfdIM/1cslOutoOr2CFyjEDZqg5eQAG+23wx9uP5MwGU5+q05CPa"
    "HmrZGbMAQ4SSmhmJpqvjuK7Dj+wUzBZNOkWnkbbA0ClKh1QeN2O9ij0kyjMlE74mp2CuAImbfPhCgffMKWlQvwZy83Vn5kP0NFoi"
    "vOz60ZdVHUU9Xun4isEfB7I73O4qq/J10vgHr6er1sOxNq+AJvQxi78Rcv6IngsihW95ZbodBokDvArdmV4Wr4AYCV05i3dATM7x"
    "29h3k2xXD2dfICQNuTjNb+VQBOf5tYUWJ8du9tdlLw5iVoylTvn2AE+fkxlvu9daWe7nv1wqy/3l/Vfv/tstet8n7/vo0zYLh5to"
    "zF15Ocd1hQvAZC6Xyl3Kkc5gsbq2dIZPTdRiBWDR3lWcn2S8kLfYC9JUNL3nqyINSJD95zg8Js/qBATMHnf4Wit8Xp31S6hph6va"
    "RF07F8/NZGAdGxQ49c8LrJfqlLB2ogkKU2E9mUE8pKxAQ3GOoXNx1U4fm+cBM3CL9dH+k/V1X3/7+ryv3+f1/bT+F+1wUx03zfIJ"
    "zXPXTJvmumu2u+a7a8a75nzSrHfNe9fMz5p71+zPmv++M+w7x/POsu88zzvTfee672z3ne++M953zn1nve+8zzvzvnPfd/b7zn+3"
    "DJ4th92yuFkeN8Pk2WzJXKvRR9/8YcXTBBt1eREdkmuCdA2GlnJ33iVopeQMQwjNrJpgEzNCr/ZyTw2HXnG2cGarPxwg9QBl49o1"
    "YLl+TPKOJuuzXcnvptsthu+/h8XobGENeXg2lM0VlymWSc4A3iXMDJ9p8LTFyyaDS8yJr/h12WuwtsyBd9x3QwaDmd6aiMkn+vDO"
    "sYmXvFgeYVG2MKK/Dk5or4fQ6DRMvi82ZEzawizhYiu4zjkNqC47J+W50BoUw1x4ifK8XRuZ65XBbXdSJVhvRC92YEAabPrVKbcO"
    "Gg3J1Ybtk8GzcUF0CB2cJi9DhTzV0KXC133z1z++9s0fv/v69bMv3v3r62c/+ubDn7/ZdtB7TeG93PBeingpU7xVMP5vSh2nMn1s"
    "EUpZIdbIo4Votkwzl4l6iIy3Dm7EEWdP82UNSol1oF206YpKv1eR5KCzmZodxBBsac4vj6BDbuysEJ6oFGNnMWpibWPnUAhwBVOJ"
    "D0oUAwtKs4WVtcdFmGHDTRHakVS2qNCL32avXM1OyV6qmo3KdJ+r3rcFFAlKwbZ+RR+xqTKL9RF0yjf/Oe8dlNRGFpoGCP6hbHxk"
    "EsKBGyVN52Uk1kNZ75E7iaAsEUjvhypvRhaabtCSlh4Kz1ymcIxNQjCdDkBOLK3QANKY6Q1E5ru9DS48ZOA8RIYf6LLwaywH7vgb"
    "ENY/ZN5arWb8jj40Andk24bKnUAeUiAbJ3wgCUzAKF30qun0W8OvXuXahw4tHq6CzigemEDDigeHYHZlNTrZAsge4MVI59mSOrE5"
    "2YE6DGdj4JSqErL8g2xVUUxEyA7hA/g1FduQIw2KAhd0kyGD9yEKzVBVpZsBYg/u8lrD+vL4J9DuDyWchcbgQtSAtPYiNCl82P70"
    "DioN+ZFhxXGjDFjzCvYCKpiJQcG6zGopSP+EvolIVFBSDRKKLa6piWkwfI54BHhJhqQZnC8qaaisbz7Z4lYvAmxHVCcWUIdcyzwj"
    "RIRZVRTJQm3m4YRNF8E/JeB2RBlCRlki88CmorD6j+zZiHKLeeo/I3jSSupoZJbOMvwmt4SIXBQCiSPrF/L1DM0mTpG70MyrSwVM"
    "LpJXaW1LE+dqYqRXGckgCnmRuyuC/OfwVzJ6BzMwcrs56MiNg8gIvghEHhNCYlIBvkRSiLN4fOQal31jd/YJUlz2o7VC/K//7trO"
    "vvzm67evbpigHLgPcgUyn+ahqZjJAsM1l7HdJyqzHDm2GBvQGJgAGDuZ5XhUyzj4E/CVAeVkwaCMCIIo0jKIssZEwUMYbQpBoX9J"
    "2vAzsAVBgdWbBChkkMXKRNOg8YI/zqQqya6puEDBQXGV8DsiUVEkMsjtWQCSCErrR4X1F0dDAlP/QgqkgsOP+IQyWFyZhcDsAkVg"
    "M/m9OOLIOHhRigxOAS86ilj6BGIR6ueQXYHAf4pIcdRIQUOxR5cBXjOMfrLAxf723Yf/2KeFWYkzYJygnx7pkhBptr+z4CCY/gnQ"
    "TW2/CXRvKgWcMmPkNIEyw6WwK5jDHLh/2pUXHeMTYmqqPZ/II8gploDmUAQ6WVFqmaQJ0SJ1KgVtGIl1OdkjCXucC4AnMESdcE5X"
    "sIXANfkIgH+WvjUctrUSWj0B8Che53xoIVT2uwGnryG5Fs+2rPYVd1+NnVAtSXViQzOmsQmNEs7AnIAJ0XIfrcHNManpPqqrjliR"
    "g19+/6dbzgDZQ7r0MHA1OhDRpOrT3Dq3J6XMGrSrvgh8TJxXELI94rwZFHSKBO0IcR/BcRaE4icliy9VsJAEHeUiyC+DWH8j2M7A"
    "ZM9rB6ShO8+lqTNThPERdFTX8/NsVzIhabc7vzlkGWEwbkKZn2Oxf7m6IpxPQdj8SIqkA5sceEH0V5zaG38aWFF/wYDgnwB/R1mk"
    "YBYDFUjWtTB5SxLihUEskpbmX03CNxJhGahoE4yJRkWYELAIIkpLSKsNld2UkJlkJUwGpdNiC6INTEjIDSpQkODcDb4wEa+1rtdi"
    "jpB4uiG1ixD6ROGFY650u1MUdAlUA+pDE2y+1b5cydIv55IW4VRCx4teICFHNyhJ1nxLAVMsUrIdWsRnuqZTTYKhGuTbMTKjNvV/"
    "AkrOT0cJbhGfjHbIffHjQQLPblDSqzm+TP13ty6skU3C8eKYex6vgtgWeH4UiQXOeuPVz1nthaSuhE7Nv8+u1T2YGiIoixjHQedx"
    "BFeG2Lwiihc4ODOCxyao+pBgUcJaRbi09QURwXw5z4ymyPbtluY6UI7+oiQWzfLbH/14dUK//WlXLE9LZ1tW65Lbl+N9pW6LeF/g"
    "29pf1cJdZazq5KZpdiW0K6hdeUVO8KbjHsa3SDp84kbQ5anuQwBi0nM4OpUrz1CgyVT5CNr+DqAhmbQwuUiYEot2AM+rOCkCy5AK"
    "dUISbOUVmN0opgorB5JfRH8SEL11KAq8YyzIE3wWEEJsxpamFUs1BJXcCIh+DE/esC9QbndApcThiRvm0wzKgwuwEoOqRQTARYLs"
    "X3vxS4iOUTf8hUrUshUmFUkV04CCcUZ5fg70iU+lgFLxVjyDaQc2Q5pAT6SWEhuznfpeome8oEBQVOyW3xJliVmnTQKjgAJa0RtM"
    "9ii5igNshUiE6jr5lxXz89/+4loxP397/937PWzTQE8TJv0m8HbBi4TYVVP9V4y0YJeY/jgwZyUXG/eJ6fbFb6PHU615k5jbgKu4"
    "6p5IG7+ueEMDsV3wsgEt61G1br8tHplmM6bTMyXc7CynDWPgrWgd5GCLKzs4F2NDumEY7l8E/Cd4/H3tqaWDf7weiH78zdfvP7z/"
    "49sHOPk+2dOzf55679a3e8ffR2Ufs4Jm9yVesYoRYp38seAnq+7FbGAzm950kqUld8VH3HzEq1ziGoVKcRoNDxIdzAGvODoHp1Sp"
    "3J6cMk7N6B4DAKdn8AKeqBqDydKvagjXzab7wvQyGwwrqCooYx/m/qj3KbkM2Re/+/Iasi++//YpraXBH9pIYhb6gcoz0f9WYMfu"
    "ArUEPUjLbyykd8v8rkiFxQZo+XsgPsakM99+n8nqlsHNhFflc5OLYIwrn1so5Zvkea1ZUkcOKnPE4fbwVEmQKM+UR2sgBJ0NG/Im"
    "sfNlVqLV6YCk3fJYWUYdUiqLDo5M/GfuOkDFHkOwwwg4kpMOHJamqdzohJhtFkY9dfVD1JGwIEO4ypZkj0VZrsjSz/xVRLa9X6lt"
    "Mo2Dk3cy1Cf4PbLbqgiQiqAwMQ1YDssERG8Ws2hCmqPKzFqLwZ6uKxVp+W35NC2cjFN6Hn4IJJH+0HceIN0fstDP3vL4kOi2Dr2g"
    "qGSbW8KW960cBjvkRuTQu+1LSUfbgIK2MBQS+PMdFJTA7FPcIYX4UBE1XYKnzgNrCR7rIphVQgyyEBt4Fwpz9OXiMtKqRPPDrqHi"
    "n9u7EJIM42FSKteb/WgMv3aRyaWXJRnNSHY0CESVd62CDE3WMLnPPBLCbP6icArIr9wcTrhUFaWIafKxRWQ/W1FmRRo6aKqa7OIG"
    "Wrsoi9nAl4JVRCek4hsaEvEFoYtwlhcWKqTfsbjptkuYo0Cf8GcH6NPKXUhQ6VUhDXMr5lm0PgWVsO/yTB4oxFrcz4h6xDo8FgjD"
    "HxlndVQ7MIDhyONNVui3eSyGNXKTjhNlzLIXFqZChQw1BNXGyrMAX18VcRtz4yp7JwFg58/L5MKRNQUIIAp0uDckgVRpaMGxhK5O"
    "rKk7W2BiJV6NIctKVllrYMy2rx4SULGYxzrWWpWDOLGmq3KjzlV7gAKIDiPkJlcdH1JVt/J1APJWJV4m+GyrUBTZR4N3olxNJZ4l"
    "s46caurlpI/DTpphplaVhs9g6/CvM1VsPZSkswN7T1qa40b/DooRVe23Gfz/VTY7XYRV3GpTwukis1505dJh6nkVkCCzbrFgHtBi"
    "ILEie0ojZZWYYjLr5Yp1JbNGi0hk8Cvnl0G1PJLBgBSlijIcBb+dAwvUl3iE80ngGiYvsVxVwbUKV6rTP9m6wCWxS3RMZShf6yiW"
    "ihmkUgh4JBRU7SrRMvkSMOud+IBkWyQwaSrXMkRZQS4rbKIVG3ETDJb0DE2FmawVqJqWKbDIGdlGWF5Rzj67B5UpRTZhNV1lBLSo"
    "gr+kNfK6hkkSq8rArrO4PsaDv0sqqNNEWRGxXsjZFFUCm1QXSCeqKkDb5joTDQonH4akAX9dZZY1H38smJY1eEOkGB29idltLUPx"
    "G90IZrCqt6HsHLq5YeiaYA1Ec3nNOZS1hFREA4UO06U4ZtlPcug7jRkKB3kF0AZYb3O+D6QHNVEbdZTVlU+/H6piXsRixaqlMLU7"
    "uDiblEZnI5NIXbAAUZKa0oH6cyTbgpPNPgOvg83dpFAGnLZeim6YD9AqAwDnD45VL/87cG70opwD9lTTlsd6RF7dyrIAWp0FOVHx"
    "Ms+SxSYOlNxinPFgzUduDcj5QHkrHpMtuQLFrjwvY6CIFo/8B+o2edlQFFC8avUhZ5RVQIMyXBI+ZmZfWmVKT+IIcPQ01f22/JfG"
    "en2e8RJQpVkvQiFJp1ZAwMVE5QZFFUpkeJcx6OacDYEVhpMnlcIN0JKn7AAr7qWtkRjWl6tDjwozMW/Wp1aaXhN1uWURsnpj8SzK"
    "QbFdiZKuDgK3aluYushCt4qaD9VyZPbcuQtHnliGJ55j7nZPf++suOmp86XOOpCWAY575RyB3d08eZNMsM2zjml5N+F2Aip/zCVo"
    "lA6spiuWhoTnVqdWGGADFMlEU3VJ8T2YH7MdzruAdDrnLRRlp51nCS04VCudCWaoYw9VQhFlmqqnxdXiN1NsUkLkoMAf0P6MrR9Q"
    "e+5vI3yx6kANOkBUQeVFq2SJ/YYRe7BpVR3N7chYvSyYnSyripfg76wxbEj0rQ4ysGNmgA3BM2gDaa2Oj3Y1QTz8BQOGCvEPqKU4"
    "uQaDis0r5GNiBBOhP/hUh1WQ1dDgjq06RgRCwWrwjwaksSpQZ31AvuksKcH85UfbR8KQZVfDp+hsyzYQ1vOqWmHD1PtkFAbRJ6iY"
    "OQFAUApS1aAhPmB0Rx9xo1xtmkqkkirNkScgXyrVXwSTtAhDHUg1poQaXAT5l6AcAIIVx3lUmJ7GFodZ2EHxFd3lh3ViDCFZhB4N"
    "hzFOHfDd4mwmBo+Bkw6bZNYPziX0nZKG3h7Q/EkgxGVCZEHZu6JDaxAzWxc0pEIFFh1ORT47x6dy+5fT2dlOo8N3oJuMwrSpL4yJ"
    "1hOe66ETi3Atxt0yJ0mBvVM9S75kDVfz7zswmMHpZjoYd7uzqJBlezLViMT7Yg8qqk8KIug8iZ2NLiljSvHzDGcLpt4sxTXQb4cr"
    "uQBGNk6whCro+dU1XuDJs+uXdiBWTV1QltitcZJe4Fg6xHmRcPA9ptjnMdvUIVjzplRQiJ2u3wxfvtdvCwAdO7dmkNeiOikHw87O"
    "6pLAO2DHAye0IKOmK3c6cLpzjgxWp/KNAH6Z2C/KEXfvmNguOq3/3yJpN/OnJVZV095EJk96zrlV5Um+ElSJS/ECMoMV54AxY/Dq"
    "BySUeR04EXvkdrFzoEOp2s2PgoJ4k5gltVmZIqDyEXgYnYFhjE0sKJDHuYPDb/b9JpKK1Ce7nAWO44qoW1x8N42k9vb8VguswUnT"
    "Fo6UqRcs+A1x5mBHzLWLSQDKqFxMAkXcZ8iyhnKaVAItT+pFv9mXdgAYyN1PEPEFk8akw99VnEmkkmvSbaEMl1d3KpFRl99mObbi"
    "TH3Gk7KnK/dZCxEGGvpRRqQ6WdanjRvELpOyw+vGfjvIveh8D5b5CzF7bmoFO6J+iw2tCFJg+a6Rvrcg8WjTwYYK0FDj6EdWdHYM"
    "+OhwGQlhMxBWdYU/uBsIvzdwOMza3EZRMU5oG5aAtOYESoG8np2WPRRGocnPJ3aeBhqukHu364FQ3L05UeiN7/GJC3Lnidw5JHd+"
    "yZ17cuel3Dkrdz7LjetyZcG88WPu3Jk7r+bOubnzce5cnTuP58bxufN/3rhBN9rQnVF0Zxu9M5FuJKUrf+nGbXrjPd04UTe61J1J"
    "9cayujOw7uysO3Przuq6M75ubLAbUeyNQ3bnl71zz+68tDtn7cpnu3Pd3nlwV47cGUOxiMqAo7ALzcJ0d6uY6euso7z5zL9HxWMo"
    "1EEHikwLy7Ovs1TyLDLOfHxLuZ+VqgfyE7rU1sBE6rKa1saQsgDxzTZ1EdgQmlNSWCUu1kEPbJCFyJ2S8ACBaJPnCcrmKte8PZi8"
    "VR3nHmeqsqAPztUyNljs2ennrHilnW+7n8QqPAu9XMZQ88MDAUJNMFqc+AK8BH7zfC3awfocTSUeApGPrTgtJQ5NTfhgEDpC5BaH"
    "3RlujCYxwY/hfIDzwfxipFo05ybLCHA2lRcASSNdKdUZHXFc5pZusTSTuPvTsXqJUEUtuR2xvAcxvKx61zjDm5cUU0xI0oQ62c2n"
    "LaQySZ99LpI2swtJiT0D14QHxSWG0QbgrNocyJjos83WJGdin0HCfkyXbJnlu1HmFI8UWna2374nsjSNA10O6dckLEFFLIV2xNC1"
    "LrBJpNZ0EEyclULI+uKHo4hcMVM2grPM99n7S5JDjD0DF5vQ4AWl6Xz2kDGhCbpTg8qvR/lp4SjKYkyPKMWOfYaO2aawhXmHcWe5"
    "aG2beAIraio35dRW2Bw+8QusiqZ+QVUjzAk1M8Nxg5lYwDzSRFpVELNsAokVmHjuriwwL5qqeBWQjjZF70zxwneiOZHhrsTYFhQE"
    "tA1NWtvUhhj7MFTwLuoFs2u5OmHh23HFWU/rjNJBhOXXfMlU6HXxcYIeMk9O1KSa4smXUIH6nsutIALb/ckFOjv44auh1HfL1/Lz"
    "Ut+mugZu7nNxcpvofhVlr51AN9AuiS7CXqNyyr5/DteJFXuys7jWcJ3cUF8ZR7d5FVM2uTbxflP9hwyPfvOKEzZHw1L9o6HchXnI"
    "s8sFQRgVS2FWm8UFghd8GAg5zXoPHbJqqZjzOM0oFGUWM6kuq5YKC1g0OD9r9qpEDYNXs9d2aSAPrtmrgzSsmDrr/pgrF4E/VQJs"
    "MCmqbKIHQQ7Vk8/hskLIR1UM21DQjBgZ+JTYfIiLI+uxerjw3lI87LP3svq9q7KHCocsyIvHCslgG+B9VP2XRl/rJTcVLFH1HJaJ"
    "aGKX299k705gsfFZCrLOLNdzQm5KVuowEajzxqzj+IzQ41Da8ozXG+ZDEHMWxvbYYsFToAAScqKy0l9SEVP6ENS0AyORFMpMOBMX"
    "SSxofii+nllqw6PtaVZ7TLMSulD182MXiMwPf75AZH749pe3r9++vcGZOkDn1SFKCJiB8U8liA2a5oVSGxAp0RcN/ZLRyz2tj8Lg"
    "N6R2xOZQqYafDj2JBX2FbsJRJs5c7maLMVaHhvE51UsfL4/le1AWatYObohnxpkoDoUevc5QY53Y5Ngi1h30CpuNZQSVQ2SilWXM"
    "1wfPFy29/NNf/u3Vyz99++pfvnr78KfPfvHtDTlmFDzIKha70QGBCqAMEiWo1lOBw8jEsJD72FW/fFAe+jFzp6OkXrb3vK6J10sj"
    "lk/41a+/uD7hV998/Pe3/9ixVOaWMfO1KkeO9s+h/L9BQ0ZpZxaOYGFN1Gs0g3koMfHsXJtiVblqCCx1T5FDRKoqRS7EaRAEOE2b"
    "O00BVGptolErniISV9gYqggekFJvJwa6DrDvuocfd7r1FRDZbu6KXD4WmmSFPWzoiCfgxIG4QHMsCQIbQelsAZGLrrS3yt8NAasR"
    "O61CgFbUiepuplHKMuHyLAoVUave2hKEH04z5hqo0Z18GcAyhw2EpvD8IXBwa6tkdZGYgBpQ5LpWDzxpD5PjDIHi7C5ObJdZc4BF"
    "9ESsYVB51DKi8wehR3Nez1xOm5PK+kQlQbHRIktxODmrfZyX3Qtet41OEbgUdChrCBcos4qH/67aFcA0aYizGgGVj+lVVbG4qzCT"
    "wqjYxuW8CFGbqgj9AfVomqNVKIfiKbh9BowiPMdNjE+EsJskfHyZoTHUp7ZLQxmMoc2w34z6ZQF9DhTQ8ms+QXFAQDSpK6U0onRt"
    "87xkvl3pAoC7NVWeBuWk7cJ5Jgq3MZNPE1YLM0NRu6IrnxHWXxeRq2XxQDNUvTrj5MJsgqVZaCYIl5tykSPKq1kgTC2zxdMErw+w"
    "19vMfbaDo1AvhN43udotBguJ6RPLGxZl97MvFg6cn3189+6DKexdWWdEZroX12OBgS7CZ5AnvHSx5xpVwnmA6HJqGRPCgavE5loe"
    "Bi5CO0dkjFpRejwoAkHXheQD9UqEiAeZ6dyl1q2+twnE8BLS1pU7b5QuHW4U7uJg9OxCgIIwqeEi76Xm80JvLJDdpxmGEmE9uJ0b"
    "JXoJaTM7e5g1bW1O9mOW0NNBNfmTj+u4+6DW6Yfba52zjRamsQi07qVwh3S6ilx2uUbIasPUdz8eGtNDxYlQlvJw/4Y/CSfC+anE"
    "YFRdtNWW2dkRZ4/m1Q3JuGpmuYsNPgXa7xEVp5qoHR905l8icoKazx2kmzRlFtlQRRwGq6QGzEvyseHZkIRYEfFMP1Q+SBzQtLjw"
    "VhbZa3pUYkk+Xi2CIhFtbUYLTq5F88w8NYKoPljEvPlIWYMZ/PafcmMhP0iEMeIILuPtIOwo+Jxt2HdIDpIQD3QYkrH5AFxECplE"
    "GJofohJ8j1X0GA/B/4oq8oGYB0cStCLztVmWKXgJgeyjiGJeVd4j8ASyUqaLGRHN4RfbVXUTKGfsQpAwPhYc5nNRVxtEYxC7i2wT"
    "Q+9CbpmIuDyh8ogsoMAgycii4IY88BUCG7wCcKENoFlsJ2Dg87KkgZMpeZHgpb7KSrI2SFU4Cxw22Ii42Asc545kI1kPNo4mkXgy"
    "MVIijbMOb2JGF4tBBARnmCTR+c44o7rTl+U0ESAPBA1Q/7HpaqDnRD/lOVA8ahP7ZsxfnfiU4q+E24ZniAKWx6Yg/aNUoX6qE6AZ"
    "xErMBGBzw9J385re1cmN2abOqEhYNamLVOsg5so5tOCNOupFgmWYsyuJoTVnQCJ2SCUuHqz/6XYr0h/ypeRMeWDfmiks8G7nyTwG"
    "/UilXMEj3FRfAWyX2NKbs4wd2IKDk/QFeg+jvs42nODpHYg8dS/LXuFF6ZN70uJq2EgmRWnApjMuCjvfn1ha7/wpD/oFdF1dJQBt"
    "pRx4UNUyqthjhtZYwQ5U/eK1663b8JpA+MvlXPvb919/8/Hxy3fffrdz2X2qSgVIk2nkMz776B5yrUij7q8ztAv25ckyn/aLeK7Y"
    "uFEu7SEz/Kn6xa02xlq/4vqg33/+05WP4fvv/vLZ5//88f0fbxktFc2wU0cXyUiEKJwQ6eKcXpv8OtHBF0BDTV4lEcEAd6ajGS6O"
    "Nv3Gk9uNXOTOm2gB5gZxyBgEaZ18R9xjH7HOpMZOSjmeewI5FFVz3vY3itHTOUnIODyNkkxPnkceIHkmeSN/YtIRKvJqFZPIIbZF"
    "sQdVtjGLgQRs7xftCIorkVY4itcxK4GVNLv2fKULgHdDML7YSbwhlGokD3eU5WVUHxmiEhsS743C9FeSX0I1J+xYk8wykSQrKh6Z"
    "AquxyL2ZyCAUxRmtWjNRRZ0SmeXw0RQPXkWfG/yBxJKHGEyMDSwyxSmx4FesMw/EmESicKIqUxPbTJI+2jWYidWv5m81mMqsSJHz"
    "jQTmBlXh1Cwz2cHmYqxL+4c3aUAY6sRKVsGs1lvdpeSptqSfSipNEcmfmOI8kGcSk0Udtow0KYkIPOIUCFGTwiilksC8kaxA5rrj"
    "bKu6OERkMCB6evGgOOJMUsZP63SJwBKqWi+DNNBaTWBFzDwek2sreeBvYI4kL2nQSbSWBVYhu2NSrisraCQvWNNJ/yz+cTpp0BWe"
    "Ocx+YwhRV52LXd3oGNLG7/Gk14YhiBMNuYurxlr03S+//PWl707V/c13f7nRqMGZCI0k14rmRKwL4c3gXAPrKZp0+LgPcp8qAwdU"
    "8orQRTFli9adpU+8vbGx0s7QcifDvbOqRkDUTKwSrzYuH/d3P/mf18f93bv//f6P+7c9DHdzbp3J2Q4NaXNOGe//h2FrTksnRT92"
    "BWS42P085BiS+mUyjwNnfd4evBb4AUdbCh7MODD6n5B13jqQr7PJAdd1O6J6KXhl9AOYv3T4+Yz0+2lW7z6Q9xvdrDRUzqlg43DP"
    "/QHzzzebh8XGT5s1ugHjpPXDa3UfKNQZhxfPPuB/iCLYBkwAj2MwY4AzwQYyS0yXB3uQa6OKK3GgI6Kbc4O+4eI3IyXIdza/KuKx"
    "B5n+xccPqb9E54odjfQdMn94rw4cAxtzdCN3kN7De3MgjhCEwTIxIiuYBprhJSwTlgGUkZWNzRDJQCzR8m35JPKreOruYAp1lyE4"
    "yHugThxB7CD0x4/D84ldVJuaxAwWE5rWA1aiQdZM6nbkjsJYPTpWzGSw7YjZmFglHsx8R4M74K/Rp3GvJE3RYcvgTt3E6L9lMnX1"
    "qxl58zyV9qZkaq6Q3tRRs1XH0hWWjQFJQmYe+pBoJB5Os9yxssSiYlLARZ5AOqjcgzyC1i8HeFV4JBrId75L9ExY95dNPE9twbmf"
    "iQgzcN6QWE3KczAyRH9pxtXDx5X8LRLJReEOkYFAbig+/W1XNlWTlnuzryMrSgW6hCjRGC/8sD5AaRpY8dBmKckUeGYaKCISVCgS"
    "M5xPKpr/EaI+Fpm1wUOTgwxeVbpkmKMoKKEES7+DqCG0qTnQ5bW4bIQEzcOm1DQmd9dEpC8YU1EhYd31INzEc/GZ4stkFigun0Zo"
    "VP0nXjexTJHzd/46kh6i73Ke17E2FIM9ECNwcxFK2UJTMmYgW/QtTdm+ErGq4ko+X1Ew2xPObo7OvA0MJpVbnNcRZFPE2kqvmUiX"
    "SUCN15foJLBKuYnCD2ALCt3kMbekBh3NWWrw0HM2xe5dBxbQZU8IcBQlhZSww50n4KQkF8rlJXl5CWQLNZO1x5DSMU03ZgAlaGKY"
    "AeLgDjjm5WaPH3PDrZD980Cia1tQ8ded+iQOhwMEUE5EV8SopNdeYl/ut61D4Rd83mB3JJete5pvcQHmMayP2T3oXv9cUEdEJ7c2"
    "2XSubHoMx2BkcrjcELjs87opXvk6kU21bF422BadVR4bJwfmluAFLDAUVUaMk4N0JtGfx7l5ybg9X60r9rpjfgzenufH29zJjk4I"
    "yEOJZZk76QoOKzMsVl+VLJwUqy8E8H7G6tZJSIphz5E3khZVv4bpdXAk5sQZmMhz4hTc7m0zD2Z09yZy2DCu1zQcmDfCcTBSGoeD"
    "GALgy2n6xAMcisn9ZChrBUvOG78YimtN9y/+11LT/eP3f/7+Fq99rl6D9DOwLLtrbdCKV/0AlKFNvtmWTCPfN27k8bPYjZcrsPN2"
    "8jiGyhWkNJ1BiWee5g4/kK/nVy9+MHgGGl47ofLUw+VjxQx07vHKDCHvtXLEDZ//s9I699I799I8e+mee2mf/6R4wg+/+HyBU3x8"
    "++v7r269/vzdt26599req/devw3KfczuY/o85ntNiHvNiHtNiXvNiXtNinvNiueaFjmtBTVYfTVmd/tO+Zg1MVSwI6t9XrBDIpm9"
    "5d8t9BIkfxpZ2C2C5tdZPCS5O9hIg90XY3LndbmzG70+8zrngHM5YY5QDn6/nTGDOEjxe8o+hcwlEBwZZTBCiGHWGkmkFXcEh3lW"
    "zOJrPmON4NHZ/AvP68FLEInFvHolCRZTMN013FFbKHvpE1BeZ+cHOkhynZ31/EDrkrYf5tqbnF22emoheeEJFocKmsZGSwHpkEeZ"
    "HN/Vi7okyqo4g1EMvlHTWQ1Gbb9Kgu3iru2Esn7yZVfMEI9XqLRSOJzYiDgYJ41EwQwSmV8e+EE5O7l7upjAnSzebBev+CHS8Vmt"
    "J/dZrx5M9fj14RUQAusYz4oQVoRAgXbw6sMtnJ3uylwWw7mxEDJ8+OGssdZNu37bQNqd8iL6ukT+Mei300LgX71EUGQRlatuQ0cJ"
    "5NjWMhaOw0I2w0N6GTnFqJQ3ufEhzpINKN9XPPQBCnYPM5C6+PDqGkinfBxX8Y7D6qQWH4Zs9X/bwk5FMIGYqw4vYGI2aFgKhFix"
    "3ua14joqts7qblbz7qrnZQX5mgdAQI8bvEYRijPPInHJ6LZ1J8q2u8lpU9G4nYNmdbW6wcnnOMoGDwl2WNPYVOwAil5UWole3iXD"
    "RlSpG0AQk7cK1Gjp1csggaF2VkYz1J8CeBU4UVegFRSWvtlgWWaf+sToShNXkOr4Ua7aAdKhpJXZZK8uGKhWJUMKU6L8iXCzTF0+"
    "yGmbHXNWjCFXug8jlTxQiBSZqBVdkMseL9hbtT5ujmM7UO3Z6+sZW9+lIM23cW34RnsqpVHgVRAVIwJ/YA52U6KhGLdvCnim7wiH"
    "l218MDHX4RUFFAjR99HCLyhuORWUePRr9rPiG5pZe9JOFtc0YvbgQVyzWj0aDN+Mh/2L5omsgAh3mW+6kVTPfikarXD0qCo4tYNH"
    "fi09R5aGhbNfZiwU2ake7x4sXUn1kMng5q82rOA0H1hxx9VdZj0sFZl9MKfoEVxsojXn8TKzQoAK3KOKDainqR2yF60PfhWhAi3m"
    "nHSVXcXMGCsIHBx6g5t5TE6sokgvVmIETO5y1EpE8VKOQII1JUr5BzG5D8cTmC2Koq+OU2BRBupf5heb7vbnYI9i56bGTcg3mSRL"
    "QEEIayEUpRufJJCGXCUPbcZDshkqQaEs+2Ds/Q5dT4ObXPMeEGm8O7dIQQToi5ARjXbMkAizJ/jIKJSkXAiTvYaUwyGOtsrTZqMl"
    "kaElYcNp3DODg1TaxpJXINOdmTFcE/ebWWpkVknLnT9XhAnXGTUK/nMFJtk3mfT5dt6Kc/6itIxjLqxrklJyWfEH9nb05dTCf3Jq"
    "Wc3+Xy94sR9+89X7f3t/L2x6MyyeDI9nw2Q3XO6Gzc3uuVlFnzCabjbVanHdDbKbvbYbczdL724I3g3Fmx35bGbezdC7mXo3Yzcz"
    "92YFf8pI3ozou5H9ZIPfTfSbBb8b+PcDQKe5517wwhIMYXjNUpZkCWIsQN8NrkPv6MjnlQm9YKUi6t6ncob3cof3coi3aon3Yor3"
    "Yov3YnH3YnL3YnPPxejuxer2Yna3Wnf3UnhIUrSxc9MSQ6kYtWhGTU5uTqLoUL4WFMZqlgKehYYc89Kpc73tULHxslWv5bks7N/8"
    "ZEG9/+bdx++fVvVi/96M45vlfDOrb0b3zSS/GexP5vzN2L8dBW4Hhdsx4nbI2I4gtwPK/fxyO97cDj/PZ6Pt7PQJFbiPwdMQ3QZw"
    "H9/7+H9ifmzz52l63QsxblW+7lXAGhDagKi2pXhfEDELGIaliGbFOCouDTALA01F16rv+bREWqUiTF7ejovR2ayRdjKo/9n3A7qk"
    "exGzoZLtSoDhDu8IwO4V6hVFC1Cyyt2zqwPVmWhcdZY9dF+rwYBNOvxe8x+9uhBQL4Kf0EH/9PBjSUfY6eEeuc5aq674G4ELs4Ld"
    "UL0lFerkru+RisZCZWnJOcqo/K6eY3ULHiZaUQ35Mschwcz0qwlWp9ipkb7gpZVAR41qKMXZqaG/9R4gLh/hKoIHs0f7sYodxou6"
    "GDbVReRsfRqvSpeoM+uPAgz64T6/JsCH6wKVtfQz3KVytuJza9HWb7765us/PFkg+7l2P/Pez8P7WXk9R+9n7Nvxez+Zb4f25Ty/"
    "n/VvjoCbm+DZiXBzMez6d+2p53689fI2BrcRuo3fbXRvY3+bGfu0uc2p24x7mo9W7EbqFc7ZwxW35dS9eGy7gXj2eHVdUVEFzJPf"
    "Tk12eJFSQICCa42OcKXGsuG85lDOBkvag4lAlr64CdgA/2GRPuTHmWMi+SWLP8iBbNde3LQ3JKv5A2q70uo8ugjQ60u5fmaYZG8X"
    "SB6aaz6gcWblzAbYsGNiSS7odSpRns5j9vaG8jLp4SvyH/x3lcUT0qy+ifoYU7IqC61dlT3D4XTtOKZ6/Q4TI6o6eFXPiBIPPisy"
    "bo2TvZ5FoVyMKAQy6mUdBEeDkOs8XkDgADEsdWG90CEI5VFtpLoiirgaXKyoaNJ9Rx8Qs7PzH1Z14Srt/uJAcBKJjstlWKvXlEAl"
    "WevMWUE9VdXnwiUbu/XG5lPY0APtWmGW3yxtQeb3dm2+RourvuFO7thrzODq4GScmqq7vgxDU706MJw8/iuU1/PCuEg2MB+Wazs4"
    "0/wRNXpVJj3Dt2STXpKbFw1IFHWrkRWbs8Yfb8HKy4Hoane1Kz9fKif95u3846atb7P3PrNvs35bEftqWVdSB4mCV2HFCu9u2uHs"
    "MosUdA5wWvRE96LYONXI09FQ0qpvZoewxzAPTEq+/ye8wG0DEw43T44XP5L1Y6kxghttznBMO7DC/jkdPNEOl+mAxzquA9lKsxuQ"
    "nfTi8C/kRL14zYEeVSBlSOpxk2wQCI3pkeWFuJ57ZJUVtRPQVl8fHQkj4j02CW8f83fJ67ZMKXtbUACu+LWAulj+lBi823tQvRf/"
    "2IZLye03G7vmfTvsE7rbZKgWJN0wWGYluVRMC1aXjuilWiBVr7SHFHATpmFql6Irckvh8ujJNYlXW+WLX6y2yrffvX32xR3H/twV"
    "ezftXbh37971+7DsQ7YOJ/gsvLZIt2zvF69DYol6Lw7k6ImFag6/EV2o+ZKhQLML1b6AG0fPHCOaoT2rlNnht+Ip/LrCqkzUVR2O"
    "neEv1940XMJDgksqzJSvW4Pnc/XiFRH9Xm5jLhbU0YneVjxIs4tF93zL60jjCY7lA/2zlwlGd0CaXYUmUFGdIlrgAubXGLeJv6ac"
    "/+JHS8r5OUU+vv35+/s8eXrH2oBb6+5Nv33X+tG3Hrn1FzJ+gtCvnXjM4HOuaisPDiwMKA02AY2sFKaJ0PAe9/R2AD/Xq1Z4RSxj"
    "AEMG3DzRjyhZxaVn70XRKiEYkRcTko8/8kanH65XFCVzU/kUWQorX/PKN0H79pfgkZtevGrWnKyor5W82wzumK/Je1D0mTQAH5zD"
    "0a9acVovwc1F46Cj6FeJNWxz3fVNrLgqDCao5oOYi0wMQBcekvgkn7ERD6r+ywCROrHThlNRCigJdGKZIgpJhalCUHHKL2aOe1xa"
    "KDJxKBjYit1bUVGFtPqTWB4s+aN6mFbmOuPX+k+/Wiqm/fybD3/6/uOdyeOpH26ddOvCrYNv3f80OLehuw3sNuy3SfE0ZW4T6jbd"
    "9rm4ztN9Dj9N8Nv0vy2O29LZFtZt2T0tSuvG9OoXGyW/2OKsKgeQMV7bHTl8rJ9j22rwkxRx0SH5azrwyZ6+2rEhB6cs6QORaA+2"
    "CnU8J/XQ4GlmDp8GTU+uwOkmfy8BwFKhqHIapnMIZSGLmxNEWHuspsPKNtH7CdXTuvcxftm8iw2TXK7x4Fv61K+AJAffyQgk9inB"
    "n3afToFX1521Th1f0tLe7E3KEo8FZtzBJ2uUpnFZ++7AWxbPWgH1l39/rbqffPXZl29f/dvbn775eFt4T8OzDd5taJ8Gfp8V25S5"
    "Tain6XabjJDCRLFjkoe6NGIuAWCBg2ezd6RWB6coGMcr11Z2FDsb4bOtYblLK8LrG5QdB3Hg2+t94q78C79daq397Pu37959/fbV"
    "bee/o8/v0PQ7bn1Ftd8w7zdE/BNefgfTb0j7Gw7/GaX/LPX4Cez/LTNg5g0ccVnB5VIcYxcxjXXY6b4K66fX87bab7rgk5pi0SO3"
    "aXwbyttAr9PgNkmeZlDlNhvUURHqWJkyLKA3n7SM+xqC/eXvlxDsu6/e//Xd0yL8ZA/PzIw1beOW0/GU8bFlg0CY2jOEXWS5Q39O"
    "RFXd6isdqQfrU5svGwQ2gzNjdbC1Bee76uCgCe7r6ChJF6qfZW7T4zZ5nqbW7JalO/9+rfL99+8+vPvr9+/uK/AODNogQyuaaEca"
    "bSCkHZ+0Y5d2XNMd87TBoXak1AKi2vFVG/RqR2XdAFsrlmuDee0IsB0ddvewr26gm4dodx5tfqXd5bS7ozZP1e7E2h1cm+9r8Yrt"
    "DrPdmbb62XYX3M09t7vuNq/e3eG3OwN3P+HmQtzdi6vrcfNK7g7Lmzfzyde5eUJvftInLyrPhiUvVTo980Z1OVWhF2La3nuY5+Qq"
    "HzqdEw2Uv6p9IEYHMSbhlaj+HMPl852vgPMsTMc9KqSFydg2VIQ41CtmEhz7xkCq163GmKdZM9dEHopLmbPvkqruFRzDnQi+RvqY"
    "laexBnFvGheKJHiWoq1sVmsX1KH3WVjbdABLemcXUQFekVzw7Hh31oMFq5PjQWqbHkbWu+ivEzcId6P/rMAPOn92TUuriPHimpUI"
    "E5U9sndbya/ZaGNgkVYFa8tLcaf01H2r2fK7/7WaLf/x9uGmMe/owhV4uIMSd8DiDma8Ax1XEOQOkLyDJ3dg5Q663AGZO1hzAXI+"
    "gTx3AOgCDt23h7337j27dPo+Hreh2kdxGeF99AuYmZUkQJpf7ciFZQ1lJoIG7kUkgg8S+3rWUAEtSrnA/EYOp2VtwCDrhPk7IwTV"
    "pLRn5rlR2UDOWJnRo/RVsiQmTwSZU2M9XvxugZ98+f05k+zPfUrtiNINbHrHoa4Y1Sf86o5tvWCv25Tdm3z/nOVL752wdtC98/aO"
    "XTudBTL8dagZVK6MIOOccuIZVPH206WlmZiN4SQ01pI4EbdmKtSZrmIfMNlqYDgou+Pq15XIc83v+enHtw9/vA/Hjrjd0bg3oO6K"
    "4d3xvXfs74oL3jHDdzzx/gX71+1fvnXK0l17TyZUz3CUMJj0inMvBZkj7QY1Nuo+Y4REUQpUPVBtD9SWeEWxChYkBEGt1wdEJQHV"
    "bULGu5VFBhO4SvOS31tVe+Fxr5hlohepoN6VpWNk6Vm17RD6NeZb/gQFZRqpIMPkzmWRA9HqwuOTBeQCgXH2whJg0U08/pHQV4Zn"
    "Bl02oTGoU5NEMYhiMl5XCSU4hDIwKKpK8xj/OyrxMOnPGu/+JrBjJ08BtiBadgsZIS8nKX7A3MnORw1AWVZxzwfyuLJKhAKEZv0u"
    "krVXVMZh8qMNd/YwOThlsrMssCqOlytQ5QoydqpWBZ6OWjpF5IGw61TgIqnECIvZgsdbaCZkfWUFR/IrClGM4fScBuFNJOGcdXeW"
    "6WWzDbT5WWiFocLULKmCURGFDg5CWTYVuLeMT5wCqybB+7E8bVn6P/nR7xZXzx+/f3bzPIM5btCjDZh0gy3tmKYb4OkGh7qBpXYo"
    "1Q1o9QzDuoG0bhCuC97VhjAw1V8DdKmHRAiQc0h+x4nv4dZFJ+dTcMc5IWbheisxPvLZHa9AyEQPGwMdlPzWw0AwQmPAMHHw9g3b"
    "suNe7piYBS6zI2l2lM0GwFnGc43dr/HL33z/7uN331gA85snuJUFFvqFwG5LaXs7RsKjEF2kv2E4MrcuEg6SwRdvRfQ/TEPXlNgk"
    "qljfujT5bz5f/Gl/8/b121O8tYGzZ7o4WiWLiHdYwrUwewwOg+XOdkHzzqkZ2kQKNfq4HNWX8vXRDXji4DFXFAgKC8ZhtmeNHf/u"
    "h0vs+Ps/PEeNY7uy0C0UDslnrGXIpyv6XfvFloDCK+Bj8aYifb34bBpkWvCpFsBKUr1DbBN2xrwG5RQ9dthE2nDEC2sUDz/BMhv8"
    "cOQTtuzoS8GYR1/iRN8Cvx2Gw37BuTPB5jZI8WKeaKh+E+ZZs8FqOPy8jaAFmNgc/JGXRsFijsExwh3nhvBal35yNdZRT3xiyAkw"
    "iI4d7DARovMBdpCiWiaJDw96XN6wKH6cOq7wVHQEHgPtzoUG1z1GYHr9C8QZyr3IdBirivL6MJAV4xXC40tn/DqigWPiMUBycLg4"
    "Z9fKzPcPP7nm5O/ff/2Htz/8+24V3qnbbsRuO+3bzgl3J4xj2khU+r+zy8lxRULxhxPisFY98gko8Rrp8Crh0oIuxMLUBtUxMmY4"
    "ikXcZpFgZ5JpVcdC8zWVKSBypYJ2n6L48JSbpPbb9uPudNBjEDWtq8B4V06fCJ4mQLjF84YUmcJFmQ4CwBV9AvEHRVHEVeKxD/Hh"
    "dT4Zh7tEPKe/N0W/iq93UQo4Re+MKp46yyoPAsCAOw5ik1iVFCGKuwBJhHeRmVVQSNusWD3Z/7ByCHzz3bf/fvcjPE+KjWPwxkB4"
    "4ye8sRfu1IZPvIc3VsSNM/HGqHjjW7yxMe5UjU88jhvL440D8sYQ+cQfeWOXXLknd2JKSfKi2XM7RbHjD+WnFDWCCQFRrOjxSqmz"
    "vlVGndj+oq6KarKsYmFKVJuU59tielpq20K8LdN9Dd/X97b4b6rhpjjWGbRCblZD4VdvX7+/w8ifOvRGA7pyhN4IRG/0ok/koxs1"
    "6Y249EZrijzNqNiZhfUCRFVGo53qdUFROmFOy5CJ805epJqtiF5iFmMTo5cjJeODCvcmJjJ6XWqyWQaveDHVAr9WuV19VjhQ4hmp"
    "ZH2+iBAWUPQgwnKy1UBkpSLwJYbmVbbxtUEOLaRTURQx49DkGrOuADRZ0iTme4sW8HE1SpSV/uBr1rLE120S36b4bQHclse+dm4L"
    "67bs9kW5r9h18q1esjVW9eW7D+/+/HZjXYFmR/xaxY4MARrElYqPIrgl+VXEG0UJZrWCJ6uZ7VwIxIkciMCF6lWbUCgsOHkgy4QE"
    "L2kTkMVssWNJCCTznUkxZ/E1Ym5ZlFYSMTHVKfYDovxDfEWI9k0PQXBQkviBBENy2ingaRxnHjAWwQ2ygBh6dFYlMI7bxSIR8Kbk"
    "DY64Vx4LK54UnHc7IEg2E0htT+a93oUBV0VDhpKyE+weqoAINPRC0dcFf1QFiCF6BxeMq8iU2N9JVnvIGvUkKfbl3qT3zHuPckEn"
    "AlIng3Oxh6zXzg86+hXMNsgIpHUOpGvCzKm3MoD+cjlE/t3bV+/v/J/3cbqP4j7G2wy4zY/b7IELKmRnrwIccVYBY0jKPXyIA5mo"
    "zj8EquAUGFwXPEYPtKg4LRF8xkWeThxMRPxHfrzqrk0Y6jNkBEPd3RpZZHWiAIWB76WScapQ6VJUhYwq3hkxl3TqIWJHFW7BFhdE"
    "zwY/aVCJHgpVZbKAWhC1ezIb0j4DeguYEXUN2Oo91xDcqkEthVsEJF/0mcV2qZJDyAxRgwEA4h8YzefuKLgoDaSqFAhcKtidCJxg"
    "mNLxiAy+ZuHUslx+mu5JDkDAD/m7gsXrWVxFaStdA9QuHGxhTFJRFSxjJ38gZaOqIKDYXJgU8hhkP2xWTDGnQARlZzg8Iqq4qjxm"
    "hODyJIlAlgdH6Tk4HCqBT5uwXuC0pjNJTxTKQ4FRQauwqJ3iDxbRBdtFKNm5ngY2CPerg1p4qj7g5IOfaWEBBeetJ22ePVQciXDz"
    "hMkuiVS6OLndQLAaveKgZdcGz85GNHcTp0JY1cgXv13VyPcf33/39uGe+UfGZadfZQ2WyaEaAKa/RHhpLw5JwOnjVJoBQYvk+xtC"
    "6mC3V9ePq2IbgEbGHPa8mPeFviuBu4LYdMeiVXaFs+mim566a7GbjnvSgKt6fNp51235tmdv+/m+1d/sgJuVcLMhniwMQn2q3xzb"
    "wmfKvdRTW0gWaIpNUoI2LLMWD5xLh4uFXKRuBPkkWc+iq2n1w3cf3n/YD6IoUqY6wigcIjJx+0Cd1wJTFhQoyPZ30t/JExuCsAKB"
    "sZBGfUF1OhyxH5QxR5sZQDvRxWBWBj/aoEfEXoKSUUF1VXGsB28mdTsw1WTAx5LWGkjMqiBl/SvgD/r3Bj1z8BcxespgZPaGNiAk"
    "o5TgO4ZbvbOn1tPWyi7wq/d/frcHFEJ2Dk2vOYd1qoJ3mUyxRRLceU012SDwvEODaai+XiHGbOg+cvDCqAHIHXZuV7W2PC1k4E2B"
    "rZYA4yPpAAVYH4c0CwFf50ElML4V4Ng09SXpmCkR9mOC6iXRtKpe7s1pcWyw0/UqRKNMGpTaBGwHIBTc0IMPyVrBANDBJjGU1+fe"
    "2aDhE9YNjp5BXmOoKmc3IfJYURliiQUhz3pl4UQZTPHgRKF1W8p90m3T8TZRtynMnqtpmhr6mtkFDCbmNI0+C+MEsf5vBslmqmxG"
    "zGbebIbPZhJtxtJmRhWZXoygFczOopPxAapXTaWCapoc0wzHc1T1vGW+r6vkZ5/fVsltg1u10L7m1rV4rdF17d4W9braVy2wqYdN"
    "cWyju4z7NiO2ubLNom1+bTNvnZLrVN3m8Da794l/XxPrctmX0rbKtvW3L81t1d4XdOnTeiKBbqD/HXDhF+caYhXIwIBwgIswiFKZ"
    "9TaHynsifC0XK849nn9K34lqOYaoFFB+VAc6SO4WQ/kMXTgMXsQHGx+RYuT25ujoIvRkZSyKVbo4QUByozIqHcntsIxQKSRztTTr"
    "suxQB4NKYUkjXq1S4wYuFMMVnKiqeA4EnRALrI/ps/gTu8WP/m7ZLX50/u/jN998uG8Ym/6lZRrlNhrEN+raiBMSyM1keBlwrpSh"
    "wUOy49AjAsaOpVOxDShBFGjslz4dVsOSe+WpCxgtCQCHVr30yvMlF7cQ93xY9bLd1tFFxTaBsRBGIwi10iSM4oCWwOoTWhSF4GCs"
    "HUQHXgQNCAI9FTkcj+DsB4HJknM7NW2sDRRFwKWmuVb8NxEM1joRAuqqIwIno04PouXmMAOeF32fMqwOnIVWPzbx5QiIJCykDhQM"
    "d13g0QkEXubkOlXXKbxO7W3Ob4thWyW39XMtrHXB3VbitkaX1but6/uSX7TBrihuOmRXL7vq2cwMXwBrLbCfLaVkfvvNn/dI/WJh"
    "bpbnZpPuxupqxK7G7Wr0rsbwYcOC/ddovF8U5ji44HFsPVDalPP/YO5x0r8DF919Ix9uFie0lGfQQ0fxxp8c6Eas0vlxK0r158t+"
    "+rO/3INL2+/vj17eujdnbebW/OWz1s+9umHtngdoKItXoeGVQ1Ak06BNSCfACVXt05znfnxLqH58+U0A/GTJShIVqHYnMcU8cWKy"
    "el5y9DRpnsXoGSG+CoGmmWnMJCFmBSxdturrXyy5YT/6v78/jn/O37377E//1y/+7Zv3H+/ZKZu7Y3WE7C6S3X1yd63sbpfdJbN4"
    "a3ZHzubjgR6db4NaVo3iBzYr5wdAeqijC5A7P17HJ7p26/RlOG4DtQzhNriJDDIMx8O/5QUwkXLqtStQBdJ/Q1SeM2CY7puV3W1q"
    "er5SA2jN6YdtyynOhGAztcwHIEVCWTnWnOpohlfTl6omw9QJd0Fx1uvvueXRb7UI4KSI+v0k37CsoO6EbOQvEKGGaSRHwqMpY7rA"
    "iNoPd9fZuux/8asVnP7+w7u7m2j34d8c/Lv3/yk0sMYNbkGFZ1f27tfanV6bR+zmL3vypq2+tt0Rd3fS7Q681bm3O/7uTsHdYbg7"
    "E3dH4+6E3B2Uu/Nyd2zuI7cN6jLct4lwzZDbzFmn1DbZtmnYFm4Uc/+16egEQYwL9pOiLktgnHHHP1he3MV5oDXua7NqoVf8AKQa"
    "miXYDZxGhl65PuvsWG9e9b9sf5DjngSCL5OEMATZrofPvVCmJrJpho3aw2h1MnCESAYO7S12MHpx1CtKnjpxh51zkM4zI0FpWr3u"
    "2PPhU2DICy4Hlts6rsBQQI7R9BBWTK6SlhhTuCJQAbtI9fjU2KSWr3lpp8d0cUwEHtEdfLks5HX9/+qH9/X/+OH7b799+/4pVnqL"
    "x23Rulss7xbpu6uQW7Pujd4/af3eW2c8ddXWkbduZtJSuPypB2/2eFwKy2tdHH5zhTaY7tXZF2ttwh8uJ7Ffvv/Dk0fivt7W/WDb"
    "KLYd5Npatk3nth1tG9W2hW2bG/Ivk3Th8TITo03bFR8gGLx1CRXbnuz1lJDWF9xJHop/AhwLs4zTfcEvymDTEzcNsqqWReds2uim"
    "p7xHt/T7BXr35ft3Hz++ffbLd998ePe0sy2LfdcDNxWxa49Vs9zUzk0p7Rrrrs12TbdrwV1D7p259PM2Arex2YethZk+xQ9qHhfB"
    "EdXrFiWezFXOavbQ6vz/6QpE+/7jv7z/8PbZT9++vSGdbzHCPXy4RRbXoOMekLwHK9dA5h7k3OKfW2T0HjRd4ql7qHUPw24h2s1Z"
    "untRN/fq4njdXLKbs3Zz424xilv0YolrbKfR/SB2nTduR5HtlLIfYG5nm8U0X232zZjfzPzbAWA9G1zDvp551irVP3r34buPb1+p"
    "TvWHz75496+vu+MWFCFFTm/gRhWlM3Tfi4j4iZBTcC/CbXThEwPsfuGXelAWU8zMNkSXRiS4CCwQkQskcF2Es66q0HS2Xx9pQpTy"
    "BVCy3ycJlqoiF5adfqc7qvO2JhhWyaq0AIC7ZXQLA2fOqCIEHJgz6WWqTLmMn/JFXV6qm/9q82wtPq/NG3bzk+0etM23tnrdFn/c"
    "5qmzsoumu/l3tI2tScjmGfKvaZNpEMDgF0HDQhfdXlefBU/7ttKiwdWkATWGZ1HHgxRz+GqLeTkTXASvX1AIM0bZ0sMBp3Va64B0"
    "Tz9GJHGTIqEmYQ8QVg2pyF5CHH6NofLiaB/BrqjDpUyGWNgn3YWoDrJYuHvRIs7jVRWGK2fOEHATWbNee9iXxEbv+7OVMu/DzYW1"
    "TTF0MDm0wWb9osINAU4Fgk5BY23ZMZqH3UgMgDAFwG/y04KqwiSOPQ7TD6UJWBCogfVWI55Rv6NpNjIbiKuB3NdS2qjyiZQjj2oi"
    "Hynl6eh75OlIZp5TGzNWZElQgmZWCPw7QuiKKhz4kXyUyo2S9zJBYhHrAORhmeHFA2zB0+dZkQklp21GRZEmLCewqWpIRL0KBVai"
    "Sp4MtQuwZj1E7PzyLmYgeRlRZVW0MIOoKU4CZMLUjhkuiyCYUDQGWeDqAPv3FPz7lY+xe6g33/Xm1d783Zv2WdXSpq82TfZ/Gvu2"
    "bUuKI8lf4QPOZmVcM/MRSbQugNQNSBpprXk4iDNQLahSF1Wtkb5+tpuZR4bHrtHME+XkPnmJi4dfzcb6m/2NT2Z/4/mbNVcxB8Jj"
    "iDwEz0NYPXxO/M5pAOaRmUdsGcp5lB9mYMxNnLU4o3G2w0KIa+Rh/YS1Na+7TcjSpxIBaZDgHGDh4JHGtgm1Up7Atd7o7pzsZsv6"
    "c9vx6oM032sUI2dc6DEhEZzF/5iaHf/r448+EDCasxfrbF73nx8cX8j8Lpqf153mtpovpvXz5+cHQHNjfiUCflUzRTcJno2xvEJC"
    "F3OBZXgTRGgBsaq5HZCcAAHav6DpCFa5Loo2RRcByS8EkcIGBw9sFfT5wrvwppGTxCf6cSZNCia2sO1SZUZLY8vS9rI2xTy0zISG"
    "mthts7TiLI06SxvPQ1/C0rUQysGXYvFYSb6UmZNUIpFf1Wq5SdGy6QhHZ2h2u20j9cypc1oXmw7dTbxJOrdJL+HdKPhT0U3lwrLy"
    "bRiH+DjvUa6cO0G2Z5g71982Ys5LHWQeWA4Vk7vTXMBjAOHxtSoy2xrsHQunB1e9t+vg1BJJPQ+GC5a+WxHbzSHYMrt6PfiZRXch"
    "8srMjuCkBltrmyB70K5FIe4gSdYwI5M+s7VYRRsZyFG2bQofsoEqxPurmklZzVdjt831h3+MZczPf4+FzA+bbQdLETunClqPvD63"
    "kqpHqcFSuWgUtvGrUjd+tXA1FpGHqeOpNO4nweKVxq2oY7+wx8nOPEqFfWJJFzfuLt0XO8TLgbVqlGIt1fcpFBq4ZqfHqG1DQZhC"
    "IlenL1n1waItFl2yaJpHPRSU1KLBZvU2zcY8ib+boNO+ePNP07X/9X5pYVzmo/kG4it19Ln5Xizd9xO18U4dpdRuIa9QEmZNITUv"
    "Dg0Tpf2IVFw3/zgYAP9PsXIJ4Zyt7HwHzzjFxq0ItVq3jzk9+sud+rif041UDVVFZENorEKqFfAw8/UbdWzSp1LF8ss7l62QpEv7"
    "mD1BCDDZMkXDEIuHqpNB7Vpc+2jtKsC4tDaYLLGIuEn3UQ9Tktj61RxoYkHD07ZPfytjvIgi1xdeY8tWVYcinMqb83AXdvlkAbsa"
    "WTI7/IpO2cbewb1MjZDCGrMlndnipzbJg8TBuw7Pbb86/kryFq6s0/xgC9eho34nRVxV4+dJTuCiTs/OD+x9bbJ8bMFc+zNj7+ba"
    "2BnbPpem0KVldGkoDe2mSzPqQ6tqaGRd2lwfmmBnQ2LRKovOWTTSg75atNmi64ImjGryQYdGBbto36CbF839oNc/pPWhg+Yo8B8n"
    "/fXy1d+f3716/XAMLevrmEgEC4xXE7ncyBsz1uYGU8dcdkonGbA3zd1Wp0WfyBPTxuJbFm5Y1vNLTZ/zyVyu8snd119wFx8cXKB9"
    "nPKWTNMK5xQHBoid9EtbKk3edHR+o2McnebrcQa1op9SF9oJhSec7pFlSF7dh+cdqq8qIJk61M9pukKkjraMwFaV5WZu5C9TxKyC"
    "vmxTS2kC6+Z5Xg2m+2gO3PJ17cQukjGbyCK0y+ghD99OHZCp3dV8mXEsXIKpB1m1SYyfWWbsTsKtLCsWtlJRiOqAu+HCMSjVMlgc"
    "bkoumsrmgt7UBk3SxvMcocqbB4CzukK30YG82Ler9RtN48VuXqzq2eaOBvlirUdTfrHzY7Pp0oq6NqoubaxLk+vSArs0yC7ts7G3"
    "dmjT7o23PMp3bRBwM3msKLv6rJcLb9qnXYvb2+zSZYVpeXeOk9fN7pwtxSM6rbBdYhF5m0cRwMG3+8dhgSQvI4RBfqpALsPqbv26"
    "tnttvH1112Aj9nmTZpn26JzO+cWvQzrn/etvX60eNXl03f6H4cBEqex/gVKZhBhe0RHVGN9z3xbag8HVU7SHtV1eDMN2GUgiVgmk"
    "H6JfnarSvB88nNvrJEgB18r8mtPn/fqrKSn765/ePr/EJmA7lvJTycPSuXtxReEaGlhmDRQdSvsTDgmdWOUJRsd8rcstOXGtyC6r"
    "JlTd8q5yChM2hf1P58e6xQFh1zGcccOc/9/SffOVNOzFu6Ff0ji9d3zOpmP0vvWKasTt456KQExIVGzSOey9SbIxkQNakOkvqiYv"
    "AN8reYzJjqGUlTgGNiTJp3qjz1++eX69RAAfHhAfHl9seenDpFMTed4lNy8MutAEeSUWlqzDFr8bcIUGfwH8YRHu1fwms1/0i1/O"
    "zu23z989//SX59ijU0+a/woQ1pOaitqzbdRqCpO2jXqrcubb8DLgHAMMjZoq6cdUvLptnxySk8yZ+qlfVYlvJW6Xk+DaG17KsopA"
    "T2hS9XRKWpyFFbjS7nbUnZa1NkndeVEkCtWApwC0AXeKnk5WeV5tDF026Kpaae0Lddzw6+hW4KCrhQgeqqA38UIvqQJdkCtckaeG"
    "c6PfAsYiMQhYC+EoFFasOAWzQMBqpeezcY9Umb/qA6nABRrUv3rFJGzCWvws4/jLOPY2mOoeIfapUZnXy1a+DwUZjXGU1casic6j"
    "2t3M5ud1Z/ZG6q7uDNhpBd9nRxyim+bjoCkNY7MCXHxQLdeDToY2TT24RtRqc5/3dF6nV1jFM9TZV/9jpin74eWnu2n9su7iSelF"
    "fRh15Yf1qOuTWf+uutkkRd+mx83h6i9CruGR/HIFgFh69NcG/tDdv/T+PyADBNyAB1SBgDkQAQkWtIIHLIMF6SDgICwoCR8GGfj6"
    "95MK/vr961c/vVpD+KZJxSkOVopC2ClU2xSlLxCgKQIf2KFFR9vc3We8TyV/df8aU71sALsPUhltM3e1VwTGCa77u3Cy0caYJN0G"
    "BHrwXcpq8LMloSiLWcHFrm1yXhLuwd913DypoeG0557e9FRN2ikcUP+H3x1HgzcTGui6JtF+ibPHszUJH5z1ZDt13SRMGpmyX49T"
    "LSuAsLFsmeE4sYaLJ3m6DfupS7sJja0aY0KmefzFn6dCnE9++O5D9WZTszXivVkhqCGJywySIlA39vyeDqctSfCxZnreX2Vj6ceB"
    "N24sF7mvzLJ57UjHl3lVyYZdvaviu8MwUMOxqbrihKuon0IURddOWBe6tuO4V52vRTHsFC+q8tltTlWghyXjxXEbzAQRHVhvS2E7"
    "NtAzcP47EAQilAQbxWqEFkKzua84dFIXnZroxSu7N+bdLWH7m3NdzstCD1tg2hzzrgnbKWy0sAXnvblsWmMSUFAKdkA+fF3Zkji4"
    "UlGKl1WQCNvagh+6cjdGuz+mQNj4Mys8aVzcWMB+2Nt+McFX893cyCq1QoeZSWnt/Aw9oXOzaOgiDTAdAcAjQHsE0I8ABxKRQhbg"
    "gWubzOCYv5sch9+8efvt84OdGo+j2Y2wfh9cS4wF79AOGJhy928wZ53Chq2gKPEGQ15/tGFr6UqHNmBS09h0tQULUJCz4B4t4I6N"
    "m4Yxm4mvTNvfHK3+4HgEl2RxVuIZOz54jkt9ORVb/v71q3cv3370ydvnbz769MdXb5/fvUS2sZbQ6KtyvJawACu1aMOWzoIxblm/"
    "hLlkSNgQ8GeoBPRwdMNJnBnrbGgmtCAwei05BJ3rpmHzZrekO+8IbWIg4meQ8FY7hZPvUSjd3d8sEpohNf1ZLgNJtAF/LGtFGC45"
    "pMZPwwbY+NEdSxFLYB6eaYz/45MJz+M/nt8t3kbDIZZFw9bgk3scvCU8uI/7363BrJBsA6VCJjNXA7xeFnVIg+rNKoNrOO18M8+P"
    "m17ysz9OL/nZ+78/v3oXfSJUH2c1lFYwy2aBOFeWJIl0tEJ9WrqyDXM3KxlZ0WKbyXhaETDMBLaoIExRidL8sDka8OVUwPDrt8//"
    "Fbd02Jph11YMYxHdAcCiTYtjdCoG1Q6VTumo41gxUO82rJcKMFOzbOThmGWgOtwKuGvTARTu2rMIz6NyhyqCasjefLgkM1ma7HkU"
    "RRatBP2d4v61y2TB9qo8vNxBtWgpdrpGUke5Zgqa6LycjSKOk3po/M+yjPk6HWGiYO+aoJE9ELFInmOD/X9GVTnN4e++mMzX3/24"
    "qOWHbRe3ZNyucSvHbR40wKwcgt4gao63du8EFoaH0Iie6939yKi7+2pEFMCXJZz5KShamAGNZ7ecniFhYzS46VnT1BxzF4ZFgyuX"
    "VQrUDmF/+LUNkMFUCci0Ws2qXrrjoN0l7QAGgTfcsBDM+9W1DFIrWDWNPRGHvgip3iTu5ka8aRFlNXZeXBKhanqZJQY7kO1MspuM"
    "t+Aiu2rsClburCHbbeFcKdQdzytDvSZqCNO7ALfRewDGd1vUsyUwHg6HeHBY6RlTPQ39lFYlox8WqNBzOWBm0qzfT4rxD8+v3z+/"
    "i801qRPm0sm4TD7qVb6TyON0E3BQIvQ3QgkuX3+ObEwnumsSCKvJneEG1qGZXJh+K35HxhBOl5VPVFGnmSa4XdKfX7efj4BffRGw"
    "CL558+0KFLplNW2zjm+DfrwAW1DiYP4xHbAqGlaG/zf2Cmhbpg08D+6Rm0jGUF7c4ZCr2HrbRSXHWtNtF64Ky0vv9yWTavP75jaK"
    "5fHUCxRh88YvYpNsYHfdNIxbUVNRavqa8a2z/z13Wn/9/fOrH55ff7uM0vKlyzgso/QwhmGEyZMnlyptm9qt9MYbKqCLO8oi4BsS"
    "mxjoU8DDSAKSPU+2wdCpOImdAoVymjHqBOV2j+wE7yagGpwljht6JNSUgvc0iQtuy6ICyrqGVmCCf26JbDDdvw9kPdmfNoF6iK2s"
    "6h2BnuFv3wY92Qm+s52B49OKpIRMcR7sToLPcBfAKgUVcyJtJWSL8yBNNbbOidCpL57zVG8qFo8Nk2MSnGhw8kCOjSZWcuH9ahnw"
    "YieJudSXch5COyt62XNg8JyHMNLgxJ/kABXTx8ndIFD9U5j6XU/LcKd2vyM8rd2HCWBgWkrI4WSv+95AtWRXfaZtj8n7sHnJ8Myq"
    "T+8GlP/qVze+XvLpTgOwzMSaLkSqrZAiU/vRcGOTA0xsBKFSPZrtlJ3sia5AdgCuVd/o7RwIRhABZ7U/qJc5Y/HJlHj//PnNT8tW"
    "jYomKqFFQy27+uHx4eWWV18+bPnsdUzm8VoGcxnqZSIepilM4jLFDwtgWR4FZsbu+zTRWql+Zzj+Y4PDdtldFcAgKq7uyA9Q/Xt2"
    "0iwc49OzirtsmCo5C7rEA/QHQ3knBhiyD6JJYzo2GDvdxbMMHk5MbHfoPkx6GvTCJjZwrebl6JnTRjNOzxf/eH794+LFLeMX996y"
    "L+OeDdv5YadPWiBqiEV5RL0Sdc6sj4KqWrXYrN+i5pu1YlCYKNewI1TvsfdBUqgnZ6aX7c/SAMKyd0TAXdpux5C77iNUIv+dIPBh"
    "NFMVizubiJ85jOgUE0/1XUjnQHE8SRi9C5OHlIYHT7iTmHinAmKkYVe09CRslqJTZwbPqOg1TMJ2OCh0om9mSfugyziLlj8fXhDj"
    "YqnwWeDOl/HDggDc5l8AlgoeeKTCEGfvafB8lpTrHIUE63aTwMDfzuEyR1we6olOD6sePygZ1wY5XU54PrLwT7iW+Rh/dcJ997M2"
    "86+0jhl5a1p1O0IVu37ZiSKaNcH5CeXmPsGZAEj2O8SQtu6LOisQf4J72NwsHWYZqHJJy/iEytndVKh58JfIxBgA46vuWjTbtG9n"
    "s/+3kzH8h1cv7+7/XDb8YktGQ/PBCg026mLBPti30fhdtNOsuqJeW5TeohKjvlyU6YOqDYo4aulFhT8q+MuFp4mtBY/bwnM+/Mws"
    "cKuVKmrylk/3CPZywUNvDEI49NmGNogBKb8d4o5hlY/9Ld/Yb3wM9Wtiv+LR9ts6WKIxwBjRXbclNKLSPyRhbQ7Gh8qX5E0/28me"
    "3e5/WvbRwmv3zVgfZTzUWnVb0RulLs5TTIVb45jDY4C1zUsukFd9NoEpvnn77vuPPnvz9mWtMEQ9dPUB/P8X4ZMWpL5qVm/aKi1/"
    "S941VmMhF1a9YzqjzKEm1TudADjYvHILGewq5A0TN1zleraeh2KimsiIh7Cp79X0UHkqqsCBaJKg29DBUdytyThwitM6ZBBDlcMr"
    "JQ8koETG5T+WE2QKdDeJXVEZYBLFwdM68ms7DzITE+KAXHcZ2d+y+0iAyXO+mnEnMU80vxWfA/y9sntbGoJwxc1AVF2Y2MatMr7H"
    "b2UptEPqEKwZGJrTxWxiG3euJnZ/rlUrnV6RhuVYNwcBNfvDJlOvge7RKpZRDHoFG981qjUpKJKxYW1ZnL4ssJ6yrxnwK/LitNrm"
    "cMWc4vnqzfsPr/p1Dh5maJ6+ZW7jxC+rYlkzp/LcaiI8UerUvZ4XTeUecca6RkabWiXDXiuCkvS1qvr8RG4fy2qfEo0k3UM9GQdA"
    "GeXAgJ0uKonGhPSR1gxjMdu2v51LoqwR+Ycl7nPsYE88MRIHerqrUOJOQMM1AayeKFkUN6SZPvZvWTDm9SuGeeK0sUqqRKuFNzxl"
    "HWQQi+7u3+KGVXbEUZ6adxsiZtpEsEqTwPIgHqw4ThObeyH3qW3qDTMxg8Oy+SF1XwZNveOmZjteL7vjd4DXsvqZSzH7WX6Az1LH"
    "FHjWjBfVj4gKiYUGm0hTiS2YkC+qgoNKVmAGYkyWIVhJPx7TJJ4QVaQAdGhUpkHEOq5+fCen0tRz+OPj+ltyr3KREC2nesW1xWFx"
    "EcJh9modhDcHvtU3UzrF27q5aF/XFdrRj7u/Iuga6yjhxQldu7fWNuz45u+EQq4q0MDE0uFa/Vaoga/qarBhtAe1j10qoHalOUIo"
    "eCv92v23oHel4WjBPoicWuDsgb519wOdDK5ucuxUQ26fJPC+Jg9WGJphGoe4YdQlt9OIeTdEp4wdnvhBctI+olv+VLPNcY0md8fH"
    "iQ34NEZfe3v6HWjfqspmnMirVmGL3rdiBcfsKX/F3twdRHQ/3CWs0RPVNlV0YCcYQKsUyIloYFXFwYFiNF8bB1JLlTbMpC/m9N1v"
    "fzGl716vweUHxyP4JMFbmR2Z6OREByj4RsFrWh2q2dmKjtjqpAX/bXbtotsHnJisirIziXGBPniShYzT7USTcybQw4mqlqzVeiYV"
    "OiTp1oK0FucQMRehqZzA1MvCWTlO/JlI2g40a2eVKh6nkiHYOseBwg7VmB444bKouQ80smYRvx+AsrI0vU82XkV/hlfOOhyYlut+"
    "VNDeP/jsDQY97G67BkZNX0wZSPi4fZd317iWmP3Cjj1QIyG4mqMh+iXgkAMJ2cTj+iiIAMiaOrIkJE8ldXqLByPsyggZxy1iJ6z9"
    "QS+imp7IwYsCP0oZtXuHSyjsw7DuqNlx2EJJsvz3U8CBeNyOQZDttcP6PWlF7wcj7zjmjarUoJhJXQ24sJ3zaTSlp1CwdigkQYzu"
    "aHBOgk/YUQypUpC96vUwontV2SRW4w7Hz6NFe0HkRCzbO1ocPBS5Z3mMMKkpefRrRwOTBUTwyUkEqLDMd1T6Z5n8/cTqVzlZNyAj"
    "uJKNEvlaMcSdJQfO0m3VbihX0hO2SQJul9dC7FBazm+4A18SNhWf0BCzgEW+k0tWzUY7mkyyEgv6PpVh3UeitpEr3wFy7An3HTBF"
    "Ret6RwN+Uav4jqJidOxSQkYfYYm9oNhfkEQ7kqdeCbBDzxdutx1GtpWdaTEcKIfE9+zA3hk3gaFWVItKgukirbOfKs7F1tkRxbPC"
    "Pf3djjQ+DqAdxWRF83WwclIIygeq3b26kLTOICXkLzcUrlXtuQ2Dqc0JBY7ZOqqiSNQf4K/wnr4D9O9ZbBYHMthK5B6wTLJqRO8S"
    "p/WQtOHwcLWW8jhKDqbyleqmLZv3YdjizxKVldXh9HEOFt5fujfn8bQzQ0F5XDZ/nMajTdPj2FLIEGfYqfMYYbXuAUrWeh08tVAT"
    "tinkmVDrVXX8N4zwoWs2Vor2z8fn3CD0y+nQ/dnz6+9+eP725afv48m7Hl0xmhkCnTEGujHuf17DQA1+wsDJwjmwU+3iHzILA7s6"
    "DXPDuxHOTWkFKtVT1BGbJF7qFBjw80NtOVimQ2c9kMJZFU6xeMDFwy8ejPOhGQ/U9bCdDuJ4SMfzez3a52N/NQmCtRANidnIiAbI"
    "NMPz4vh6ZgL6/v27pRxnXb/Tyo6LPm6IsFniRop7LOw+JKhtY1beY4O+LfIxuWe7HtavH16vONOL/PvnU0Ts5W8Ld+SDSljURVQl"
    "k5oJGigqp6i4olIL+i5owqgkowKNyjUqXpB9lG0o1wIJ9rw9DQoEDpBZRSjNy3ytXEal+MGqRimho+mEy7LAqKz2RYPOYcdPPpsb"
    "Sf766qd1/SynUzy5wqEWz7v1LJyOyXiCPpyu08kbT+V4YsfTPJ70wQp4sBAm6+HBspitjmCRRGtlsWR2hNoLn9DJQq+i8o5gjxWD"
    "2hLoyGO5UupZpdx4Xk+mVBVR6tQEXT8sKgDHkHWAh2T1ZfesGnJEIHrWPoMvdb/LjraCnULBEYUXoSYTdULP8KW0xHqRhOhQR9gn"
    "C5i1Vxh0Jw3v3uDU+S+7rsHT7IAxKjL6e0epJdtw7yNkIbsh7Cg/RLNDP9RMidRDP9S3BH3YUQZfyD1xF7huZGZ2tn9o+guqLjlV"
    "J+6houYdPLalDAMxYQHDw97Z+yBIxh3luEXqfbfJsRidhIyYHPT5DtiIolL9HYAa3q2wA7O6SMfugNEr8mF2uLZWYipjHyWsXS5B"
    "Yr/MYh/OZd//NoX8Pvlf333//Ppx996H3XoapLc6GHEm6WD3hNvmdbxbR1yuqD6nowx050j0U7fsGusEqfl2YnRSW60g+JokHQi+"
    "Vo08Y69Z25ch3dPHmlKiVFHAOwYbxb18AtxAb/Mwl4WfIA3EThEJJ25SNSuMxNcPzEOcozh/cW7jvK9rIiyXuJLiKptXYFidDys3"
    "ruppxa+bYdomcQfF3RV3XtiUYbuuO3ne5VEDRO0QNUfUKlHjRG0EZT6eZ9aBt2j1Te8Fb7uzI0wmat+wbJR7tF+iW6xJ3TW0LnVd"
    "Q2uybs9WJUxVR/C7yNzrQCe1ncf35548pINPtMHBZuyIRRUFIDqbvtWj1FFVVISv0Juu+eGQ8yhHnzfrXK34m+mM/vr5P1994JR+"
    "2MsFuaR0nVWFAQYeVUWMa7YCkDdy1YtLHDFP3LlXX5C346LdEJBVZnxHTWFlksAu2b/HCYp7ZPfNkc87pVMafqn1a86rHr2jqqmw"
    "C4kne5FjtSPKb1BPRbsaWTHuQGDJeTqHEZKiQsK9anw8QjLv/6gbgtqIGiVqm6iJVi0VNVjUbrPmi1px1ZhRm06aNmrhqKCj7o56"
    "fVoncx7ulxNSz2f/ePvdP/75aAhiQ1U31IDfVBlF5VBUYW3vgDqr2RXvfdKrup13oIvU4uEtpgRhwe+GEODpwh1ot1WUfRbo4A8l"
    "8c8U3cv4YZGUEHZPioE05i4VH7EXlqd7fxqEzQNwWNxVr1WxuJsCPDuypUgZ7gV9idJ9O0oAPc+6LM24bOcVHRd72AdxiyzbB7Zc"
    "Vdx0RxS7CnVmT8ralsuqqWmsooo0RsvjmKoiUpjndNY4n30x9ye//euPLx+wLaw/zO6LgGQDvrkPfQOyTVX9oDUV4ClsIwCERFUa"
    "rzWVFrCY37ALqsIWrQO5W1Cy7IvwF2+sI1BKpSFv6MULd+2+IaNdpfkrHjBsbLx0lkV84JebTOfCYoSm4wpTXfwoQ1rZDz0m53Ue"
    "Nuz8c190+IN+j7o/ngvxzIjnSTxr4jkUjqh4esWTLSmtzKgpFU0fPS4ZWqF65wo0Bpt02KZ0MJZlPSf48qQZSseohmhNXePYtM1K"
    "YPzLW4HhtMMxbYDkceXYCkwe72mxfHNRrrWZrYPdoBXWz1Er0mCKedWJ3QS7oXsTS8XxdPUqVpkMDdwnY53yLkI7aIAz8n0zLe/Q"
    "v/bb0L8Wt8TSjrW2ak1dXLHBa23+mhvDYtNYbCiLzWaxES02qcUGttjctkpcC6ejcWC+s/4upaHiKnqFimpgK7Dti4y9SpyYQ5S8"
    "HWfQwYOsogTUpP0aCVU0V8Rq3CK5S1wAeep8U2ilQgPYm+2C0WCdRyZsBk/Kg82SCTsLGrxtOrV9xmm071pEBcf7/rBkw2KO6zzu"
    "gbg/4t6J+yrsubgfl70a9/Fsu65W7WLxBms4WsqzFR0M7Gh7r3b5FEAIoYUYdYgRiRitiJGMGOWYAyANOLtW2aRePsZNehptfmqN"
    "smMC9zj3qzlMhnBr6qJ3lUSpekPzOYKELSNS52vGTktEy3zNIEh/CAAoISiVBNTTEOHz5swT4AwlL82ZMyzin+aip3+smBEPeFgB"
    "0SmCPc04TRE5aQFVinhLExQTWCyK00u7BvF7EENhE2hTh/FLHKjuYAtZffMEchDyatodBsS2KzZYGhiy7j9XeBGOHVKTqq+IAhS7"
    "dmNH79rtGzuBpy7hpe8/YgJ8GB3rky8n0+eTt2b4xOlZtFfUbE2FdyXPUpWuLrOwQXPWD+lRnF046yuK2asKiCpiqlV7s4Kx2z74"
    "lPqlmaO+4wSr9VRPMk1MrN9K0mPZtxWFUe6wVcDOFtLP2AmCV+bIdsBbKCQ9j0JA/Jx5n/7+8u3LA7mAkSwfqohEWqSdA7W/PHUF"
    "wAmf2dNAdzHereR80nezsAtxh1j/3WuDSNZcVc0EFd4dRBB0It1ZK8AN0kXsRfaULqTCDvatQ6VJVmM1S/Ym6rm1WuvDJOyejOKC"
    "fpGj3M9my407eiN+2YUZaZwuDv/Ol2ZGAgiRXaYekm9PXdHpjB3S2+DCKfhUlj2dIIYpKmWEPdFlHRvEWLbRK3WwivRNHXrYIbCY"
    "dW236TnFHHNgsnaBgd6VSHOGYpRjN6GNVtvIrTv/RXpqUigJXkETEjEntcmqBSi63bDWASTElTFntz6fdOXPXn54fvs+dn9lBDfs"
    "RXZBFxZ7nsMTGmeLWnBND5uwC0v2wKUm5MIT0iYoX+NrVqUaDkaTkpBj7aUdyBeAOE34RAVtXk3JO0KHNxXGm9YxIQmNtkHQPQr+"
    "ykGGjUdOdR16GuFlogDyoyaXqiDf0VS0WwCo3eRF2UEFqWpIcsM1B7pHgaVD1+MmVQD5uaL2UvQFFX+GE8TqV3HNiXUKCjoJUwrP"
    "panrKJMaNl3LHr9sYvQxTuE8YFSzCVUApscT7P0hlIsTCANZnQcI1/zRtV0Sevusg/0QV8COORXNT8Na4KKByWDLxHkDsEwO0QT4"
    "6ppW5e8/m1bl7//69vkBfC3MFdBkm1LbWiV5YA3v+O4sBPm7gm/qKUG9sA3XLuzIdE0IIb+9FtZsfko62QoqYXcHFmeZrM7Ajlpc"
    "4v6jTBkAhwK8YQ2v46ej/pSuDPhPqoCB6GALK6SwG+AQRo3tsioCJpxwVmzolgAqTQl+if7dKjThAoSmemG1o3x1c8x6VKReJk8V"
    "6AAhdQzhUNfs9WUVFoT/ahuDt6Na8pRkJaJSsjbouImw01mW6kDdBb8TGrV9TGeVtEXUD7xWGrirdihKgRRc20Te0EdpZkbTNrwj"
    "bbkDJbLacXjH6vwhqOH03+EeZx30FFWWWEa0r6pL1WpO8DVNuz1BKqKq4Eg6WuuGoXS9cODjnG/iTKPONzNCpQY+YFrZ7BRhxVoL"
    "wD4YUlIeZcjkt6jyhTInXDHpTP7nQ/sYNe5VdfkZPlQVPbU5B5fQUOeqMG4GgkTdh65RYXOVMsAPpWr2fv1ZQTjx+Ni55ngP4Sh3"
    "XmqUNizm6hDRh0lHGYjLBtMpTOgdO2LzaxnV9Xp4xZYrTp2SRzE8+VnaRH826ceoO6NejTo36uOoq6MeX3X8pP7DyRBPjYcTJZ42"
    "80l0Kb65yuB3UynHv795QFtY9Piq44P6DwdDODLiabKMZBjleQbi7MSZe5jVacaR0/MOioxOBdOIhVLlXYqMxIL+gUyp4FrbB9Fe"
    "FdZpYjX2OfHFVTmWVgWL3otNFl2DVPogfLMPEjPYzsYHpwU8xscmdM40tdSho6CJrCr5EZLcosPa2wUC3jGpm/jeKkyU7nRnu0ls"
    "+EP+r4k4h2TBZh51SrCjNtmWBTaWMxHaitLpnA5JXVyEOLir8OI7/qyWMeqtjrkzY0zQkRkI3c25IqcFNjt7My7MJ+9/erf64glV"
    "zK7CEpIG+yDPK3WWGs4s519EwX0btG2mSzXMCXq2a0g6NLkT0R34pfPX8RBxMPIExZeEupmokd10xuGj1kicYF7ur6OCLbxoINgF"
    "VtjL0OLoFK8i/j35kU0YgvhGgXE2CG0fEJtVwSyQR4+OE8BL1tEFh1fYB1HbBqnXAbEOgNwBqu7qFkSCCOsNirzqeOuVNoZz5GVs"
    "mk3kh3b/4yLvhCAf7+ze3QAQ5nH+zNM8LY9fzTCtv3r/+rvnt/+Iems9IOLhEQ+WNIbEdAdnRZqEB/EgMoWUpDswz0nK4qARkz4k"
    "sdlmd3pIXmvXt8oKQM08VpkWMXtD6gfWbVjRcbGHfTBvEXRFVwXOEsAP7Jfdd7IPiAn4s+0YrBDVeyHZhKTQrjqUWNQGzCxYCq5o"
    "97EsTB1gfuUsH7hU8nKSByDxT+auuR++ffPfcf8/mCKzmRJNmGjeRNMnmkWryRTNqWhqRTNsMtEezLfJtHsw+2aTMJqLqykZzcxg"
    "gUbjdGvzD6mN3KdMZbSMZZxUVVFuFFMOF8DCsvkapGmopyn68ne/v6boyzc/Pq/xuIc3myzqaGwvdjh3yCar2dSX+JNQW4lGK/Gy"
    "NTRo7foCC7OV8XUZv0wioNtw7RRZXYPUZZym0bKW0Z1UiZ6QmThV3BYYdGgAc/JjNL7VyVStMmqy+ulEqpvYMCfuvsZL0jYVQ8KX"
    "3NRoV7Wd9j5a9qLuWfRS1FlBnUVNF6zraHgHmzya69GUj2Z+8ADWHTkvobi84tKLy3JdsnE5jwU1gxB9PS3Ez1+9+/7941KMgYQY"
    "ZIgBiBiciIGL1Rie7Zho46DKv10LQmGrTuluZ1pIy0OHCHAlP3G6RTXcrs0IymlpMq6XRUFz4IdFn7AhFiLeyVY8ADh/+Dxkf/hk"
    "xm1699/reBlnDyKA0qc2QruYcnY9eu9LHGYN0cRBD68VXjh+S/zOMARxdEAD29QFntHt2/axIe8nQFMrbUZmywKkCkYlSmXwWjYB"
    "maGD0aRTwwrBI1MVEjfyNEDTuH761WS2fvrTuzevHwfW+qjPcZdZwkZoTOJlNPR6iDcD4mW8ZZyd+GLhnePnxE8FgW9TFRYwYkzi"
    "cVN0rTsbfLX3Svq7jlfWCB2QuhbbXTk0tdP83z91xmn/dNq7v3x5ez9FoiH34KlELyZ6ONH7iZ7R7DWtHlVD87qIpQt+mPLgWGpi"
    "jEocB2coyohMbw4kIP8ty7Kmm+lcTHRdnZ56KyNEF23paGavJng0z2fTfTXrJ4s/OgPRUZhciNm1CD4H6FddcR/wkL2+GCfgrq42"
    "/I2MP0OfMrOfoPDsyee/2TePQx8Ob2U+C1Bwlr0t/FV3kIOmMAhWM7X2EA5MNk5j8J81IRt1M80sRoHyro8Z90fxFTIexbHiuaz0"
    "kfR/s4OuN2bxkdpu1dHhM351aLzgCLOlCDeu6k3HHqiORKNLm9On4xzIY1XYg+R98cDYjsU9D91mc8rl/Q93r2dVL6uRMhswwbYJ"
    "Vk80iKKxFA2paGRFAwyNPrUMImWzs7NOwIMlhJtsMxYNJlEspzxqyQC/jTozJTBwKUtT13MUDaLBY1SnAccKUhUPHIrONh23V8Fb"
    "RnFzFZxSJtRBHsc0Kxt1wxMf48cJxkd3rzD2PBE4RnymV/ny55Nee/vy8peX1SDJKBXzjwbMyCZmZRShnwou7nWAmTCUVoSQZCbc"
    "YHIALjpu6AYBys1qHhZOEZcQyZlLG6axv4ghES0DGQc5TkCYmzht65SiqLD18ThLf2vuVSqp16xIoh8inD7LKBUD3SrKyPzsgpSy"
    "J2aL4FZ5WA2JdWPbiBoS4ij7tTZqRIGTheprn34UFZ77iFl6MTNjlsXtXNAeeeVgLqoEPj22lUYhETo0R1VhBlND6SNAUc/ByZKB"
    "quGtDZZOxgMkJFQxFeW19zIqHugkeJ1XZnGVIMjA9YAR09+xSry6WYWqg+PyLbbxbJWTS59srF5tY7skuSQwcSu7EuZFFLh1voy1"
    "qy/xqH+sdrsw3yMcfISKXwtS5mKVUMgSi1xi/UuojIlFM7GgZi22iYU4c5FOw1t6DqyLRKW3iy+ij5ThDnYeUbZuqHXLF91ucZ7Y"
    "xFfuV2q5KHxpma3rvUCMbn9VB2O5/9AUMyt2lO3ZWPgqG7ewplDmHMs/dSqkfRSZ2KmAKpbqbjnWjHvzuY6y3gK0QK+FKZirKmRB"
    "NM9DT2u80j5qkQuOcVdEBXhQrgxQkoilJ0ZrlNdsqkBiFTFgfioNI2mspSxnLdmZy3lCpc9aBBQLLWMRZizQ/L+UckLRRh0Zj8Rw"
    "Wh76Yc8jMeT77FBV7CZ7mx+76YbnpS+7CouSZnnD57kzPOvgVT8H1T3eeI7Uf/6zmWnpm0ffe9Uok7KJemjVUZP+SqgjLyK2MYnl"
    "x7rG0yC3K43jVjoQotIoxWHbwe75HuKpHR+UULWuKCfmxh99eItDZuo9K59H/J48osXEGcouYdqO/iGJVfdtUtRp5LLG2M3h7y//"
    "MGHJv33z/O4xOxJCyXOUOUagY3Q6Rq4ZnmzyHU5kvdQfm9iDpJBGAmhtbVetEu65KdK84c/KFboWWR/ovPFDOmVnv34II8XjfsyO"
    "OdxS6oqveV7BAk1Fzg/wtYrqymiDioE10Vp1sPdDyE/1eJR26SUPt59lmIIJfelVsB+JOkuoUAnlHVWd1QlEQrbv68iMtGHf5+lb"
    "0eVUSdKTiiK9zqDLoKkT6O4YLzm/FUUV9B8qQ6iEZwOd4riEjLRJ8n055ruGuWOKDx90FlzUmI6Ylt7PfxXqCF+9++fL24es8ZK1"
    "CvmsOdE1ZcBicgx4PoiFH+5s1e70XekYUfJDAFqbuMDOUTtCV1URpl2x1IOuasYnI2wEN7IKggc5KaX8ukKs7NeAwhT6QFexTOaf"
    "sIymkdsv4bNrW93j4DgHl3oaqjk+9/uJRfLz9//75cdv3rx/+11sLIxO8+xNT252I+odKZ8w2epugnXioHzTvWZ/89PPQ4nfd6/e"
    "/7i8wuWqh9cJNw+PtVSzp85h4ZlAkrk2FTzB/1UCHB6WngJzqMkCLYrVFAl5Fq4r4JtsaixCMW4T4kpTVGirIe4wQ118/osZ6uLd"
    "91ztPy3DECIOIRYxRSlC/GKJbMwhj+mdwtuG7/hX304URyw8WJZeowHHvkl1TS8dwG6/nutB3r57/92C7nEDHIwflGRHdJi+26EW"
    "tqxrbO8hr9yuZjeXMgFKM3kU+3X43sAbU9X2cAONkKOg3rraieBo3oBeVgXbcaPZLkPtJojarh/CaNsklH1gjfIeRelO++HukKW3"
    "Xb2iVX/GPlLdXdif4o9kj8ip7z5HH8td2C8D+4YElrfG3E64OAdjRbdTXibOixuIjxxH9obMpMPv2gNgfSdJ7D70azubKf1VzmEO"
    "z1MXgtb/Pqn0vz2/ipXh63fHIQmjFQcyDnIY/zAzcdLmCY2TvS6EuEjmBRQXV1x4cVGuC3ZezOtCZyNj03yUc1gWN8S2qyKLtw7T"
    "RXRWN9A6Dwnxa7NPSP3ZYEpUUXjijlUMnm3cHlSrHqTaPsYbHiRxhJFKd1+YvpXKYYOdy5QdEuwK1GzuMSTeC8Zl543TFVK5wZYt"
    "ClLj0liYyOwXhaVvmyB+SWaa1EeF4+aWFeKAZXYrcpXJqgpqafdJb2i6UD7z1tQ2AgP1BvbnopzPjQDDShXduvz5s8b9GboCP5+7"
    "Al8ebBW7penBQ3fcWA3uM4kcQtbOYqld194lIC5sihvy+s3ZWE8VmPvKKYgi88N3xJeFU8VdotCxLgmt/9YVSN70Yv6W8+n08wDE"
    "9PePfv78w8u3D0mn1HEeG8s3zU+juILoDFr309YYwkWZBaPG5CrRiK+zQ7F2dGHfBkJ9F1W5MzuxIddYw1kY1hHQuA0qlE4a8ez9"
    "JB3ZGPy++/15/ZB4vfvc0DL7nV+9+eHNj29ef/Trnz5evjzj25Knae5yhUiDFw12ELPEwl+7eOqPm/74uhm5xFBrfDtVuQTcqNup"
    "ur0OtEtTHFliO0wsLtrNhwg2mdvpg3LdmA9Co9tNXKYm2hjR3kvt1MXdxV7xt1XibpKMeljWdrpIuq/U2+ncZdNT8Fjd6mCEBg/i"
    "J/id+QnNxVRM5Dw3lKnfHM64AebtJiwHE+09dn1+QwjJdkCfPgHP5YsgQHLb/UVwDN2cBrsBC/1GMudkbYcmpEPS3S24ebWekfFV"
    "E7v+cNx23ld/XvbVn1+eH4nIdkBV31CAD68RnVcmd4nGPV+9U2oH5rzJ3eX7yXfzbvhkUKGNvz/89wW3474waIynW5G5nAh4cPP2"
    "XMgNMrfhjrTmzRp9Jdqm9o5XyHdx96AAbPGbh0NNtp3gba+4e4WsYrJiBsmteDiAcBw355zG9QRZnjggZm7FUaV3cC3cvCEeH594"
    "/66Ptxm1qG5yuc/vA1Tzm2KmJp4dP28+9jtlhnN2YKvcvDM47Qdfz4spd/hkJu8SM8eu+Z+fO8d2l2xr+RprYPKYrK+FGWOyHj+v"
    "FO5ntGbczC7gCyFcdBuxih3Q2bcBYm2wJVwsXMZ3+eRi0XUr+7k57ouJB5fO6bORedlnL/Ovi8+WDZ/H+bEadL1FWU9HjYa9Xfa/"
    "N5JFy53tevye8DXFf78dkBlR2a3Tk7J/TbX7exyFECYmnxqcbrf3uFRH3fatdq+yBgSOydR4aPujnHXA2Wowb34wPB6Us2Rs3OYB"
    "HwTFTdaJdddVeBvp5oNjrwxpnMtQw/zVUsP8vEL/J/bY37wDNCHdT/mQbOu+qkXJ5I0jzYhmBYPQbUSnKvhsMBOMOqEC3a4rMoR8"
    "PN71kLzxW1icYGEirjsqKfTG4vH8cyQnbSFwm1Zqlaq6dftrSqdujpcXbIPJJzXipne9vh27IiOBcivOdZLBsWOySi7QW2Xy6fwZ"
    "2GfFizAQPDNZlAyZWkIVW4ldm6aV8ALIrN28MTtlHtBFrLcJfD2UD8mFKi5JdBXXhcjfKAu/n1NlKq8P6H+ITgzQg4xW4NugokiN"
    "36ZqgdSocNXDiJ9Trv7znZ9KDoYEvngMlURKXCYJZCk3a5VyOdnDdXTZwyCNV9vMIHSqOws2mnw4YwBArk1mpDjJAvTK44TIxc1g"
    "B4oTDphJ152MATnbW/bDxeKcut7893a9+bcjG2jy6c+3k9KyHC5mXOYyZNjU5E2iDbzD0+PpJ+Tisr1s8Y9D5tlkaurEk9Aw6112"
    "a/nwj280rvv8cvkanM7rx6CPoDHenKqBl7fB5JBgDe9aRDKuNRQ82OyyL7pMY7sMTglez75IzxMyg+cJvUU3S6g4y4TtmCRkF9sT"
    "dxfcZFGgJNjPbk2aE2CXD5kNGWVQJmcXjQvXGWCo9o0rrmm72g5K3VuSUXaG67tkPFzJXJPbxfSbaQClftH05A65uLIoB4h4RaeD"
    "ggKTRQyD2D14fl3ulNXlwT0yqGQzAOBvybMQGQADJjs9j2kPBPMvuhuTB1/Pwbs5LY3pxiRI5MTEHmQntbE9ZaH83UluyEI8SHxM"
    "OSXXBwVRApN5JgLLFjJHh9nim1UdMbGQyIss3JHEHkqTmZLI/Jrsf44Wq1sSAYexkGeInCnL6/IyfcRCgyMpPGOyaauksEtiA63d"
    "nsuwANHXrvPYKjh2rKiuSTx0mY/vnOrsaAK08ZMyaglAcPwYv975OJ4LgG/C4Ki2juZjEteHXd84eMwUMpWPyXO5cimM62SQ3vzu"
    "XHd8uBZC00zD6MQ61c1gXSRncmF37ySfpJ/epW/KyW8Xak4CQARkbkGYDvP1xMd36RsgbHGbnJIr365K5MtuLjYuU/X3JKj+wdOK"
    "tCVHyi9z4E//84OrandZEyHzJXFPetq1ysN2wxQZcch6nczHCSPdZK4LmSMZ1o5TMaRa/HY8KEzmutHtC2+XfWysGtM+pvpViqo9"
    "LQwHVLeVqo8NvQLL4HELt2E7UaGoPrU5cTgdPhSycyH47zuZzXf//cHrevvOt/emnYrMPZSt23LYVN5+hfJQKOvqtiRUwunX0dSI"
    "s+KUvDMQI1sR4Wk7inh2WI/5gZNM7j+azW/OGplQC8uTrkveyxVHsjJanJs0IhraNG/OBJMaOKRM1vUM5e/47gkpFRzy25A7ZIU5"
    "wC9+c64ak5uMkORyghGzSSyzTQOYB9g8CinwdsVdJnsdWUxVcqWpysXQkttfQuFIdGBd/YNWi+Zd8sFJMP+4eMDxBbnv0+B6gZgN"
    "vqk0h6HDdYjV56rKgd3db6D/626BnYVetAZLPEPuvrQKgwnaZ1ya0EW+tPn3ehsZp8QPsMv6uaqtC715NxCBBsjb+TZOvB2Ph5r8"
    "98V1WFewIkvn2eFYFOAppzv/3WWEOronzU+OnSe5AfrFsXUVWxmMSH49a24OaehdlvshhX4oWNF1vTRa+sk1OkTdbne/Qa+383Vl"
    "itt5gKu7n02mJcwnGWfRCflwWS+nGgCGkYrbGUWRk+LHSXWf6fDriGV4XTw4SiBvfrJnLuzTDQEsRCcWRM0MRKeJzJSHmcNvy4Mi"
    "b3hwc0L7s6nh5Ku3rz76/Pn1XxcaPDSoq1L9wDmj/P8BW4UgNkYaZAT35OexSOWTOlGMSuXp8B8lI1E+JNzPIDGzHubyHYObyB8Y"
    "Chym1tKff//q9eKxbyhhcAt3Q2NeOpxrDNB3Tnhsv03kfc0iQDOmi+1jp0O7Pz47wA+4TEGjqov3JW12dpdYwL7UB++d3sH85keo"
    "VWSqTv961MDAzNtJF6ri4iOJm6yI54ClQNijRvGESo0i/oWEbrGNwp5HncPR1ARKQoTOfoYmcqnC4v8UactWQrPAdRZ50B440hb+"
    "tIlbLfKuRU62yNcWudwiz1vkgIv8cIE7biGWe6CdW0jpFsq6yGe3kN0tVHiRKG+h0VtJ9h4o+BaCvoW+b+b2W4j/FlrAhTRwoRRc"
    "CAcXOsKFrHChMlyIDhcaxIUkMRAoLuSKC/XiQsy40DYupI47iuTOK7iTz4kQks0fKls0rziRDbNK3IHkoagU7J7m86UfO+QXCiBH"
    "B5N5yGhoEtAajJbmNlJuqthILp4AIdrPi520ubELGo1RBZLUJKb6VhPZiLM7+W+90C0KIl+jNx9QqBjH032/NIkFlZKHO45Nff06"
    "nurH7QIpKKiWrR7Ft0MP64AGo2HkT6VuhPxt7oMBdqy6/1sALYjqW4ksmD39jQk6fD7yJy90twsX7kKUu9DoRo7dhYB3oeddyHsX"
    "al94JsVzckimPo18QgZOhHDgTWrnwDdGlBK5+N1XTNpH6TUiMBeiLtYeIbJ9a/XqqIGIWKJBII+dhYseSZrKT7TtyuFRLqK4yt4j"
    "OErRCYANjMS7L/8DPRCHf9vO344YUL/+lI2apfswpY/TKJK3O7FUPU1qY1hxCT3kpXpsF5SBxRuvMqvjBTBpN85E7vUxPS8yGxMP"
    "Yv76FjXs2e2KeTWgivrUbecAXsWdSCg2pgM8S3l8D9BXkysKUCENRldj52rXiN8/L7t3mVBin6d47wEqr92jjDuY/aRgyefgcXPi"
    "leR8xU8ziP/0XDjlg81ddkjibmEdpvmEVWfPASNlEO6SelC06yBf9D/dkETLXl69ksYvlPIL4fxCR7+Q1S9U9gvRfSHl2S7uYbJ2"
    "0TTfslhEq5/RHb+ljtlAE5a0C7ckfjS/lnCRKnJDH0NWwZ6d/aRIa32wI2e1NZynhpuGCMq60BJDVmXwnWI9nD7jmzhZwT128t97"
    "HoC7J+qSsrATzwNfKHaD8xAXVhNpcyaVa+NdTlBQYS4XBthIb7dS3820eIExL5LpRaK9mYIvcJtFTrHINzZTkUWWspU+MDILzhRj"
    "kU8rUG1FFq7I0BXIuyKvV+T8inxggSssEjgu5I6R+DGSQkbCyEgmGYkmV5qviX8oMBOtZDkzkcZKshEJOGZyjkjc8UDqEVgVZsaF"
    "yMYQmRoii0MgeIjcD4EXInJGzE7O3FT2x9/OFCyv/r7gvuP8AftfcVWfjkvRIHEyqA4TW+icPhGmG6iL2zhTTmjoOs6F3D4e5wAh"
    "sv3Q84fOxWhfT17wr989/7D0usca+AgeFYGlIuhUBKSKYFXCUFQgRY0BTS3lqQ5YRgRFYX4d1y09CAEw9qoO5wwfbMQrMLfFA0Uw"
    "2kQgJ0itNOCgSBpxCuYp7aMTjKeb16mmPtewJqL5bAOdqHCdqfGkshJWMEwZhtPh8MCwjPIxoNjqKEpTda0eoNY24SRNtliHBXIM"
    "FKY0NhETrM6VwISqd87xc8yWagPCuFyoWDu2jYqm1Lu0OTDB9TVWoePFRFVNWjLBYe1q0yCrMfClkbHBuGrRwPT1ZnS2tZwCDUPx"
    "qfDH9tFQA9Jkc5V39lSU0U8O7GJzLXf2N9TRM05AFwXAdrUGZZa27+wMKurGtxXGangY+6S7mrswQn9G6NwIPR1Lt8fUBxI6RELv"
    "SGwrmbcbyg9gUznFj+Vv2Nqn1hgq1+qNPmmoWrs2ukWJh6a21VSc00egEGXwAtkuQhmtYoY4N7w4a3oRxHfU+0zYODoomb0w7G5m"
    "Z03HIiYnLA4XZWkP8RidySfUHQ8CGarngsA7soGvR85gHb+d6K5+8fL6x+e3f136fgIqwoSXEJAUiL2gvrJDKC6MOpmeac4OCiQe"
    "BbwR325qpgSkdnMfYhPyy+5YeADDll0J93rqKgIadtdisA6I7qahkLK3Y/4lyXeBzaB2qukbWbPiv9SsdkLeSNqAgihw9G0C3N4E"
    "ee3K1u8xYwj8bMYyfv3q3cu3H312H+tv3/z4LyuY57LhWFL8gWpjL0SG/e2QSLcmMMfWltrj+xffijpcVMM+Cej7bqqXuSEJ4E+D"
    "ympiPr9V9MUI8P0Gz7mpTuaGEiMweehaGwN1awJuKrp2V86GFKQWgISp9nsSwvxQSbkBth5qOQDVVVOQ8laWaxugjsY14OdAyd0I"
    "37GrZj0D59nbPIrWNOyIGyde9SkmARXp4E1SugZiA9CO4nEMFBXqR5TZN1fWdtY49jhSskKwTuoJ4okneFmEQzcBkUJZ3+DjDSkL"
    "X7Rp0rojj9o05QFCfavYgfIFbw7niD188xav5OX8acBc3+AWOJYtC/+HVFU33zRNGQ1O0Pw3QmezBOGGhukmfvobuB8bmZmnRTcb"
    "V1/Nlf5/+UClP0o7AOfPhgsUr/XGwUMpBUD8uSEsbZsMxZ/NOQkkV12ELjc6613vkxBW70r43AiGYNxQ+FMag71qj8CI7rKsboNI"
    "gF0zGTxRXVYYipIg9qS/NeYCNfmzJB/fw6tQhF0hWXup++Ltvj0swGkXOdI0v7rKacPIzBWMf56aPT/558vbb55f/edKrNTZYX2I"
    "fIKAIVD6dVdPNRksdqAEqE0AEHKjM9syBoilkQfjlA3DrnpUwnhgjXlRkwolnmDEhUAe0LuHLLoJq060QzC6NvE90f858iNB0Upe"
    "FImNFtKjiRApkiWtVCSRoGMm7wi8HivlR6QDiVQhM43ISjGS8zW0TZkGuAq1kzpr89mqw84mSZV3x0/Takr/gU1l5pCK/FIRLiHQ"
    "qSxEK5GDZeZnmZ42H4if/m4GC3vz9ruVoAe4AFXRdIHhq7QVxc9I8xwCDSG+TxYWBHxLx4I4mcgRFkRDMNm/p+7e/w8jFUANZxyx"
    "DwxmHOhpDuL0xKmL0xrhKyK0RYS9sOCLO1o1KYXVNQoHxmRz7AyMSRInwBi9uSX0VzNE+Pevfnj1t7+9ev3y0+JmW0vNU1IULuPG"
    "qXgcHU53GlkOxFNS8bI+hIeS16ZOd6KBBX9zFCTagfCUvJ6RUDPJnUNUCKbNnWTEdNIAZQP1YUoemwV3X3L/EJHWJ6+DzMjeaTmi"
    "UPFphHjhGp9XxaO/nZcw368eXtGIne+pLBS8Pu3utplZ1f3drAX/SU3lJt13fvcYBTzNLhfRUJKemieiKjPjySsiT0uay2cH6/vu"
    "8XCcbrtHrY0+wVtxbEgh9Su+3Ucq4r5pxhujftOdFKKLHdcQ3W2KwwPqKJI9vErUwAk8RZ7h3LgznLGfDk+/wBseOSpQNJ4+Jp0P"
    "H8kuH2YMu2Xgx3TRoT+8BBaZr9Pzh/B5vJyV5bBp89JfAMuk5H+JPj54FJ4+KSZuXu7NJ/oqZYY+5ytLk7z+hmmOJBwiZf1S90WM"
    "KlrER3x7JBOTpxTupnmqXtaKWMwoRTRbAxd9xdt2KD5KKKBF+F9Dz224TdvBC0KxJFKZJhdi8zTN3TuygJG//1amxyDLkcqVDu0Q"
    "jxE1O55GxWBGcjF5oxBjavB4vQYfX+d32vB1p8/HfTUmZzAix1PqV8Ynn0+p+xfANLDJ9zhfvuo0Mhr9TLxyOPPc2ewcU8bKr/pU"
    "3z25lK4M3AG1MnoOKsTmU2AlILNokvJ+6JpKI0d7v/FT8hAVgpJQZr6ExlP5GlZ0mvKVG97x2KElNq7kkZzE0j291DlD9RVvo4CU"
    "yrVBk/B6EgryILpWKfgc/8NzGomqkdjPKwWavBwwo5/tWm/X68+433+aema+eP7h+R8/rS0z20YFJbcdYWGvBNusS1SVQkrZNKkk"
    "y/XcnRJPP6Gw1EtwNzjgjuy+0SPyKo+CgKMW80YWGkUmN0AqFo88oHzdM47MSnk+C815o3AE7VRJZZYbkhLeJbihJNXPqA1RIo9O"
    "b8Yi/OTBVaarirbVZo3h3pG0YRcJZRpD9NQ8AmZ8Gk+KlM1jSTXKKgivetgBguKkYOjEniQLj2khm7IqT56QYz+I9zUyHp08NYmy"
    "Tm92wFGaptKVp00ZZoS7fb0ycrn596D9K10ZSkuiedGKZViTKnKwz548ZZfIkfnxPp7l5eCm9exTrnYPbyAyrfXkqso+y0Z183oY"
    "DoeXnlhhjR9AjOrqjEvANXT8HdR9P/WrKMU47FRkagcXpOIHF2rg/JqFlnRSpoNlcPtIhR9Pk160EIPMlXRwxZ5+/JnZ6L0q13TP"
    "sC8zke7P3r5//fJqoQqMH7R+7DwQcZDWAYxj+1RHr5Lffu73/sP0Vl/98Oa/H/gfl8RHZPCI7B6R+WMhV4i8CxOOVoRHWqGTIqzS"
    "DLkU4ZgWpKYZxCngO625mpDGGR86jdC//XoaoX979fqRzAetBUaDKOTu+zB3oY9kpLA7C6tIhNCVbCpIinmchAWhvQqKDzH0LjZT"
    "6ypCUKSJMC8b1eEh+jxTuEpFFFRC9jS43O7HfGc86l8I6ProF0FTAz2iY3mCr/IQMmUGdWIRhtp9aqwZX3jr9voidCai+P2terqu"
    "lYEBWiEdDsRa8NlCKkWkSFjm9rQmgq7AErnwR87UkpF2cqWkvFvuXVWkrJjpykGxxbEfg1nLQliH4BuRhOrOg4T6pH4OhMH7+++K"
    "16Fexa7tQtkf6yLsuM/ijnv+6yPccKTaijRcMwNLJP0IfCArVchEIxIYRh7IR2ZikkhaEvhMItVJpEFZWYQW6pmLlQapK0WMSPJU"
    "vbMeMRvlC3exRm18i3TR3wBgtKoM2ZQySmHzhZ43yKUC09RMQhUJqgJ5VSS2WkivIiHWNGdz2fMc6vv5P1/+8v2qXSNoeqSoCuxV"
    "K7FVJL2aCbEiWVYk0ookW8uohPEKIxkHeRr/MDPLnIXZnOZ54SUKjEUzmVEkOgocSJEeaUF7n4HgI0j8CiAfweVn4PkISh8B61cw"
    "+wnnfprNGQrqy19P/AVvX717+7JE2Ow8ffIuRED2mOTQuDu9WpFZH5AwDgA+NW8sS4JrVsU3eT8Drd6QwgknDsrOQq07HONM6YBf"
    "XBXkSnArqziqd/ic+CUAnZ4wNJTu6k1uby3yevcBJJxUHUYgYXdQgL9qEmOc9z2EAFbToyHtwhiGA5vEnL1hfJre2FxHGTmWdYWn"
    "K0TbBqmJWPOE5IPVcMss6u8x5NNU/ebfp9P+N89/W+teAFw1EsKpkoL+8NqCTeVB6sy/nz+AVXKJeOGHi4QWV50ukLCs7cTbGTPw"
    "ytVb00TsTffM/gplmIc3cm78W68krk9X2w3YWEsZRbwnbnR4EfJmEgE32KRSkvNHI4iAimGJGRe9M9ZqPbzmL2MVlewpWvTV2pt4"
    "F20zyXtoD7ytYDqTRDXuIAJbvJuPfThXzxHLQ2QVJIBOA7LaG4gIEef9RweKCySe8CNHr9Ym7Lc6mp+OkRWwq2wzyGqkYsZl9FWd"
    "TC54q+dYEix4wDvX4rqKRqzPB1pBECN3rIetX3XUtajNZLs6uCzKfFxNpx6dToosi+oSn8A+ivkTijdlot7ZRH9pdjDs3mF6EpS2"
    "+VsAGbZ5g+eJ2pPRkTm+Dx9c4Ohfq9OaZGxGN1+PFUtOTcjAlfeC4gLQAyto97aqHXdSl645M9aD5WXuuE9z0cDURrMxEs5Fjig7"
    "p70OMLxggGX80xSDf377/N3751g7dmuHIwMQO6+BMRzIAEXySZyCjZlXGDtIOB6SAes1rgMxIstgv6GmAXJ32Vpn84C5A/RH9pxn"
    "Y4NgJtLGrQn0oCib2wg3lMW5Mcm6G8hpbuCv8b+nvPvfG7RWVtGNPW4nwIR/W2UvKNP4lrNPkPnXB/EqVD6Hy7x7lwj4ifFwFNDY"
    "yzF73rfxbUXyoY9jIQKKQDCyTbL6aLchc2ZY4tDRhwaMBb/fRpAF5d1PdjGfPhWjj5elB2Hmp0Xzp0+/uBbNn15+fIknREO0kQZR"
    "40EoPC1rOEl2CMOraEiVJcFHW93BiSO5SsIBDajZJMHTrw2x210JXgaBNwp3e8aREioWZlJYAEDBCDwrp7Yzj6TsKE91Jd8YZc7K"
    "nJoRofA1ekrs7D6VwTObQkc+WlVNyEJzZwxckPAVUlcaMOPZniLM7TrzUYGQpDgqKvAS63xMB9og4OyuWSbL4C14Smpes0sIZp/p"
    "skREE6hHq7CXXeyePEAG0yTHtN9hShV9zl1Fp31kfnMeZhaweZL6+cCvBnNMkhluu/Lt1p9qdpsmp+KHScnxhL5DQtqzu1Fu3LWc"
    "Zhfyk4ne6avn99+++uiTt8/frGlbNMNlre8CmNrMthQD/LJ/tzSYGLLWPbpuC7MxBUjfZRtsAZsJblptqMgWI/r25K2vZl0+FQXQ"
    "rTLUfnbImjxgTyRl163g/ryyt3g9rdaKN98o7Kg1J9DxgQ4BVRlVr2VHMNqMfhOqqhYqauXhWDW0rmb1xjV2jKqGo7Ec2fcePOWs"
    "2hCr5imoRlZdhAVFWY3cAPqZZbfA5kcRs/ZsQUkzVA9omZ+ySjeBIYz2hOTzm2lEN1bVMwve0KDsHRcNYbGcx+/Q4RrWx7pyllUV"
    "V1xYjHGdxjUclndc+XFXxB2z7qa40+ZdGHcoCP6SSq7lOeyMPlTo9yRyLvNMcO1wehJ+ubsO1QaFPgzs7KzyXzOSRh9OAQ1mVpKg"
    "IHqTVTVVANOfhQ9bUBebVZeLDvonx1hAExamVywnttKU2QDS3ug4QUPWkw5E9Nk/ZUU6CxtHnLoENXhZMIKlqd0JarHg0J6kHS0m"
    "WK4FDQlZXvKsAOaYxJ8mF+e3H3/083/87e37WHcAaLDhm1jTO4RG4SQj0kmukb4PfqSCcjIHW6Ch17jqTcAP+dYFfXtN9CVFRj0J"
    "S0gLJKgUgJA5iRMMOLwJBQLaJt2RDRycZVqgbdCjjK8J4zABNP//DMJ8y/i49VXCW4b3j58WPzsOyTxc1fSrCoDtReAvNX1phTkt"
    "IV9P/vBXf/HJlzMF8Ns3f/nLm2jywvjwT7vBinIAyhviAUXxgBu0XHF4XnIoqSLGrsEcV00dnDo3ggEKUlTFc0P1nSOA3MCNVZiv"
    "urGjRhWVQGjQyXQDSIvx8TjaNaR24U17O5IkMUndAIg6SbZ53GCFz5qlpm7gJwEDMj5nswmfxYbtqro/6vsuU9VytdlEVkAirZ8d"
    "azmhn8utXks6XIqFWEjopEsS2Rh4jNJK9v6p9HBDJ112KU9itwWUdfKa2NDnSNsbNRLoczwkshGQboNZT+iR3F28ew15lGwCPTAr"
    "FgNoLfTU+QsfUHBVUsUY0mKHmfc0vAs2yuQBcl3UfKb1gyXjTWuAyIKeI4Q0Cr29sQ7QwW473MAondUhfGNYwnHbGQIQPJahjT8V"
    "ucs3VJ4W1XTcEDQthF+6dfWNOUJ6rkMb3tB9U0T0cmvSCruqdxs54FSv29lvpv0wuOLmDTcHDX85OaeffvePv71bQ4aHnU/SCPwX"
    "ExD8Z7v+28RRBZuPbLPWBVMEV5dxTli0x3ljEfS5mFuLQgx2PD05ZgkMLZNKHmkvjxVlttptNCkzEI7c7rMWP4SRdgonLmES0DmM"
    "a4USXwszWeAhFaEmEW7FHifdueNzTp1CGeqFg7NKB3owdz8gITWdq2MVFXjK3uNZ2Aessneov7FQCxg2vEuycE0rPVMQb87KnyFW"
    "dj07q/Ny98OMHaZ6HLSHnl33YTNoe2GQC/JSWbia6GTEpnUmNOzoNMjVcnbTn+tmhpH52Z9mqudv/rHko6alRPWbWXjh/4Sx1cTw"
    "dAwNlUBjmx1Do+KSTguL7KKbt3q7GzTOpp452MRVzVYN3bxqLjEN4rBp28fUS2KzgVo91Z4Eq6uwtaRAGW9sOqFmZl8vtKuitWCU"
    "d1fiVL+5rhx9GFVQKFldbWisNf9IxDtYhCzE2VDGvPkfndgcqmGBJnAkSxirppOOQSyLcJUkRLzU5w4VlUe32oa+V896nDhg9zQy"
    "J3b4iqloT9fTOwKq3gUIS7aI5cL8CBy+ngps/CAlvzY4hqfzdSEYvIuxK+GtswjHjjbehdnfooYDK4CDjVAH1Tc2tlKwaZZOqmql"
    "iU887gM6KairVZNNSm5SflElhq3w6de/mrTuu+9fvfnb4lYDseOJeZVaWdO0XVx3Kg8GK6U8vkoIncGkaDaNMiqI7biw6W+6/JT2"
    "VEbCptnfVBEsmhdNnQXjRmAGBSUpCvEQnKu4p2GpWkb1yZjoApq6K9ONFIREA6hIdQuRSLHrxqiyUoy9oNhRvmJBWl6AiXQTdt3Z"
    "wklWOSqFeRhYVDmGnSsApoIeDmmSgv7pJNS8gjoBr9qjm5N09ABAzOJLm38FpCaPK6PCsvUl5/OQD4q5ophHCimmkH2KmamYtIr5"
    "rJDqilmwJUOG+j39SbfP7Fo4OyoVi+rP9z4KEytKiJNOX9TYoxxyioHpGEV6AMW3evLJ2lv32p/UQVvhnFzBmftkHiMcw+U/tzD+"
    "Zkpw/uI/X33z5v27WN30kAWcM4QheQjWJSth82ABSiFPDxag6tMbChI+s/YPfmYYgnl44tCtwzoP+TUV0+vPgbjffT6zZfz4/MOr"
    "h/KgClzzdLXkLBIffl6SyvMoKVftQlXUC0Xh0A1BHc3zFCZwntk45+t6mAcxDnBBBMnbh+D4W/1qV8A3PzmjB/NDXnHLDpvEKgYr"
    "DmNjgcJQDfs4KbS3YXaTdwWh6rUcHxw2l6bZ+P0vJxyA3393n4glIGoBiZuznSaCmsqWgu69CYEFmBQmkYT3JCqv6DRP4syKHwhg"
    "MzcBA+mXm4hekVJkp3lBRaiaUwpKc5TFKzgtVU1ZcB5kL/syW0fqGN0bArKIQsJZL0XLVKS0HRImheZCARy3CxWFeorMAPbF/8bK"
    "6Ap3BuIUT2moXJtvLqeC+P6mbDw8WNHQTEM8TcyXf5wm5su/P05MGO+yabzlIBjJQFZwbWM2jcsgSvlkJk503CeTT6KMPkmGIyOE"
    "mR/dMxOr2ylFic7qvLDrZE8LYX7puczz15MB8bM3P71+9fzR/YM/+tXL238uLDqROTOyagbCzcjFufJ0zhyekd8zcn8GWtDIGBrI"
    "RCPPaOQgjfykq1T2WUpIelevMbteGRxszt2V0ENYy6BD3cpoNUvsl1f+dxquOZ712Uwl+Obtu+8/+uL5Lx+ga3pgrA9k9pHnPpBu"
    "Bzrumak7snhHPt7IN7ty0f4r1tqdbMCyq9lAt+l5xFDMblfjA7Kuncj96zXHl86H1ZcztdPL2zVftNDUrqS1KNhNFyN6VWG/XlIV"
    "GnkT9qHXJ1bgUSSRoLc+ag4ywVbqIJHnqtalHQur+SUsLOd7731gZ9g1/HLTNSys7q7GOQhmOY9EsmMJqlO/Ac8L0HV9Gbt1WNcR"
    "ZwOj/7AT/05VqOxg7Ppsoui5xIdzvtFtYO+lOU0oDdk0zGwc9GE+uBd0rdfRaGlrCLcU+zMRfQ7xPZPm7iJ4rg5eTwpZ5zghh6yD"
    "ga27Mu7YsJnjPo864F9qi8h7PHEiT0tv3um//XSOXL9+9/L65bu3MXgduaNnVukHvumZiTpyVKd+4ekcUhP5+BC7cWQ+nnRVVGMP"
    "gxkGOk7CNEFh7h6mdXznNESf/W7q8/nszU9v/vtheBbVE8i0I9H2QsIdHh6WW1yJcZXOKziu7nXlx10Rd8yym8JOi7sw7tBZX0Zd"
    "uurZMSwzCNbXU2P01/dRfPXt87c4zL9+883zd0vahNxyyQlDu2Euen+XFZ+cw1qGBEM/jUoVs6X1b3beMgnR6Y9szsw4PWPW7L+Y"
    "pv2rjz/66v23S91iMBkzwlGq5sqIiLF0MHPtq8Af1d7SigQsksFEXk6pY4KfSBsj26kisozKJ/FNWg4AIQQV1Fs32XkomGOtsPr/"
    "aTTTomjpSThBGTamV7ZXiwSrSBiwYU8iA8xIhgqoiiiv3qYM2hr4JLp2YlgRWGM78Tlee7cOVFZ1W/nAk+BFLf5rXlTXGOxPREHI"
    "x/TvEy3NGsxmfba7jMwOB2mTzY+ORLdpAaS9y5ovzb/ArHmshaYAOFzFTUlAerRd0rYPL7mwq1ShM4vDoWjH07AZgYekgMvGRsN6"
    "XVNHR0HXTBJLbCl8yVN/dtiX1TloU/sHwjkh0DOFgJbg0BQ2CgGlJdQ0BaFidAq9V0d78GyCzxO8oeAnBQ/q2ij3DfY//w9pYxS6"
)


def render_figure2(yaml_path=None, session_json_path=None, out_stem="communication_neuroprostheses_reproduction", verify_overlaps=True, export_html=None):
    """Render the full 8-study benchmark overview figure (Figure 2).

    Loads the benchmark YAML, syncs its JSON mirror, draws panels (a)
    vocabulary funnel through (f) full-pipeline WER including the
    funnel and reproduction curly-brace annotations, optionally verifies
    that no text bounding boxes collide, and saves the figure as
    ``<out_stem>.png`` and ``<out_stem>.pdf``.

    Parameters
    ----------
    yaml_path : str, optional
        Path to ``reproduction_benchmark.yaml``. Defaults to the copy in
        this script's directory.
    session_json_path : str, optional
        Unused since the 2026-11-11 phase-diagram round moved panel (g)
        (session-level PER vs. WER) to ``render_phase_memory_figure.py``
        panel (a); kept in the signature for backwards compatibility with
        external callers (e.g., the review repo's ``main_fig_2`` wrapper).
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
    # 2026-11-11 phase-diagram round (user: "fig2g已经不需要出现在fig2上了，
    # 删掉"): the session-level PER-vs-WER phase diagram (then lettered (g))
    # leaves Figure 2 entirely -- it lives on as panel (a) of
    # render_phase_memory_figure.py -- so the figure drops from 5 rows
    # (22.9 in) to 4 (17.1 in), keeping rows 1-3 at their tuned sizes.
    fig = plt.figure(figsize=(22.4, 17.1))
    outer = fig.add_gridspec(
        4, 1, height_ratios=(1.02, 0.08, 0.96, 0.92), hspace=0.72
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
                  "English communication-neuroprostheses studies",
                  fontsize=10, fontweight="bold")

    # ---- Row 2: (b) Dataset size, (c) WER, (d) Speed ----
    # 2026-11-09 star round (user, timeline repo: starred the 3 studies
    # satisfying Open Data + Vocab > 71.4k + WER < 12.8% -- Fan et al., 2023,
    # Card et al., 2024, Jude et al., 2026a -- in the timeline HTML's copy of
    # this upper half): "\u2605 " prefixes the tick labels of those studies in
    # (b)/(c)/(d), a star glyph rides above each starred (c) bar, the (c)
    # title's second line becomes the star key, and the reproduction-bracket
    # drops below turn solid+bold for starred studies, dashed+faded otherwise.
    # The timeline identifies rows by study id (fan2023/card2024/jude2026a);
    # these YAML rows carry study_key instead, and both jude2026a sessions
    # (T17/T18) star together just as both id-tagged rows do there.
    # 2026-11-12 star rounds (user, both repos: star additionally requires
    # vocal paralysis, then online decoding): a study must have online
    # (closed-loop) decoding and at least one vocal-paralysis participant
    # (△), dropping Fan et al., 2023 (T5 speaks naturally ▲, offline rows
    # aside) and excluding offline AI-conference open-data reanalyses
    # (Feng/Li/Zhang ...), keeping Card et al., 2024 (T15△) and
    # Jude et al., 2026a (T17△/T18▲) -- both online.
    STAR_KEYS = {"Card et al., 2024", "Jude et al., 2026a"}
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
    ax_b.set_xticklabels(
        [("\u2605 " + r["label"]) if r["study_key"] in STAR_KEYS
         else r["label"] for r in rows_b],
        fontsize=6.4, rotation=75, ha="right", rotation_mode="anchor")
    ax_b.set_xlim(-0.75, len(rows_b) - 0.25)
    ax_b.set_ylabel("released sentences")
    # 2026-11-06 same-height-title round (user: "for each row of fig2,
    # make the top of panels of that row have the same height"): the
    # axes tops already align in every row (same gridspec slots;
    # (g)'s equal-aspect shrink takes width, not height), but in row 2
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
        if r["study_key"] in STAR_KEYS:
            ax2.text(x, top + 18.8, "\u2605", ha="center", va="bottom",
                     fontsize=7.8, fontweight="bold")
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
    c_badges = []
    for asr_v, asr_lab, ly, lx_lead in (
            (2.5, "2.5% LibriSpeech (Radford et al., 2022)", 62.6, 4.55),
            (5.8, "5.8% Switchboard (Xiong et al., 2017)", 68.1, 3.44),
            (12.8, "12.8% 14-dataset average (Radford et al., 2022)",
             73.6, 2.50)):
        ax2.plot([lx_lead, lx_lead], [asr_v, ly - 0.2], color="#999999",
                 lw=0.55, linestyle="-", zorder=2)
        _t_c = ax2.text(5.50, ly, asr_lab, ha="right", va="bottom",
                        fontsize=5.6, color="#666666", zorder=3,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                  ec="none", alpha=0.88))
        c_badges.append((asr_v, asr_lab, ly, lx_lead, _t_c))
    ax2.set_ylim(0, 125)
    ax2.set_yticks(range(0, 101, 25))
    ax2.set_xticks(range(len(rows_cd)))
    ax2.set_xticklabels(
        [("\u2605 " + r["label"]) if r["study_key"] in STAR_KEYS
         else r["label"] for r in rows_cd],
        fontsize=6.4, rotation=75, ha="right", rotation_mode="anchor")
    ax2.set_xlim(-0.75, len(rows_cd) - 0.25)
    ax2.set_ylabel("word error rate (%)")
    ax2.set_title("(c) Word error rate (online)\n"
                  "\u2605 2 studies satisfy: Open Data, Online, Vocab > 71.4k, "
                  "WER < 12.8%, Vocal paralysis \u25b3",
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
    # 2026-11-10 semilogy + handwriting WPM round (user: "把fig2d的速度改成
    # semilogy。再去找一下正常人handwriting wpm（要靠谱文章）。记得把html也更新。"):
    # panel (d) switches to a logarithmic y-axis (1.5–420 wpm, 1-2-5 log
    # ticks 2–200 with minor ticks) and adds the able-bodied handwriting
    # reference line at 25.1 wpm (Kaifosh et al., 2025, Nature 645:702–711,
    # doi:10.1038/s41586-025-09255-w, n = 75 participants handwriting
    # prompted phrases at 5 chars/word).
    for x, r in enumerate(rows_cd):
        spd = r["speed_wpm_mean"]
        spdsd = r["speed_wpm_sd"]
        vocab = r["vocab_tag"]
        s_cal = r["speed_caliber"]
        state_c = OPEN_C[r["openness_code"]]
        if spd is None:
            ax3.text(x, 1.75, "n/r", ha="center", va="bottom",
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
        ax3.text(x, top * 1.07, note, ha="center", va="bottom",
                 fontsize=6.8)
        ax3.text(x, top * 1.32,
                 f"vocab {vocab}\n{s_cal}",
                 ha="center", va="bottom", fontsize=4.6, color="#555555",
                 linespacing="normal")
    LX = 3.92
    d_badges = []
    for spd_v, spd_lab, above in (
            (160, "160 wpm natural conversation (Card et al., 2024)", True),
            (51.6, "51.6 wpm keyboard typing (Dhakal et al., 2018)", True),
            (36.2, "36.2 wpm smartphone typing (Palin et al., 2019)", True),
            (25.1, "25.1 wpm handwriting (Kaifosh et al., 2025)", True),
            (10, "5\u201310 wpm eye typing (Hamid & Kristensson, 2024)", True),
            (8, "8 wpm point-and-click BCI (Willett et al., 2021)", False)):
        ax3.axhline(spd_v, color=C_NORM, linewidth=0.9, linestyle=(0, (4, 2)))
        yv = spd_v * 1.04 if above else spd_v / 1.05
        _t_d = ax3.text(LX, yv, spd_lab, ha="left",
                        va="bottom" if above else "top",
                        fontsize=5.6, color="#666666", zorder=3,
                        bbox=dict(boxstyle="round,pad=0.12", fc="white",
                                  ec="none", alpha=0.88))
        d_badges.append((yv, spd_lab, above, _t_d))
    ax3.set_yscale("log")
    ax3.set_ylim(1.5, 420)
    ax3.set_yticks([2, 5, 10, 20, 50, 100, 200])
    ax3.set_yticklabels(["2", "5", "10", "20", "50", "100", "200"])
    ax3.yaxis.set_minor_locator(
        matplotlib.ticker.FixedLocator(
            [3, 4, 6, 7, 8, 9, 30, 40, 60, 70, 80, 90, 300, 400]
        )
    )
    ax3.set_xticks(range(len(rows_cd)))
    ax3.set_xticklabels(
        [("\u2605 " + r["label"]) if r["study_key"] in STAR_KEYS
         else r["label"] for r in rows_cd],
        fontsize=6.4, rotation=75, ha="right", rotation_mode="anchor")
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
    # 2026-11-11 split round (user: fig2 is too long to publish -- the
    # RAM/VRAM panels moved to render_phase_memory_figure.py panels c/d,
    # with a new WER-vs-CER companion alongside); the phase-diagram round
    # then retired the former row-4 (g) PER-vs-WER panel to the companion
    # figure's panel (a), so (f) is now the last row of Figure 2.
    gs_r4 = outer[3].subgridspec(1, 2, width_ratios=(1.0, 1.0), wspace=0.16)
    ax_h = fig.add_subplot(gs_r4[0, 0])
    ax_i = fig.add_subplot(gs_r4[0, 1])

    fc_ours = (*to_rgb(C_SPEECH_I), 0.35)
    c_ref = OPEN_C[2]
    fc_ref = (*to_rgb(c_ref), 0.35)

    x_h, e_spans, e_seps = _panel_e_group_geometry(per)
    w = 0.36
    ax_h.bar(x_h - w / 2, [p["ours"] for p in per], w,
             label="our reproduction (16 GB home PC)",
             color=fc_ours, edgecolor=C_SPEECH_I, linewidth=0.8)
    ax_h.bar(x_h + w / 2, [p["paper"] for p in per], w,
             label="paper / benchmark anchor",
             color=fc_ref, edgecolor=c_ref, linewidth=0.8)
    for i, p in enumerate(per):
        xe = x_h[i]
        o_val, o_sd = p["ours"], p.get("ours_sd")
        p_val, p_sd = p["paper"], p.get("paper_sd")
        top_o = o_val + o_sd if o_sd is not None else o_val
        top_p = p_val + p_sd if p_sd is not None else p_val
        if o_sd is not None:
            ax_h.errorbar(xe - w / 2, o_val,
                          yerr=[[min(o_val, o_sd)], [o_sd]],
                          color="#333333", capsize=1.8,
                          elinewidth=0.9, capthick=0.9, zorder=4)
        if p_sd is not None:
            ax_h.errorbar(xe + w / 2, p_val,
                          yerr=[[min(p_val, p_sd)], [p_sd]],
                          color="#333333", capsize=1.8,
                          elinewidth=0.9, capthick=0.9, zorder=4)
        o_txt = f"{o_val:.1f}\u00b1{o_sd:.1f}" if o_sd is not None else f"{o_val:.1f}"
        p_txt = f"{p_val:.1f}\u00b1{p_sd:.1f}" if p_sd is not None else f"{p_val:.1f}"
        ax_h.text(xe - w / 2 - 0.08, top_o + 1.1, o_txt,
                  ha="center", va="bottom", fontsize=5.0)
        ax_h.text(xe + w / 2 + 0.08, top_p + 1.1, p_txt,
                  ha="center", va="bottom", fontsize=5.0)
        if p["note"] is not None:
            ymax_h = max(top_o, top_p)
            ax_h.text(xe, ymax_h + 6.2, p["note"], ha="center", va="bottom",
                      fontsize=4.8, color="#555555")
    ax_h.set_xticks(x_h)
    ax_h.set_xticklabels([p["label"] for p in per], fontsize=6.0, rotation=75,
                         ha="right", rotation_mode="anchor")
    ax_h.set_xlim(-0.6, float(x_h.max()) + 0.6)
    # Dotted separators rise from the axis base to the header band at
    # y=100; the bold metric headers sit inside the axes top band
    # (y 100.5 up), above every bar/value/note (max ~96.8) and below
    # the legend / "error bar" note. They cannot hang below the tick
    # labels: row-4 axes patches start right under them and paint white
    # over anything hanging past the label tips.
    for x_sep in e_seps:
        ax_h.axvline(x_sep, color="#999999", linewidth=0.8,
                     linestyle=":", ymax=100.0 / 118.0)
    for (g_lo, g_hi), g_key in zip(e_spans, PANEL_E_GROUPS):
        ax_h.text(0.5 * (g_lo + g_hi), 100.5, PANEL_E_GROUP_TITLES[g_key],
                  ha="center", va="bottom", fontsize=6.8,
                  fontweight="bold", color="#444444")
    ax_h.set_ylabel("Greedy PER / CER / 1\u2212r (%)")
    ax_h.set_ylim(0, 118)
    ax_h.set_yticks(range(0, 101, 25))
    ax_h.legend(frameon=False, fontsize=6.6, loc="upper left")
    ax_h.text(0.98, 0.97, "error bar is calculated across days",
              transform=ax_h.transAxes, ha="right", va="top",
              fontsize=6.0, color="#555555")
    ax_h.set_title("(e) Reproduction: acoustic / synthesis stage (no LM), "
                   "grouped by metric: PER / CER / 1\u2212r",
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

    fig.legend(handles=make_color_legend_handles(), loc="upper center",
               bbox_to_anchor=(0.512, 0.992), ncol=6, frameon=False,
               fontsize=8.4, handlelength=1.4, columnspacing=1.5)
    fig.subplots_adjust(left=0.045, right=0.985, top=0.952, bottom=0.10)

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
        drop_pts_c.append((rows_cd[i], x_pt, bb.y0 - 0.0025))
    y_bus_c = lbl_y0_rep - 0.28 * corridor_rep
    # 2026-11-09 star round: SVG stroke widths 1.85/0.95 map 1:1 to points;
    # dash pattern 3.2,2.4 svg units -> (2.3, 1.7) pt; marker radii 3.2/1.8
    # svg units -> markersizes 4.4/2.5 pt, with a 1.7 pt white core dot on
    # the starred drops.
    for r, x_pt, y_top in drop_pts_c:
        if r["study_key"] in STAR_KEYS:
            fig.add_artist(Line2D(
                [x_pt, x_pt], [y_top, y_bus_c],
                transform=fig.transFigure, color=rep_c,
                linewidth=1.85, alpha=0.98, solid_capstyle="round", zorder=5
            ))
            fig.add_artist(Line2D(
                [x_pt], [y_top],
                transform=fig.transFigure, marker="o", markersize=4.4,
                color=rep_c, alpha=0.98, zorder=6
            ))
            fig.add_artist(Line2D(
                [x_pt], [y_top],
                transform=fig.transFigure, marker="o", markersize=1.7,
                color="white", zorder=7
            ))
        else:
            fig.add_artist(Line2D(
                [x_pt, x_pt], [y_top, y_bus_c],
                transform=fig.transFigure, color=rep_c,
                linewidth=0.95, alpha=0.45, linestyle=(0, (2.3, 1.7)),
                solid_capstyle="round", zorder=5
            ))
            fig.add_artist(Line2D(
                [x_pt], [y_top],
                transform=fig.transFigure, marker="o", markersize=2.5,
                color=rep_c, alpha=0.48, zorder=6
            ))
    fig.add_artist(Line2D(
        [drop_pts_c[0][1], drop_pts_c[-1][1]], [y_bus_c, y_bus_c],
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
        "Our reproduction on a 64 GB RAM / RTX 5060 Ti (16 GB) home PC",
        ha="right", va="center", fontsize=7.4, fontweight="bold",
        color=rep_c
    )

    if verify_overlaps:
        bad = check_figure_overlaps(fig)
        if bad:
            plt.close(fig)
            msg = "\n".join(f"  {pct:5.1f}%: {a!r} <-> {b!r}" for a, b, pct in bad)
            raise RuntimeError(f"Overlap check failed:\n{msg}")

    if export_html is None:
        export_html = (
            os.path.basename(out_stem) == "communication_neuroprostheses_reproduction"
        )
    f2_svg_tuple = None
    if export_html:
        f2_svg_tuple = _build_fig2_full_svg(
            fig=fig,
            renderer=renderer,
            inv_fig=inv_fig,
            axes_dict=dict(
                a=ax1, b=ax_b, c=ax2, d=ax3,
                e=ax_h, f=ax_i,
            ),
            data=data,
            c_badges=c_badges,
            d_badges=d_badges,
            funnel_geom=dict(
                selected_idx=selected_idx,
                drop_points=drop_points,
                y_bus=y_bus, y_apex=y_apex, y_arm=y_arm, y_tip=y_tip,
                brace_x0=brace_x0, brace_x1=brace_x1, brace_xm=brace_xm,
            ),
            repro_geom=dict(
                drop_pts_c=drop_pts_c,
                y_bus_c=y_bus_c, y_apex_r=y_apex_r, y_arm_r=y_arm_r, y_tip_r=y_tip_r,
                brace_x0r=brace_x0r, brace_x1r=brace_x1r, brace_xmr=brace_xmr,
            ),
        )

    fig.savefig(out_stem + ".png", bbox_inches="tight")
    fig.savefig(out_stem + ".pdf", bbox_inches="tight")
    plt.close(fig)

    if export_html and f2_svg_tuple is not None:
        export_interactive_html(
            out_html=out_stem + ".html",
            f2_svg_tuple=f2_svg_tuple,
            verify_overlaps=verify_overlaps,
        )
    return len(rows_b), len(per)




def _poly_pts(cx, cy, r, n, rot_deg=90.0):
    """Build an SVG polygon points string for a regular n-gon."""
    pts = []
    for i in range(n):
        a = math.radians(rot_deg + i * 360.0 / n)
        pts.append(f"{cx + r * math.cos(a):.2f},{cy - r * math.sin(a):.2f}")
    return " ".join(pts)


def _star_pts(cx, cy, r_out, r_in, n=5, rot_deg=90.0):
    """Build an SVG polygon points string for a 5-pointed star."""
    pts = []
    for i in range(2 * n):
        r = r_out if i % 2 == 0 else r_in
        a = math.radians(rot_deg + i * 180.0 / n)
        pts.append(f"{cx + r * math.cos(a):.2f},{cy - r * math.sin(a):.2f}")
    return " ".join(pts)


def _build_fig2_full_svg(
    fig,
    renderer,
    inv_fig,
    axes_dict,
    data,
    c_badges,
    d_badges,
    funnel_geom,
    repro_geom,
):
    """Convert the live 6-panel Figure 2 Matplotlib figure into interactive SVG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Rendered Figure 2 instance at 200 DPI.
    renderer : matplotlib.backend_bases.RendererBase
        Active Agg renderer of ``fig``.
    inv_fig : matplotlib.transforms.Transform
        Inverse of ``fig.transFigure``.
    axes_dict : dict of str to matplotlib.axes.Axes
        Mapping of panel keys ``'a'``..``'f'`` to their Matplotlib axes.
    data : dict
        Parsed ``reproduction_benchmark.yaml`` dictionary.
    c_badges, d_badges : list
        Badge text objects captured from Panels (c) and (d).
    funnel_geom, repro_geom : dict
        Curly-bracket geometry captured from Row 1->2 and Row 2->3.

    Returns
    -------
    tuple of (str, int, int)
        ``(svg_inner, vw, vh)`` for ``<svg id="f2-svg">``.
    """
    w_px, h_px = fig.canvas.get_width_height()
    vw = int(round(w_px * 0.5))
    PT = 1.3889
    SW = lambda pt: round(pt * PT, 2)
    SX = lambda px: px * 0.5
    SY = lambda py: (h_px - py) * 0.5
    esc = html_lib.escape

    def d2svg(ax, dx, dy):
        px, py = ax.transData.transform((dx, dy))
        return SX(px), SY(py)

    def f2svg(fx, fy):
        px, py = fig.transFigure.transform((fx, fy))
        return SX(px), SY(py)

    def multiline_svg(
        sx, sy, text, fs_pt, color,
        anchor="middle", va="bottom", weight="normal", cls=""
    ):
        lines = str(text).split("\n")
        fs = fs_pt * PT
        lh = fs * 1.20
        cls_attr = f' class="{cls}"' if cls else ""
        w_attr = f' font-weight="{weight}"' if weight != "normal" else ""
        if len(lines) == 1:
            if va == "bottom":
                yy = sy - 0.12 * fs
                dom = "alphabetic"
            elif va == "top":
                yy = sy + 0.82 * fs
                dom = "alphabetic"
            else:
                yy = sy
                dom = "central"
            return (f'<text{cls_attr} x="{sx:.1f}" y="{yy:.1f}" font-size="{fs:.2f}" '
                    f'fill="{color}"{w_attr} text-anchor="{anchor}" '
                    f'dominant-baseline="{dom}">{esc(lines[0])}</text>')
        out_t = [f'<text{cls_attr} x="{sx:.1f}" font-size="{fs:.2f}" fill="{color}"'
                 f'{w_attr} text-anchor="{anchor}">']
        n_l = len(lines)
        for k, line in enumerate(lines):
            if va == "bottom":
                yy = sy - (n_l - 1 - k) * lh - 0.14 * fs
            elif va == "top":
                yy = sy + k * lh + 0.82 * fs
            else:
                yy = sy + (k - 0.5 * (n_l - 1)) * lh + 0.32 * fs
            out_t.append(f'<tspan x="{sx:.1f}" y="{yy:.1f}">{esc(line)}</tspan>')
        out_t.append("</text>")
        return "".join(out_t)

    def rot_xtick_svg(tx, ty, label, fs_pt, bold=False, color="#1f2937"):
        lines = str(label).split("\n")
        fs = fs_pt * PT
        lh = fs * 1.15
        w_attr = ' font-weight="bold"' if bold else ""
        if len(lines) == 1:
            return (f'<g transform="translate({tx:.1f},{ty:.1f}) rotate(-75)">'
                    f'<text class="f2-lbl" x="0" y="{0.34 * fs:.1f}" font-size="{fs:.2f}" '
                    f'fill="{color}"{w_attr} text-anchor="end">{esc(lines[0])}</text></g>')
        spans = []
        for k, line in enumerate(lines):
            yy = (k - 0.35 * (len(lines) - 1)) * lh + 0.34 * fs
            spans.append(f'<tspan x="0" y="{yy:.1f}">{esc(line)}</tspan>')
        return (f'<g transform="translate({tx:.1f},{ty:.1f}) rotate(-75)">'
                f'<text class="f2-lbl" font-size="{fs:.2f}" fill="{color}"{w_attr} '
                f'text-anchor="end">{"".join(spans)}</text></g>')

    def draw_axes_chrome(
        ax, yticks, yticklabels, ylabel_str, title_str,
        title_fs=9.8, minor_yticks=None, xticks=None, xticklabels=None, xlabel_str=None
    ):
        bb = ax.get_window_extent(renderer)
        ax_x0, ax_x1 = SX(bb.x0), SX(bb.x1)
        ax_y0, ax_y1 = SY(bb.y1), SY(bb.y0)
        c = [
            f'<line x1="{ax_x0:.1f}" y1="{ax_y0:.1f}" x2="{ax_x0:.1f}" y2="{ax_y1:.1f}" '
            f'stroke="#1f2937" stroke-width="{SW(0.8)}"/>',
            f'<line x1="{ax_x0:.1f}" y1="{ax_y1:.1f}" x2="{ax_x1:.1f}" y2="{ax_y1:.1f}" '
            f'stroke="#1f2937" stroke-width="{SW(0.8)}"/>',
        ]
        for yv, ylab in zip(yticks, yticklabels):
            _, sy = d2svg(ax, 0, yv)
            c.append(f'<line x1="{ax_x0 - 3.5:.1f}" y1="{sy:.1f}" x2="{ax_x0:.1f}" y2="{sy:.1f}" '
                     f'stroke="#1f2937" stroke-width="{SW(0.8)}"/>')
            c.append(f'<text x="{ax_x0 - 6.0:.1f}" y="{sy:.1f}" font-size="{9.0 * PT:.2f}" '
                     f'fill="#1f2937" text-anchor="end" dominant-baseline="central">{esc(ylab)}</text>')
        if minor_yticks:
            for yv in minor_yticks:
                _, sy = d2svg(ax, 0, yv)
                c.append(f'<line x1="{ax_x0 - 2.0:.1f}" y1="{sy:.1f}" x2="{ax_x0:.1f}" y2="{sy:.1f}" '
                         f'stroke="#1f2937" stroke-width="{SW(0.6)}"/>')
        if xticks is not None and xticklabels is not None:
            for xv, xlab in zip(xticks, xticklabels):
                sx_t, _ = d2svg(ax, xv, 0)
                c.append(f'<line x1="{sx_t:.1f}" y1="{ax_y1:.1f}" x2="{sx_t:.1f}" y2="{ax_y1 + 3.5:.1f}" '
                         f'stroke="#1f2937" stroke-width="{SW(0.8)}"/>')
                c.append(f'<text x="{sx_t:.1f}" y="{ax_y1 + 13.0:.1f}" font-size="{9.0 * PT:.2f}" '
                         f'fill="#1f2937" text-anchor="middle" dominant-baseline="central">{esc(xlab)}</text>')
        if xlabel_str:
            xl_cx = 0.5 * (ax_x0 + ax_x1)
            c.append(f'<text x="{xl_cx:.1f}" y="{ax_y1 + 30.0:.1f}" font-size="{9.0 * PT:.2f}" '
                     f'fill="#1f2937" text-anchor="middle" dominant-baseline="central">{esc(xlabel_str)}</text>')
        yl_bb = ax.yaxis.label.get_window_extent(renderer)
        yl_x = SX(0.5 * (yl_bb.x0 + yl_bb.x1))
        yl_y = 0.5 * (ax_y0 + ax_y1)
        c.append(f'<text x="{yl_x:.1f}" y="{yl_y:.1f}" font-size="{9.0 * PT:.2f}" fill="#1f2937" '
                 f'text-anchor="middle" dominant-baseline="central" '
                 f'transform="rotate(-90 {yl_x:.1f} {yl_y:.1f})">{esc(ylabel_str)}</text>')
        t_lines = [ln for ln in title_str.split("\n") if ln != ""]
        t_cx = 0.5 * (ax_x0 + ax_x1)
        t_fs = title_fs * PT
        t_lh = t_fs * 1.22
        t_top_y = ax_y0 - 24.5 if len(title_str.split("\n")) > 1 else ax_y0 - 11.5
        for k, ln in enumerate(t_lines):
            c.append(f'<text x="{t_cx:.1f}" y="{t_top_y + k * t_lh:.1f}" font-size="{t_fs:.2f}" '
                     f'font-weight="bold" fill="#111827" text-anchor="middle">{esc(ln)}</text>')
        return "\n".join(c), ax_x0, ax_x1, ax_y0, ax_y1

    rows_a = data["panel_a_vocab_funnel"]
    rows_b = data["panel_b_dataset_size"]
    rows_cd = data["panel_cd_online_wer_speed"]
    per = data["panel_e_repro_acoustic"]
    wer = data["panel_f_repro_full_wer"]
    STAR_IDS = {"card2024", "jude2026a"}

    open_how = {
        "Willett et al., 2021": "brain-to-text\nwithout\nlanguage model",
        "Wairagkar et al., 2025": "brain-to-voice",
        "Levin et al., 2026": "brain-to-text\nwithout\nlanguage model",
        "Wairagkar et al., 2026": "brain-to-voice",
    }
    dagger_note = (
        "\u2020 recomputed from released data (decode outputs / trial timing / Source Data files)\n"
        "\u2021 digitized from the paper's figure markers\n"
        "error bar is calculated across days"
    )

    # Background rect is prepended after the body so that vh (measured
    # from the drawn elements; see the return below) is known.
    s = []

    # Top 6-item clickable legend
    leg_items = [
        ("open", OPEN_C[2], "open data"),
        ("partial", OPEN_C[1], "data on request"),
        ("closed", OPEN_C[0], "closed data"),
        ("unknown", OPEN_C["P"], "preprint"),
        ("reanalysis", OPEN_C["R"], "reanalysis on open data"),
    ]
    leg_y = 20.0
    leg_xs = [595.0, 730.0, 900.0, 1040.0, 1160.0]
    for (val, col, lab), lx in zip(leg_items, leg_xs):
        s.append(f'<g class="lg-row" data-dim="data_availability" data-val="{val}" style="cursor:pointer">')
        s.append(f'<rect x="{lx:.1f}" y="{leg_y - 5.5:.1f}" width="17" height="10" '
                 f'fill="{col}" fill-opacity="0.35" stroke="{col}" stroke-width="{SW(0.8)}"/>')
        s.append(f'<text x="{lx + 23:.1f}" y="{leg_y:.1f}" font-size="{8.4 * PT:.2f}" '
                 f'fill="#1f2937" dominant-baseline="central">{esc(lab)}</text>')
        s.append("</g>")
    rep_c = C_SPEECH_I
    s.append('<g id="lg-repro-toggle" style="cursor:pointer">')
    s.append('<title>Click to toggle showing only the 8 reproduced open-data studies</title>')
    s.append(f'<rect x="1385.0" y="{leg_y - 5.5:.1f}" width="17" height="10" '
             f'fill="{rep_c}" fill-opacity="0.35" stroke="{rep_c}" stroke-width="{SW(0.8)}"/>')
    s.append(f'<text x="1408.0" y="{leg_y:.1f}" font-size="{8.4 * PT:.2f}" '
             f'fill="#1f2937" dominant-baseline="central">our reproduction</text>')
    s.append("</g>")

    # ---- Panel (a) ----
    ax1 = axes_dict["a"]
    minor_locs = [m * 10 ** k for k in range(0, 6) for m in range(2, 10) if m * 10 ** k >= 2.5]
    ch_a, a_x0, a_x1, a_y0, a_y1 = draw_axes_chrome(
        ax1,
        [10 ** k for k in range(0, 7)],
        ["1", "10", "100", "1,000", "10,000", "100,000", "1,000,000"],
        "vocabulary size",
        f"(a) Maximum vocabulary size used in the {len(rows_a)} English communication-neuroprostheses studies",
        title_fs=10.0,
        minor_yticks=minor_locs,
    )
    s.append(ch_a)
    for anchor_v, anchor_lab, above in (
            (11100, "11,100 word families", False),
            (42000, "42,000 lemmas", False),
            (71400, "71,400 word forms", True)):
        _, sy_line = d2svg(ax1, 0, anchor_v)
        s.append(f'<line x1="{a_x0:.1f}" y1="{sy_line:.1f}" x2="{a_x1:.1f}" y2="{sy_line:.1f}" '
                 f'stroke="#7F7F7F" stroke-width="{SW(0.9)}" stroke-dasharray="5.5,2.8"/>')
        lx, ly = d2svg(ax1, -0.55, anchor_v * (1.12 if above else 0.72))
        s.append(multiline_svg(
            lx, ly, anchor_lab + " (Brysbaert et al., 2016)",
            7.0, "#555555", anchor="start", va="bottom" if above else "top"
        ))
    for x, r in enumerate(rows_a):
        sid = PANEL_A_IDS[x]
        v = r["vocab_num"]
        is_open = r["is_open_vocab"]
        col = OPEN_C[r["openness_code"]]
        bx0, by0 = d2svg(ax1, x - 0.38, v)
        bx1, by1 = d2svg(ax1, x + 0.38, 2.5)
        bw, bh = bx1 - bx0, by1 - by0
        cx = 0.5 * (bx0 + bx1)
        dash = ' stroke-dasharray="4.2,2.8"' if is_open else ""
        fop = "0.18" if is_open else "0.35"
        lbl_clean = r["label"].replace("\n", " ")
        s.append(f'<g class="f2-slot" data-id="{sid}" data-panel="a">')
        s.append(f'<title>{esc(lbl_clean)} \u2014 vocabulary: {esc(r["vocab_display"])} (click to inspect study)</title>')
        s.append(f'<rect class="f2-hit" fill="none" pointer-events="all" x="{bx0 - 3:.1f}" y="{a_y0:.1f}" width="{bw + 6:.1f}" height="{a_y1 - a_y0 + 135:.1f}"/>')
        s.append(f'<rect class="f2-bar" x="{bx0:.1f}" y="{by0:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                 f'fill="{col}" fill-opacity="{fop}" stroke="{col}" stroke-width="{SW(0.8)}"{dash}/>')
        _, vy = d2svg(ax1, x, v * 1.15)
        s.append(multiline_svg(cx, vy, r["vocab_display"], 7.6, "#1f2937", anchor="middle", va="bottom", cls="f2-val"))
        how = open_how.get(r["study_key"]) if is_open else None
        if how is not None:
            _, hy = d2svg(ax1, x, v * 0.85)
            s.append(multiline_svg(cx, hy, how, 4.6, "#555555", anchor="middle", va="top"))
        s.append(f'<line x1="{cx:.1f}" y1="{a_y1:.1f}" x2="{cx:.1f}" y2="{a_y1 + 3.5:.1f}" '
                 f'stroke="#1f2937" stroke-width="{SW(0.8)}"/>')
        s.append(rot_xtick_svg(cx, a_y1 + 6.0, r["label"], 6.7))
        s.append("</g>")

    # ---- Green Funnel Curly Bracket (Row 1 -> Row 2) ----
    guide_c = OPEN_C[2]
    _, sy_bus = f2svg(0, funnel_geom["y_bus"])
    _, sy_apex = f2svg(0, funnel_geom["y_apex"])
    _, sy_arm = f2svg(0, funnel_geom["y_arm"])
    _, sy_tip = f2svg(0, funnel_geom["y_tip"])
    sx_b0, _ = f2svg(funnel_geom["brace_x0"], 0)
    sx_b1, _ = f2svg(funnel_geom["brace_x1"], 0)
    sx_bm, _ = f2svg(funnel_geom["brace_xm"], 0)

    for idx_sel, (x_pt, y_top) in zip(funnel_geom["selected_idx"], funnel_geom["drop_points"]):
        sid_f = PANEL_A_IDS[idx_sel]
        sx_pt, sy_top = f2svg(x_pt, y_top)
        s.append(f'<g class="f2-drop" data-id="{sid_f}">')
        s.append(f'<line x1="{sx_pt:.1f}" y1="{sy_top:.1f}" x2="{sx_pt:.1f}" y2="{sy_bus:.1f}" '
                 f'stroke="{guide_c}" stroke-width="{SW(1.15)}" stroke-linecap="round" opacity="0.85"/>')
        s.append(f'<circle cx="{sx_pt:.1f}" cy="{sy_top:.1f}" r="2.4" fill="{guide_c}" opacity="0.9"/>')
        s.append("</g>")

    sx_first, _ = f2svg(funnel_geom["drop_points"][0][0], 0)
    sx_last, _ = f2svg(funnel_geom["drop_points"][-1][0], 0)
    s.append(f'<line x1="{sx_first:.1f}" y1="{sy_bus:.1f}" x2="{sx_last:.1f}" y2="{sy_bus:.1f}" '
             f'stroke="{guide_c}" stroke-width="{SW(1.25)}" stroke-linecap="round" opacity="0.88"/>')
    s.append(f'<line x1="{sx_bm:.1f}" y1="{sy_bus:.1f}" x2="{sx_bm:.1f}" y2="{sy_apex:.1f}" '
             f'stroke="{guide_c}" stroke-width="{SW(1.4)}" stroke-linecap="round" opacity="0.92"/>')
    dx_c = 0.016 * vw
    brace_d = (
        f"M {sx_b0:.1f} {sy_tip:.1f} "
        f"C {sx_b0:.1f} {sy_arm:.1f}, {sx_b0 + 0.35 * dx_c:.1f} {sy_arm:.1f}, {sx_b0 + dx_c:.1f} {sy_arm:.1f} "
        f"L {sx_bm - dx_c:.1f} {sy_arm:.1f} "
        f"C {sx_bm - 0.35 * dx_c:.1f} {sy_arm:.1f}, {sx_bm:.1f} {sy_arm:.1f}, {sx_bm:.1f} {sy_apex:.1f} "
        f"C {sx_bm:.1f} {sy_arm:.1f}, {sx_bm + 0.35 * dx_c:.1f} {sy_arm:.1f}, {sx_bm + dx_c:.1f} {sy_arm:.1f} "
        f"L {sx_b1 - dx_c:.1f} {sy_arm:.1f} "
        f"C {sx_b1 - 0.35 * dx_c:.1f} {sy_arm:.1f}, {sx_b1:.1f} {sy_arm:.1f}, {sx_b1:.1f} {sy_tip:.1f}"
    )
    s.append(f'<path d="{brace_d}" fill="none" stroke="{guide_c}" stroke-width="{SW(1.45)}" '
             f'stroke-linecap="round" stroke-linejoin="round" opacity="0.92"/>')
    sx_ftxt, sy_ftxt = f2svg(funnel_geom["brace_xm"] - 0.008, 0.5 * (funnel_geom["y_bus"] + funnel_geom["y_arm"]) + 0.001)
    s.append(multiline_svg(
        sx_ftxt, sy_ftxt,
        "Filtered to 8 open-data studies exceeding native-speaker vocabulary (>71,400 word forms)",
        7.4, guide_c, anchor="end", va="center", weight="bold"
    ))

    # ---- Panel (b) ----
    ax_b = axes_dict["b"]
    ch_b, b_x0, b_x1, b_y0, b_y1 = draw_axes_chrome(
        ax_b,
        list(range(0, 12001, 2000)),
        [f"{v:,}" for v in range(0, 12001, 2000)],
        "released sentences",
        "(b) Open-sourced dataset size\n",
        title_fs=9.8,
    )
    s.append(ch_b)
    for x, r in enumerate(rows_b):
        sid = PANEL_BCD_IDS[x]
        n_sent = r["released_sentences"]
        col = OPEN_C[r["openness_code"]]
        is_star = sid in STAR_IDS
        lbl_txt = ("\u2605 " + r["label"]) if is_star else r["label"]
        bx0, by0 = d2svg(ax_b, x - 0.38, n_sent)
        bx1, by1 = d2svg(ax_b, x + 0.38, 0)
        bw, bh = bx1 - bx0, by1 - by0
        cx = 0.5 * (bx0 + bx1)
        s.append(f'<g class="f2-slot" data-id="{sid}" data-panel="b">')
        s.append(f'<title>{esc(r["label"])} \u2014 {esc(r["sentences_display"])} released sentences (click to inspect study)</title>')
        s.append(f'<rect class="f2-hit" fill="none" pointer-events="all" x="{bx0 - 3:.1f}" y="{b_y0:.1f}" width="{bw + 6:.1f}" height="{b_y1 - b_y0 + 125:.1f}"/>')
        s.append(f'<rect class="f2-bar" x="{bx0:.1f}" y="{by0:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                 f'fill="{col}" fill-opacity="0.35" stroke="{col}" stroke-width="{SW(0.8)}"/>')
        _, vy = d2svg(ax_b, x, n_sent + 220)
        s.append(multiline_svg(cx, vy, r["sentences_display"], 7.3, "#1f2937",
                               anchor="middle", va="bottom", cls="f2-val"))
        _, ny = d2svg(ax_b, x, n_sent + 1150)
        s.append(multiline_svg(cx, ny, r["span_note"], 4.6, "#555555", anchor="middle", va="bottom"))
        s.append(f'<line x1="{cx:.1f}" y1="{b_y1:.1f}" x2="{cx:.1f}" y2="{b_y1 + 3.5:.1f}" '
                 f'stroke="#1f2937" stroke-width="{SW(0.8)}"/>')
        s.append(rot_xtick_svg(cx, b_y1 + 6.0, lbl_txt, 6.4))
        s.append("</g>")

    # ---- Panel (c) ----
    ax2 = axes_dict["c"]
    ch_c, c_x0, c_x1, c_y0, c_y1 = draw_axes_chrome(
        ax2,
        list(range(0, 101, 25)),
        [str(v) for v in range(0, 101, 25)],
        "word error rate (%)",
        "(c) Word error rate (online)\n"
        "\u2605 2 studies satisfy: Open Data, Online, Vocab > 71.4k, WER < 12.8%, Vocal paralysis \u25b3",
        title_fs=9.8,
    )
    s.append(ch_c)
    for asr_v in (2.5, 5.8, 12.8):
        _, sy_l = d2svg(ax2, 0, asr_v)
        s.append(f'<line x1="{c_x0:.1f}" y1="{sy_l:.1f}" x2="{c_x1:.1f}" y2="{sy_l:.1f}" '
                 f'stroke="#7F7F7F" stroke-width="{SW(0.9)}" stroke-dasharray="5.5,2.8"/>')
    for asr_v, asr_lab, ly, lx_lead, t_obj in c_badges:
        sx_ld, sy_l0 = d2svg(ax2, lx_lead, asr_v)
        _, sy_l1 = d2svg(ax2, lx_lead, ly - 0.2)
        s.append(f'<line x1="{sx_ld:.1f}" y1="{sy_l0:.1f}" x2="{sx_ld:.1f}" y2="{sy_l1:.1f}" '
                 f'stroke="#999999" stroke-width="{SW(0.55)}"/>')
        bp = t_obj.get_bbox_patch().get_window_extent(renderer)
        s.append(f'<rect x="{SX(bp.x0):.1f}" y="{SY(bp.y1):.1f}" width="{bp.width * 0.5:.1f}" '
                 f'height="{bp.height * 0.5:.1f}" rx="2.5" fill="#ffffff" fill-opacity="0.88"/>')
        tx, ty = d2svg(ax2, 5.50, ly)
        s.append(multiline_svg(tx, ty, asr_lab, 5.6, "#666666", anchor="end", va="bottom"))
    s.append(multiline_svg(
        c_x0 + 0.98 * (c_x1 - c_x0), c_y0 + 0.015 * (c_y1 - c_y0),
        dagger_note, 6.0, "#555555", anchor="end", va="top"
    ))
    for x, r in enumerate(rows_cd):
        sid = PANEL_BCD_IDS[x]
        wmean = r["wer_mean"]
        wsd = r["wer_sd"]
        col = OPEN_C[r["openness_code"]]
        is_star = sid in STAR_IDS
        lbl_txt = ("\u2605 " + r["label"]) if is_star else r["label"]
        bx0, by0 = d2svg(ax2, x - 0.38, wmean)
        bx1, by1 = d2svg(ax2, x + 0.38, 0)
        bw, bh = bx1 - bx0, by1 - by0
        cx = 0.5 * (bx0 + bx1)
        top = wmean + wsd if wsd is not None else wmean
        note = f"{wmean:.1f}\u00b1{wsd:.1f}" if wsd is not None else f"{wmean:.1f}"
        s.append(f'<g class="f2-slot" data-id="{sid}" data-panel="c">')
        s.append(f'<title>{esc(r["label"])} \u2014 online WER: {esc(note)}% (click to inspect study)</title>')
        s.append(f'<rect class="f2-hit" fill="none" pointer-events="all" x="{bx0 - 3:.1f}" y="{c_y0:.1f}" width="{bw + 6:.1f}" height="{c_y1 - c_y0 + 125:.1f}"/>')
        s.append(f'<rect class="f2-bar" x="{bx0:.1f}" y="{by0:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                 f'fill="{col}" fill-opacity="0.35" stroke="{col}" stroke-width="{SW(0.8)}"/>')
        if wsd is not None:
            _, ey_hi = d2svg(ax2, x, wmean + wsd)
            _, ey_lo = d2svg(ax2, x, max(0.0, wmean - wsd))
            s.append(f'<line x1="{cx:.1f}" y1="{ey_hi:.1f}" x2="{cx:.1f}" y2="{ey_lo:.1f}" '
                     f'stroke="#333333" stroke-width="{SW(1.0)}"/>')
            s.append(f'<line x1="{cx - 2.8:.1f}" y1="{ey_hi:.1f}" x2="{cx + 2.8:.1f}" y2="{ey_hi:.1f}" '
                     f'stroke="#333333" stroke-width="{SW(1.0)}"/>')
            s.append(f'<line x1="{cx - 2.8:.1f}" y1="{ey_lo:.1f}" x2="{cx + 2.8:.1f}" y2="{ey_lo:.1f}" '
                     f'stroke="#333333" stroke-width="{SW(1.0)}"/>')
        _, vy = d2svg(ax2, x, top + 1.6)
        s.append(multiline_svg(cx, vy, note, 6.6, "#1f2937",
                               anchor="middle", va="bottom", cls="f2-val"))
        _, ny = d2svg(ax2, x, top + 7.0)
        s.append(multiline_svg(cx, ny, f"vocab {r['vocab_tag']}\n{r['wer_caliber']}", 4.6, "#555555", anchor="middle", va="bottom"))
        if is_star:
            _, sy_st = d2svg(ax2, x, top + 18.8)
            s.append(multiline_svg(cx, sy_st, "\u2605", 7.8, "#1f2937", anchor="middle", va="bottom"))
        s.append(f'<line x1="{cx:.1f}" y1="{c_y1:.1f}" x2="{cx:.1f}" y2="{c_y1 + 3.5:.1f}" '
                 f'stroke="#1f2937" stroke-width="{SW(0.8)}"/>')
        s.append(rot_xtick_svg(cx, c_y1 + 6.0, lbl_txt, 6.4))
        s.append("</g>")

    # ---- Panel (d) ----
    ax3 = axes_dict["d"]
    ch_d, d_x0, d_x1, d_y0, d_y1 = draw_axes_chrome(
        ax3,
        [2, 5, 10, 20, 50, 100, 200],
        ["2", "5", "10", "20", "50", "100", "200"],
        "words per minute",
        "(d) Communication speed (online)\nn/r: speed not reported",
        title_fs=9.8,
        minor_yticks=[3, 4, 6, 7, 8, 9, 30, 40, 60, 70, 80, 90, 300, 400],
    )
    s.append(ch_d)
    LX = 3.92
    for spd_v in (160, 51.6, 36.2, 25.1, 10, 8):
        _, sy_l = d2svg(ax3, 0, spd_v)
        s.append(f'<line x1="{d_x0:.1f}" y1="{sy_l:.1f}" x2="{d_x1:.1f}" y2="{sy_l:.1f}" '
                 f'stroke="#7F7F7F" stroke-width="{SW(0.9)}" stroke-dasharray="5.5,2.8"/>')
    for ly_d, spd_lab, above, t_obj in d_badges:
        bp = t_obj.get_bbox_patch().get_window_extent(renderer)
        s.append(f'<rect x="{SX(bp.x0):.1f}" y="{SY(bp.y1):.1f}" width="{bp.width * 0.5:.1f}" '
                 f'height="{bp.height * 0.5:.1f}" rx="2.5" fill="#ffffff" fill-opacity="0.88"/>')
        tx, ty = d2svg(ax3, LX, ly_d)
        s.append(multiline_svg(tx, ty, spd_lab, 5.6, "#666666", anchor="start",
                               va="bottom" if above else "top"))
    s.append(multiline_svg(
        d_x0 + 0.98 * (d_x1 - d_x0), d_y0 + 0.015 * (c_y1 - c_y0),
        dagger_note, 6.0, "#555555", anchor="end", va="top"
    ))
    for x, r in enumerate(rows_cd):
        sid = PANEL_BCD_IDS[x]
        spd = r["speed_wpm_mean"]
        spdsd = r["speed_wpm_sd"]
        col = OPEN_C[r["openness_code"]]
        is_star = sid in STAR_IDS
        lbl_txt = ("\u2605 " + r["label"]) if is_star else r["label"]
        bx0, _ = d2svg(ax3, x - 0.38, 1.5)
        bx1, by1 = d2svg(ax3, x + 0.38, 1.5)
        bw = bx1 - bx0
        cx = 0.5 * (bx0 + bx1)
        s.append(f'<g class="f2-slot" data-id="{sid}" data-panel="d">')
        if spd is None:
            s.append(f'<title>{esc(r["label"])} \u2014 speed not reported (click to inspect study)</title>')
            s.append(f'<rect class="f2-hit" fill="none" pointer-events="all" x="{bx0 - 3:.1f}" y="{d_y0:.1f}" width="{bw + 6:.1f}" height="{d_y1 - d_y0 + 125:.1f}"/>')
            _, nry = d2svg(ax3, x, 1.75)
            s.append(multiline_svg(cx, nry, "n/r", 5.6, "#777777", anchor="middle", va="bottom", cls="f2-val"))
        else:
            _, by0 = d2svg(ax3, x - 0.38, spd)
            bh = by1 - by0
            top = spd + spdsd if spdsd is not None else spd
            note = f"{spd:.1f}\u00b1{spdsd:.1f}" if spdsd is not None else f"{spd:.1f}"
            s.append(f'<title>{esc(r["label"])} \u2014 online speed: {esc(note)} wpm (click to inspect study)</title>')
            s.append(f'<rect class="f2-hit" fill="none" pointer-events="all" x="{bx0 - 3:.1f}" y="{d_y0:.1f}" width="{bw + 6:.1f}" height="{d_y1 - d_y0 + 125:.1f}"/>')
            s.append(f'<rect class="f2-bar" x="{bx0:.1f}" y="{by0:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                     f'fill="{col}" fill-opacity="0.35" stroke="{col}" stroke-width="{SW(0.8)}"/>')
            if spdsd is not None:
                _, ey_hi = d2svg(ax3, x, spd + spdsd)
                _, ey_lo = d2svg(ax3, x, max(1.5, spd - spdsd))
                s.append(f'<line x1="{cx:.1f}" y1="{ey_hi:.1f}" x2="{cx:.1f}" y2="{ey_lo:.1f}" '
                         f'stroke="#333333" stroke-width="{SW(1.0)}"/>')
                s.append(f'<line x1="{cx - 2.8:.1f}" y1="{ey_hi:.1f}" x2="{cx + 2.8:.1f}" y2="{ey_hi:.1f}" '
                         f'stroke="#333333" stroke-width="{SW(1.0)}"/>')
                s.append(f'<line x1="{cx - 2.8:.1f}" y1="{ey_lo:.1f}" x2="{cx + 2.8:.1f}" y2="{ey_lo:.1f}" '
                         f'stroke="#333333" stroke-width="{SW(1.0)}"/>')
            _, vy = d2svg(ax3, x, top * 1.07)
            s.append(multiline_svg(cx, vy, note, 6.8, "#1f2937",
                                   anchor="middle", va="bottom", cls="f2-val"))
            _, ny = d2svg(ax3, x, top * 1.32)
            s.append(multiline_svg(cx, ny, f"vocab {r['vocab_tag']}\n{r['speed_caliber']}", 4.6, "#555555", anchor="middle", va="bottom"))
        s.append(f'<line x1="{cx:.1f}" y1="{d_y1:.1f}" x2="{cx:.1f}" y2="{d_y1 + 3.5:.1f}" '
                 f'stroke="#1f2937" stroke-width="{SW(0.8)}"/>')
        s.append(rot_xtick_svg(cx, d_y1 + 6.0, lbl_txt, 6.4))
        s.append("</g>")

    # ---- Blue Reproduction Curly Bracket (Row 2 -> Row 3) ----
    _, sy_bus_c = f2svg(0, repro_geom["y_bus_c"])
    _, sy_apex_r = f2svg(0, repro_geom["y_apex_r"])
    _, sy_arm_r = f2svg(0, repro_geom["y_arm_r"])
    _, sy_tip_r = f2svg(0, repro_geom["y_tip_r"])
    sx_b0r, _ = f2svg(repro_geom["brace_x0r"], 0)
    sx_b1r, _ = f2svg(repro_geom["brace_x1r"], 0)
    sx_bmr, _ = f2svg(repro_geom["brace_xmr"], 0)

    drop_svg_c = []
    for i_c, (_r_c, x_pt_c, y_top_c) in enumerate(repro_geom["drop_pts_c"]):
        sid_c = PANEL_BCD_IDS[i_c]
        sx_pt_c, sy_top_c = f2svg(x_pt_c, y_top_c)
        drop_svg_c.append((sid_c, sx_pt_c, sy_top_c))
        is_star = sid_c in STAR_IDS
        s.append(f'<g class="f2-drop" data-id="{sid_c}">')
        if is_star:
            s.append(f'<line x1="{sx_pt_c:.1f}" y1="{sy_top_c:.1f}" x2="{sx_pt_c:.1f}" y2="{sy_bus_c:.1f}" '
                     f'stroke="{rep_c}" stroke-width="{SW(1.85)}" stroke-linecap="round" opacity="0.98"/>')
            s.append(f'<circle cx="{sx_pt_c:.1f}" cy="{sy_top_c:.1f}" r="3.2" fill="{rep_c}" opacity="0.98"/>')
            s.append(f'<circle cx="{sx_pt_c:.1f}" cy="{sy_top_c:.1f}" r="1.2" fill="#ffffff"/>')
        else:
            s.append(f'<line x1="{sx_pt_c:.1f}" y1="{sy_top_c:.1f}" x2="{sx_pt_c:.1f}" y2="{sy_bus_c:.1f}" '
                     f'stroke="{rep_c}" stroke-width="{SW(0.95)}" stroke-dasharray="3.2,2.4" stroke-linecap="round" opacity="0.45"/>')
            s.append(f'<circle cx="{sx_pt_c:.1f}" cy="{sy_top_c:.1f}" r="1.8" fill="{rep_c}" opacity="0.48"/>')
        s.append("</g>")

    s.append(f'<line x1="{drop_svg_c[0][1]:.1f}" y1="{sy_bus_c:.1f}" '
             f'x2="{drop_svg_c[-1][1]:.1f}" y2="{sy_bus_c:.1f}" '
             f'stroke="{rep_c}" stroke-width="{SW(1.25)}" stroke-linecap="round" opacity="0.88"/>')
    s.append(f'<line x1="{sx_bmr:.1f}" y1="{sy_bus_c:.1f}" x2="{sx_bmr:.1f}" y2="{sy_apex_r:.1f}" '
             f'stroke="{rep_c}" stroke-width="{SW(1.4)}" stroke-linecap="round" opacity="0.92"/>')
    brace_r_d = (
        f"M {sx_b0r:.1f} {sy_tip_r:.1f} "
        f"C {sx_b0r:.1f} {sy_arm_r:.1f}, {sx_b0r + 0.35 * dx_c:.1f} {sy_arm_r:.1f}, {sx_b0r + dx_c:.1f} {sy_arm_r:.1f} "
        f"L {sx_bmr - dx_c:.1f} {sy_arm_r:.1f} "
        f"C {sx_bmr - 0.35 * dx_c:.1f} {sy_arm_r:.1f}, {sx_bmr:.1f} {sy_arm_r:.1f}, {sx_bmr:.1f} {sy_apex_r:.1f} "
        f"C {sx_bmr:.1f} {sy_arm_r:.1f}, {sx_bmr + 0.35 * dx_c:.1f} {sy_arm_r:.1f}, {sx_bmr + dx_c:.1f} {sy_arm_r:.1f} "
        f"L {sx_b1r - dx_c:.1f} {sy_arm_r:.1f} "
        f"C {sx_b1r - 0.35 * dx_c:.1f} {sy_arm_r:.1f}, {sx_b1r:.1f} {sy_arm_r:.1f}, {sx_b1r:.1f} {sy_tip_r:.1f}"
    )
    s.append(f'<path d="{brace_r_d}" fill="none" stroke="{rep_c}" stroke-width="{SW(1.45)}" '
             f'stroke-linecap="round" stroke-linejoin="round" opacity="0.92"/>')
    sx_rtxt, sy_rtxt = f2svg(repro_geom["brace_xmr"] - 0.012, 0.5 * (repro_geom["y_bus_c"] + repro_geom["y_arm_r"]))
    s.append(multiline_svg(
        sx_rtxt, sy_rtxt,
        "Our reproduction on a 64 GB RAM / RTX 5060 Ti (16 GB) home PC",
        7.4, rep_c, anchor="end", va="center", weight="bold"
    ))

    # Helper for paired-bar legends in Panels (e)-(h)
    def draw_inpanel_legend(ax_x0, ax_y0, items, fs_pt=6.5):
        fs = fs_pt * PT
        lh = fs * 1.38
        lx = ax_x0 + 12.0
        ly0 = ax_y0 + 10.0
        for k, (kind, col, lab) in enumerate(items):
            cy = ly0 + k * lh
            if kind == "patch":
                s.append(f'<rect x="{lx:.1f}" y="{cy - 4.2:.1f}" width="14.5" height="8.4" '
                         f'fill="{col}" fill-opacity="0.35" stroke="{col}" stroke-width="{SW(0.8)}"/>')
            elif kind == "dash":
                s.append(f'<line x1="{lx:.1f}" y1="{cy:.1f}" x2="{lx + 14.5:.1f}" y2="{cy:.1f}" '
                         f'stroke="{col}" stroke-width="{SW(1.0)}" stroke-dasharray="5.5,2.8"/>')
            s.append(f'<text x="{lx + 20.0:.1f}" y="{cy:.1f}" font-size="{fs:.2f}" '
                     f'fill="#1f2937" dominant-baseline="central">{esc(lab)}</text>')

    # ---- Panel (e): Reproduction acoustic / synthesis stage ----
    ax_h = axes_dict["e"]
    ch_e, e_x0, e_x1, e_y0, e_y1 = draw_axes_chrome(
        ax_h,
        list(range(0, 101, 25)),
        [str(v) for v in range(0, 101, 25)],
        "Greedy PER / CER / 1\u2212r (%)",
        "(e) Reproduction: acoustic / synthesis stage (no LM), grouped by metric: PER / CER / 1\u2212r",
        title_fs=9.3,
    )
    s.append(ch_e)
    draw_inpanel_legend(e_x0, e_y0, [
        ("patch", C_SPEECH_I, "our reproduction (16 GB home PC)"),
        ("patch", OPEN_C[2], "paper / benchmark anchor"),
    ], fs_pt=6.6)
    s.append(multiline_svg(
        e_x0 + 0.98 * (e_x1 - e_x0), e_y0 + 0.03 * (e_y1 - e_y0),
        "error bar is calculated across days", 6.0, "#555555", anchor="end", va="top"
    ))
    x_e, e_spans, e_seps = _panel_e_group_geometry(per)
    for x_sep in e_seps:
        sx_a, sy_a = d2svg(ax_h, x_sep, 0.0)
        sx_b, sy_b = d2svg(ax_h, x_sep, 100.0)
        s.append(f'<line x1="{sx_a:.1f}" y1="{sy_a:.1f}" x2="{sx_b:.1f}" y2="{sy_b:.1f}" '
                 f'stroke="#999999" stroke-width="{SW(0.8)}" stroke-dasharray="1.4,2.2"/>')
    for (g_lo, g_hi), g_key in zip(e_spans, PANEL_E_GROUPS):
        hx, hy = d2svg(ax_h, 0.5 * (g_lo + g_hi), 100.5)
        s.append(multiline_svg(hx, hy, PANEL_E_GROUP_TITLES[g_key], 6.8,
                               "#444444", anchor="middle", va="bottom",
                               weight="bold"))
    w = 0.36
    for i, p in enumerate(per):
        xe = x_e[i]
        sid = PANEL_BCD_IDS[i]
        o_val, o_sd = p["ours"], p.get("ours_sd")
        p_val, p_sd = p["paper"], p.get("paper_sd")
        top_o = o_val + o_sd if o_sd is not None else o_val
        top_p = p_val + p_sd if p_sd is not None else p_val
        o_txt = f"{o_val:.1f}\u00b1{o_sd:.1f}" if o_sd is not None else f"{o_val:.1f}"
        p_txt = f"{p_val:.1f}\u00b1{p_sd:.1f}" if p_sd is not None else f"{p_val:.1f}"
        bx0_l, by0_l = d2svg(ax_h, xe - w, o_val)
        bx1_l, by1_l = d2svg(ax_h, xe, 0)
        bx0_r, by0_r = d2svg(ax_h, xe, p_val)
        bx1_r, _ = d2svg(ax_h, xe + w, 0)
        cx_slot = 0.5 * (bx1_l + bx0_r)
        cx_l = 0.5 * (bx0_l + bx1_l)
        cx_r = 0.5 * (bx0_r + bx1_r)
        lbl_clean = p["label"].replace("\n", " ")
        s.append(f'<g class="f2-slot" data-id="{sid}" data-panel="e">')
        s.append(f'<title>{esc(lbl_clean)} \u2014 Acoustic/synthesis: Ours {esc(o_txt)}% vs. Paper {esc(p_txt)}% (click to inspect study)</title>')
        s.append(f'<rect class="f2-hit" fill="none" pointer-events="all" x="{bx0_l - 3:.1f}" y="{e_y0:.1f}" width="{bx1_r - bx0_l + 6:.1f}" height="{e_y1 - e_y0 + 115:.1f}"/>')
        s.append(f'<rect class="f2-bar" x="{bx0_l:.1f}" y="{by0_l:.1f}" width="{bx1_l - bx0_l:.1f}" height="{by1_l - by0_l:.1f}" '
                 f'fill="{C_SPEECH_I}" fill-opacity="0.35" stroke="{C_SPEECH_I}" stroke-width="{SW(0.8)}"/>')
        s.append(f'<rect class="f2-bar" x="{bx0_r:.1f}" y="{by0_r:.1f}" width="{bx1_r - bx0_r:.1f}" height="{by1_l - by0_r:.1f}" '
                 f'fill="{OPEN_C[2]}" fill-opacity="0.35" stroke="{OPEN_C[2]}" stroke-width="{SW(0.8)}"/>')
        if o_sd is not None:
            _, ey_hi = d2svg(ax_h, xe - w / 2, o_val + o_sd)
            _, ey_lo = d2svg(ax_h, xe - w / 2, max(0.0, o_val - o_sd))
            s.append(f'<line x1="{cx_l:.1f}" y1="{ey_hi:.1f}" x2="{cx_l:.1f}" y2="{ey_lo:.1f}" stroke="#333333" stroke-width="{SW(0.9)}"/>')
            s.append(f'<line x1="{cx_l - 2.3:.1f}" y1="{ey_hi:.1f}" x2="{cx_l + 2.3:.1f}" y2="{ey_hi:.1f}" stroke="#333333" stroke-width="{SW(0.9)}"/>')
            s.append(f'<line x1="{cx_l - 2.3:.1f}" y1="{ey_lo:.1f}" x2="{cx_l + 2.3:.1f}" y2="{ey_lo:.1f}" stroke="#333333" stroke-width="{SW(0.9)}"/>')
        if p_sd is not None:
            _, ey_hi = d2svg(ax_h, xe + w / 2, p_val + p_sd)
            _, ey_lo = d2svg(ax_h, xe + w / 2, max(0.0, p_val - p_sd))
            s.append(f'<line x1="{cx_r:.1f}" y1="{ey_hi:.1f}" x2="{cx_r:.1f}" y2="{ey_lo:.1f}" stroke="#333333" stroke-width="{SW(0.9)}"/>')
            s.append(f'<line x1="{cx_r - 2.3:.1f}" y1="{ey_hi:.1f}" x2="{cx_r + 2.3:.1f}" y2="{ey_hi:.1f}" stroke="#333333" stroke-width="{SW(0.9)}"/>')
            s.append(f'<line x1="{cx_r - 2.3:.1f}" y1="{ey_lo:.1f}" x2="{cx_r + 2.3:.1f}" y2="{ey_lo:.1f}" stroke="#333333" stroke-width="{SW(0.9)}"/>')
        tx_o, ty_o = d2svg(ax_h, xe - w / 2 - 0.08, top_o + 1.1)
        tx_p, ty_p = d2svg(ax_h, xe + w / 2 + 0.08, top_p + 1.1)
        s.append(multiline_svg(tx_o, ty_o, o_txt, 5.0, "#1f2937", anchor="middle", va="bottom", cls="f2-val"))
        s.append(multiline_svg(tx_p, ty_p, p_txt, 5.0, "#1f2937", anchor="middle", va="bottom", cls="f2-val"))
        if p.get("note") is not None:
            _, ny = d2svg(ax_h, xe, max(top_o, top_p) + 6.2)
            s.append(multiline_svg(cx_slot, ny, p["note"], 4.8, "#555555", anchor="middle", va="bottom"))
        s.append(f'<line x1="{cx_slot:.1f}" y1="{e_y1:.1f}" x2="{cx_slot:.1f}" y2="{e_y1 + 3.5:.1f}" stroke="#1f2937" stroke-width="{SW(0.8)}"/>')
        s.append(rot_xtick_svg(cx_slot, e_y1 + 6.0, p["label"], 6.0))
        s.append("</g>")

    # ---- Panel (f): Reproduction full-pipeline WER ----
    ax_i = axes_dict["f"]
    ch_f, f_x0, f_x1, f_y0, f_y1 = draw_axes_chrome(
        ax_i,
        list(range(0, 126, 25)),
        [str(v) for v in range(0, 126, 25)],
        "Word error rate (%)",
        "(f) Reproduction: full-pipeline WER, one pair per panel-(c) bar",
        title_fs=9.3,
    )
    s.append(ch_f)
    draw_inpanel_legend(f_x0, f_y0, [
        ("patch", C_SPEECH_I, "our reproduction (16 GB home PC)"),
        ("patch", OPEN_C[2], "paper / benchmark anchor"),
    ], fs_pt=6.6)
    s.append(multiline_svg(
        f_x0 + 0.98 * (f_x1 - f_x0), f_y0 + 0.03 * (f_y1 - f_y0),
        "error bar is calculated across days", 6.0, "#555555", anchor="end", va="top"
    ))
    for i, v in enumerate(wer):
        sid = PANEL_BCD_IDS[i]
        bx0_l, _ = d2svg(ax_i, i - w, 0)
        bx1_r, by1_f = d2svg(ax_i, i + w, 0)
        cx_slot = 0.5 * (bx0_l + bx1_r)
        s.append(f'<g class="f2-slot" data-id="{sid}" data-panel="f">')
        if v["ours"] is None:
            s.append(f'<title>{esc(v["label"])} \u2014 Direct voice synthesis (see acoustic stage in Panel e)</title>')
            s.append(f'<rect class="f2-hit" fill="none" pointer-events="all" x="{bx0_l - 3:.1f}" y="{f_y0:.1f}" width="{bx1_r - bx0_l + 6:.1f}" height="{f_y1 - f_y0 + 115:.1f}"/>')
            _, nry = d2svg(ax_i, i, 8.0)
            s.append(multiline_svg(cx_slot, nry, v["note"], 5.2, "#777777", anchor="middle", va="bottom", cls="f2-val"))
        else:
            o_val, o_sd = v["ours"], v.get("ours_sd")
            p_val, p_sd = v["paper"], v.get("paper_sd")
            top_o = o_val + o_sd if o_sd is not None else o_val
            top_p = p_val + p_sd if p_sd is not None else p_val
            o_txt = f"{o_val:.1f}\u00b1{o_sd:.1f}" if o_sd is not None else f"{o_val:.1f}"
            p_txt = f"{p_val:.1f}\u00b1{p_sd:.1f}" if p_sd is not None else f"{p_val:.1f}"
            bx0_l, by0_l = d2svg(ax_i, i - w, o_val)
            bx1_l, _ = d2svg(ax_i, i, 0)
            bx0_r, by0_r = d2svg(ax_i, i, p_val)
            cx_l = 0.5 * (bx0_l + bx1_l)
            cx_r = 0.5 * (bx0_r + bx1_r)
            s.append(f'<title>{esc(v["label"])} \u2014 Full-pipeline WER: Ours {esc(o_txt)}% vs. Paper {esc(p_txt)}% (click to inspect study)</title>')
            s.append(f'<rect class="f2-hit" fill="none" pointer-events="all" x="{bx0_l - 3:.1f}" y="{f_y0:.1f}" width="{bx1_r - bx0_l + 6:.1f}" height="{f_y1 - f_y0 + 115:.1f}"/>')
            s.append(f'<rect class="f2-bar" x="{bx0_l:.1f}" y="{by0_l:.1f}" width="{bx1_l - bx0_l:.1f}" height="{by1_f - by0_l:.1f}" '
                     f'fill="{C_SPEECH_I}" fill-opacity="0.35" stroke="{C_SPEECH_I}" stroke-width="{SW(0.8)}"/>')
            s.append(f'<rect class="f2-bar" x="{bx0_r:.1f}" y="{by0_r:.1f}" width="{bx1_r - bx0_r:.1f}" height="{by1_f - by0_r:.1f}" '
                     f'fill="{OPEN_C[2]}" fill-opacity="0.35" stroke="{OPEN_C[2]}" stroke-width="{SW(0.8)}"/>')
            if o_sd is not None:
                _, ey_hi = d2svg(ax_i, i - w / 2, o_val + o_sd)
                _, ey_lo = d2svg(ax_i, i - w / 2, max(0.0, o_val - o_sd))
                s.append(f'<line x1="{cx_l:.1f}" y1="{ey_hi:.1f}" x2="{cx_l:.1f}" y2="{ey_lo:.1f}" stroke="#333333" stroke-width="{SW(0.9)}"/>')
                s.append(f'<line x1="{cx_l - 2.3:.1f}" y1="{ey_hi:.1f}" x2="{cx_l + 2.3:.1f}" y2="{ey_hi:.1f}" stroke="#333333" stroke-width="{SW(0.9)}"/>')
                s.append(f'<line x1="{cx_l - 2.3:.1f}" y1="{ey_lo:.1f}" x2="{cx_l + 2.3:.1f}" y2="{ey_lo:.1f}" stroke="#333333" stroke-width="{SW(0.9)}"/>')
            if p_sd is not None:
                _, ey_hi = d2svg(ax_i, i + w / 2, p_val + p_sd)
                _, ey_lo = d2svg(ax_i, i + w / 2, max(0.0, p_val - p_sd))
                s.append(f'<line x1="{cx_r:.1f}" y1="{ey_hi:.1f}" x2="{cx_r:.1f}" y2="{ey_lo:.1f}" stroke="#333333" stroke-width="{SW(0.9)}"/>')
                s.append(f'<line x1="{cx_r - 2.3:.1f}" y1="{ey_hi:.1f}" x2="{cx_r + 2.3:.1f}" y2="{ey_hi:.1f}" stroke="#333333" stroke-width="{SW(0.9)}"/>')
                s.append(f'<line x1="{cx_r - 2.3:.1f}" y1="{ey_lo:.1f}" x2="{cx_r + 2.3:.1f}" y2="{ey_lo:.1f}" stroke="#333333" stroke-width="{SW(0.9)}"/>')
            tx_o, ty_o = d2svg(ax_i, i - w / 2 - 0.02, top_o + 1.4)
            tx_p, ty_p = d2svg(ax_i, i + w / 2 + 0.02, top_p + 1.4)
            s.append(multiline_svg(tx_o, ty_o, o_txt, 5.0, "#1f2937", anchor="middle", va="bottom", cls="f2-val"))
            s.append(multiline_svg(tx_p, ty_p, p_txt, 5.0, "#1f2937", anchor="middle", va="bottom", cls="f2-val"))
            if v.get("note") is not None:
                _, ny = d2svg(ax_i, i, max(top_o, top_p) + 7.8)
                s.append(multiline_svg(cx_slot, ny, v["note"], 4.8, "#555555", anchor="middle", va="bottom"))
        s.append(f'<line x1="{cx_slot:.1f}" y1="{f_y1:.1f}" x2="{cx_slot:.1f}" y2="{f_y1 + 3.5:.1f}" stroke="#1f2937" stroke-width="{SW(0.8)}"/>')
        s.append(rot_xtick_svg(cx_slot, f_y1 + 6.0, v["label"], 6.0))
        s.append("</g>")

    svg_inner = "\n".join(s)
    # 2026-11-11 RAM/VRAM split: the bottom row lost its rotated x-tick
    # labels, so the former hardcoded vh = 2245 tight crop is measured
    # from the drawn SVG elements instead (max y over text/rect/line).
    ys = [float(m) for m in re.findall(r'\sy="(-?[0-9.]+)"', svg_inner)]
    vh = int(max(ys)) + 14 if ys else int(h_px * 0.5)
    svg_inner = (
        f'<rect x="0" y="0" width="{vw}" height="{vh}" fill="#ffffff"/>\n'
        + svg_inner
    )
    return svg_inner, vw, vh


# Robinson projection lookup table (every 5 degrees of latitude, 0..90 deg)
_ROB_LAT = np.arange(0.0, 95.0, 5.0)
_ROB_X = np.array([
    1.0000, 0.9986, 0.9954, 0.9900, 0.9822, 0.9730, 0.9600, 0.9427,
    0.9216, 0.8962, 0.8679, 0.8350, 0.7986, 0.7597, 0.7186, 0.6732,
    0.6213, 0.5722, 0.5322,
])
_ROB_Y = np.array([
    0.0000, 0.0620, 0.1240, 0.1860, 0.2480, 0.3100, 0.3720, 0.4340,
    0.4958, 0.5571, 0.6176, 0.6769, 0.7346, 0.7903, 0.8435, 0.8936,
    0.9394, 0.9761, 1.0000,
])

# Sovereign states whose best official language is Tier 2 in (c):
# high-/mid-resource tonal, agglutinative, unsegmented-script, or Semitic
# official language (with sufficient text for L/G/LLM and existing TTS, so Tier 1 in (d)),
# and NO Tier-1 Indo-European analytic/fusional co-official language.
_CTC_TIER2_A3 = {
    "CHN", "TWN", "VNM", "THA",
    "JPN", "KOR", "PRK",
    "TUR", "AZE", "UZB", "CYN",
    "HUN", "EST", "GEO",
    "IDN", "MYS", "BRN",
    "SAU", "EGY", "MAR", "DZA", "TUN", "LBY", "JOR", "IRQ", "SYR",
    "LBN", "PSX", "ISR", "ARE", "QAT", "KWT", "OMN", "YEM", "SAH",
    "MRT", "SOM", "SOL",
}

# Sovereign states / territories whose best official language is itself low-resource
# (insufficient text corpus to train open-vocabulary WFST + LLM rescorer -> Tier 3 in (c);
#  lacks mainstream TTS so requires recorded audio -> Tier 2 in (d)),
# and having NO high-resource co-official language (English/French/Spanish/Portuguese/Russian/Arabic).
_CTC_TIER3_A3 = {
    "GRL",                      # Kalaallisut (sole official since 2009)
    "BTN",                      # Dzongkha (sole official)
    "LAO", "KHM", "MMR",        # Lao, Khmer, Burmese (sole official, low-resource)
    "MNG", "TKM",               # Mongolian, Turkmen (sole official, low-resource)
    "ETH", "ERI",               # Amharic/Oromo/Tigrinya/Afar (low-resource)
    "MLI", "BFA", "NER",        # Indigenous languages sole official after 2023-2025 constitutions
}

_B2V_TIER2_A3 = _CTC_TIER3_A3

# 2026-11-12 official-language tooltip round (user: "html中Limits of CTC-TLG节，
# 鼠标指针放在某个国家上时，列出其官方语言。维基百科或者联合国应该有这个数据"):
# per-country official-language strings for the Figure 6 country hover tooltip,
# keyed by the Natural Earth adm0_a3 codes of _WORLD_110M_B64. Reviewed against
# Wikidata P37 (official language) + Wikipedia "List of official languages"
# (snapshot 2026-11), cross-checked with mledoze/countries (GitHub, 2026-11);
# NE-only codes (PSX/SOL/CYN/SDS/KOS/SAH) curated by hand. The tooltip shows at
# most 8 entries and appends "+N more".
_OFFICIAL_LANGUAGES = {
    "AFG": ("Pashto", "Dari (Persian)"),
    "AGO": ("Portuguese",),
    "ALB": ("Albanian",),
    "ARE": ("Arabic",),
    "ARG": ("Spanish",),
    "ARM": ("Armenian",),
    "AUS": ("English (de facto)",),
    "AUT": ("German",),
    "AZE": ("Azerbaijani",),
    "BDI": ("Kirundi",),
    "BEL": ("Dutch", "French", "German"),
    "BEN": ("French",),
    "BFA": ("Mooré", "Bissa", "Dioula", "Fulfulde (2024 const.)", "French (working language)"),
    "BGD": ("Bengali",),
    "BGR": ("Bulgarian",),
    "BHS": ("English",),
    "BIH": ("Bosnian", "Croatian", "Serbian"),
    "BLR": ("Belarusian", "Russian"),
    "BLZ": ("English",),
    "BOL": ("Spanish", "Aymara", "Guaraní", "Quechua", "+33 further co-official indigenous languages (2009 const.)"),
    "BRA": ("Portuguese",),
    "BRN": ("Malay",),
    "BTN": ("Dzongkha",),
    "BWA": ("English",),
    "CAF": ("Sango", "French"),
    "CAN": ("English", "French"),
    "CHE": ("German", "French", "Italian", "Romansh"),
    "CHL": ("Spanish",),
    "CHN": ("Standard Chinese (Putonghua)",),
    "CIV": ("French",),
    "CMR": ("French", "English"),
    "COD": ("French", "Lingala", "Kikongo", "Swahili", "Tshiluba (national languages)"),
    "COG": ("French",),
    "COL": ("Spanish",),
    "CRI": ("Spanish",),
    "CUB": ("Spanish",),
    "CYN": ("Turkish",),
    "CYP": ("Greek", "Turkish"),
    "CZE": ("Czech",),
    "DEU": ("German",),
    "DJI": ("Arabic", "French"),
    "DNK": ("Danish",),
    "DOM": ("Spanish",),
    "DZA": ("Arabic", "Tamazight (Berber)"),
    "ECU": ("Spanish", "Kichwa", "Shuar"),
    "EGY": ("Arabic",),
    "ERI": ("Tigrinya", "Arabic", "English (working languages)"),
    "ESP": ("Spanish (Castilian)", "Catalan", "Galician", "Basque (co-official in their regions)"),
    "EST": ("Estonian",),
    "ETH": ("Amharic (federal working language)", "Oromo", "Afar", "Somali", "Tigrinya (regional official)"),
    "FIN": ("Finnish", "Swedish"),
    "FJI": ("iTaukei (Fijian)", "Fiji Hindi", "English"),
    "FLK": ("English",),
    "FRA": ("French",),
    "GAB": ("French",),
    "GBR": ("English (de facto)", "Welsh (Wales)", "Scots Gaelic (Scotland)"),
    "GEO": ("Georgian",),
    "GHA": ("English",),
    "GIN": ("French",),
    "GMB": ("English",),
    "GNB": ("Portuguese",),
    "GNQ": ("Spanish", "French", "Portuguese"),
    "GRC": ("Greek",),
    "GRL": ("Kalaallisut (sole official since 2009)",),
    "GTM": ("Spanish",),
    "GUY": ("English",),
    "HND": ("Spanish",),
    "HRV": ("Croatian",),
    "HTI": ("Haitian Creole", "French"),
    "HUN": ("Hungarian",),
    "IDN": ("Indonesian",),
    "IND": ("Hindi", "English (union official)", "+22 scheduled languages"),
    "IRL": ("Irish (first official)", "English"),
    "IRN": ("Persian (Farsi)",),
    "IRQ": ("Arabic", "Kurdish"),
    "ISL": ("Icelandic",),
    "ISR": ("Hebrew", "Arabic (special status)"),
    "ITA": ("Italian",),
    "JAM": ("English (Jamaican Patois national)",),
    "JOR": ("Arabic",),
    "JPN": ("Japanese (de facto)",),
    "KAZ": ("Kazakh", "Russian"),
    "KEN": ("Swahili", "English"),
    "KGZ": ("Kyrgyz", "Russian"),
    "KHM": ("Khmer",),
    "KOR": ("Korean",),
    "KOS": ("Albanian", "Serbian"),
    "KWT": ("Arabic",),
    "LAO": ("Lao",),
    "LBN": ("Arabic",),
    "LBR": ("English",),
    "LBY": ("Arabic",),
    "LKA": ("Sinhala", "Tamil", "English (link language)"),
    "LSO": ("Sesotho", "English"),
    "LTU": ("Lithuanian",),
    "LUX": ("Luxembourgish", "French", "German"),
    "LVA": ("Latvian",),
    "MAR": ("Arabic", "Tamazight (Berber)"),
    "MDA": ("Romanian", "Russian & Gagauz (regional official)"),
    "MDG": ("Malagasy", "French"),
    "MEX": ("Spanish (de facto)", "+68 indigenous national languages (e.g. Nahuatl, Maya, Mixtec)"),
    "MKD": ("Macedonian", "Albanian (co-official)"),
    "MLI": ("Bambara", "Fulfulde", "Soninke", "+10 further national languages (2023 const.)", "French (working language)"),
    "MMR": ("Burmese",),
    "MNE": ("Montenegrin", "Serbian", "Bosnian", "Albanian", "Croatian"),
    "MNG": ("Mongolian",),
    "MOZ": ("Portuguese",),
    "MRT": ("Arabic", "French (working language)"),
    "MWI": ("English", "Chichewa"),
    "MYS": ("Malay",),
    "NAM": ("English (official since 1990)", "Afrikaans & German (recognized regional)", "Oshiwambo & other national languages"),
    "NCL": ("French",),
    "NER": ("National languages: Hausa, Zarma, Fulfulde, Kanuri, Tamachek, Arabic (2025 const.)", "French (working language)"),
    "NGA": ("English",),
    "NIC": ("Spanish",),
    "NLD": ("Dutch", "Frisian (Friesland)"),
    "NOR": ("Norwegian (Bokmål & Nynorsk)", "Sámi (co-official)"),
    "NPL": ("Nepali", "Maithili, Bhojpuri, Tharu et al. (provincial official)"),
    "NZL": ("English (de facto)", "Māori", "New Zealand Sign Language"),
    "OMN": ("Arabic",),
    "PAK": ("Urdu (national)", "English (official)"),
    "PAN": ("Spanish",),
    "PER": ("Spanish", "Quechua", "Aymara"),
    "PHL": ("Filipino", "English"),
    "PNG": ("Tok Pisin", "Hiri Motu", "English", "Papua New Guinean Sign Language"),
    "POL": ("Polish",),
    "PRK": ("Korean",),
    "PRI": ("Spanish", "English"),
    "PRT": ("Portuguese",),
    "PRY": ("Spanish", "Guaraní"),
    "PSX": ("Arabic",),
    "QAT": ("Arabic",),
    "ROU": ("Romanian",),
    "RUS": ("Russian",),
    "RWA": ("Kinyarwanda", "French", "English", "Swahili"),
    "SAH": ("Arabic (Hassaniya spoken)",),
    "SAU": ("Arabic",),
    "SDN": ("Arabic", "English"),
    "SDS": ("English", "indigenous languages (official at state level)"),
    "SEN": ("French", "Wolof (national)"),
    "SLE": ("English",),
    "SLB": ("English",),
    "SLV": ("Spanish",),
    "SOL": ("Somali", "Arabic"),
    "SOM": ("Somali", "Arabic"),
    "SRB": ("Serbian",),
    "SUR": ("Dutch",),
    "SVK": ("Slovak",),
    "SVN": ("Slovene", "Italian & Hungarian (co-official locally)"),
    "SWE": ("Swedish",),
    "SWZ": ("siSwati", "English"),
    "SYR": ("Arabic",),
    "TCD": ("Arabic", "French"),
    "TGO": ("French", "Ewe & Kabye (national)"),
    "THA": ("Thai",),
    "TJK": ("Tajik", "Russian (inter-ethnic communication)"),
    "TKM": ("Turkmen",),
    "TLS": ("Tetum", "Portuguese"),
    "TTO": ("English",),
    "TUN": ("Arabic",),
    "TUR": ("Turkish",),
    "TWN": ("Mandarin Chinese (Guóyǔ)",),
    "TZA": ("Swahili", "English"),
    "UGA": ("English", "Swahili"),
    "UKR": ("Ukrainian",),
    "URY": ("Spanish",),
    "USA": ("English (de facto)",),
    "UZB": ("Uzbek", "Karakalpak (in Karakalpakstan)"),
    "VEN": ("Spanish",),
    "VNM": ("Vietnamese",),
    "VUT": ("Bislama", "English", "French"),
    "YEM": ("Arabic",),
    "ZAF": ("Afrikaans", "English", "Ndebele", "Northern Sotho", "Sotho", "Swazi", "Tsonga", "Tswana", "Venda", "Xhosa", "Zulu", "South African Sign Language (12th, 2023)"),
    "ZMB": ("English",),
    "ZWE": ("Chewa", "Chibarwe", "English", "Kalanga", "Khoisan", "Nambya", "Ndau", "Northern Ndebele", "Shangani", "Shona", "Zimbabwean Sign Language", "Sotho", "Tonga", "Tswana", "Venda", "Xhosa (16, 2013 const.)"),
}



def _robinson_xy(lon_deg, lat_deg):
    """Project (lon, lat) in degrees to normalized Robinson (x, y)."""
    lon = np.asarray(lon_deg, dtype=float)
    lat = np.asarray(lat_deg, dtype=float)
    abs_lat = np.clip(np.abs(lat), 0.0, 90.0)
    sign_lat = np.sign(lat)
    rx = np.interp(abs_lat, _ROB_LAT, _ROB_X)
    ry = np.interp(abs_lat, _ROB_LAT, _ROB_Y)
    x = 0.8487 * rx * np.deg2rad(lon)
    y = 1.3523 * sign_lat * ry
    return x, y


def _ctc_tier(adm0_a3):
    if adm0_a3 in _CTC_TIER3_A3:
        return 3
    if adm0_a3 in _CTC_TIER2_A3:
        return 2
    return 1


def _b2v_tier(adm0_a3):
    if adm0_a3 in _B2V_TIER2_A3:
        return 2
    return 1


def _build_fig6_svg(verify_overlaps=True):
    """Render Figure 6 (The Limit of CTC-TLG vs. Brain-to-Voice across ~7,000
    spoken languages, Panels a-d) as interactive vector SVG.

    Parameters
    ----------
    verify_overlaps : bool, optional
        When True, run ``check_figure_overlaps`` on the Matplotlib layout pass.

    Returns
    -------
    tuple of (str, int, int)
        ``(svg_inner, vw, vh)`` for ``<svg id="f6-svg">``.
    """
    C_T1_MAP = "#78A65A"
    C_T1_BG  = "#EFF6EA"
    C_T1_EC  = "#3E6B24"

    C_T2_MAP = "#DFC266"
    C_T2_BG  = "#FBF7E8"
    C_T2_EC  = "#7D6618"

    C_T3_MAP = "#B06C49"
    C_T3_BG  = "#F8EFEA"
    C_T3_EC  = "#6E3B23"

    C_BOX_BG = "#F7F8FA"
    C_BOX_EC = "#4A5568"
    C_INK    = "#1D242B"
    C_MUTED  = "#5A6577"
    C_OCEAN  = "#F5F8FA"
    C_GRAT   = "#D6DEE6"

    fig = plt.figure(figsize=(22.4, 11.4), dpi=200)
    outer = fig.add_gridspec(
        2, 12, height_ratios=[0.66, 1.34],
        hspace=0.28, wspace=0.42
    )
    fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.06)

    ax_a = fig.add_subplot(outer[0, 0:6])
    ax_a.set_xlim(0, 100)
    ax_a.set_ylim(6.5, 31.0)
    ax_a.axis("off")
    ax_a.set_title("(a) Brain-to-text decoding pipeline",
                   fontsize=10.0, fontweight="bold", pad=10)

    ax_b = fig.add_subplot(outer[0, 6:12])
    ax_b.set_xlim(0, 100)
    ax_b.set_ylim(6.5, 31.0)
    ax_b.axis("off")
    ax_b.set_title("(b) Brain-to-voice decoding pipeline",
                   fontsize=10.0, fontweight="bold", pad=10)

    ax_cd = fig.add_subplot(outer[1, 0:12])
    ax_cd.set_xlim(0, 184)
    ax_cd.set_ylim(4, 72)
    ax_cd.axis("off")

    ax_c_title = fig.add_subplot(outer[1, 0:6])
    ax_c_title.axis("off")
    ax_c_title.set_title("(c) Brain-to-text applicability (~7,000 spoken languages)",
                         fontsize=10.0, fontweight="bold", pad=10)

    ax_d_title = fig.add_subplot(outer[1, 6:12])
    ax_d_title.axis("off")
    ax_d_title.set_title("(d) Brain-to-voice applicability (~7,000 spoken languages)",
                         fontsize=10.0, fontweight="bold", pad=10)

    fig.canvas.draw()
    if verify_overlaps:
        bad6 = check_figure_overlaps(fig)
        if bad6:
            plt.close(fig)
            raise RuntimeError(f"Figure 6 overlap check failed: {bad6}")

    renderer = fig.canvas.get_renderer()
    w_px, h_px = fig.canvas.get_width_height()
    vw, vh = int(round(w_px * 0.5)), int(round(h_px * 0.5))
    PT = 1.3889
    SW = lambda pt: round(pt * PT, 2)
    SX = lambda px: px * 0.5
    SY = lambda py: (h_px - py) * 0.5
    esc = html_lib.escape

    def d2svg(ax, dx, dy):
        px, py = ax.transData.transform((dx, dy))
        return SX(px), SY(py)

    s = [
        f'<defs>'
        f'<pattern id="f6-hatch-t3" width="6.5" height="6.5" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">'
        f'<line x1="0" y1="0" x2="0" y2="6.5" stroke="{C_T3_EC}" stroke-width="1.35"/>'
        f'</pattern>'
        f'<pattern id="f6-hatch-t2" width="6.5" height="6.5" patternTransform="rotate(45 0 0)" patternUnits="userSpaceOnUse">'
        f'<line x1="0" y1="0" x2="0" y2="6.5" stroke="{C_T2_EC}" stroke-width="1.35"/>'
        f'</pattern>'
        f'<marker id="f6-arr" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="6.5" markerHeight="6.5" orient="auto-start-reverse">'
        f'<path d="M 0 1.2 L 9 5 L 0 8.8 z" fill="{C_BOX_EC}"/>'
        f'</marker>'
        f'</defs>',
        f'<rect x="0" y="0" width="{vw}" height="{vh}" fill="#ffffff"/>',
    ]

    # Panel titles
    for ax_t in (ax_a, ax_b, ax_c_title, ax_d_title):
        bb = ax_t.get_window_extent(renderer)
        tcx = SX(0.5 * (bb.x0 + bb.x1))
        ty = SY(bb.y1) - 10.0
        s.append(f'<text x="{tcx:.1f}" y="{ty:.1f}" font-size="{10.0 * PT:.2f}" '
                 f'font-weight="bold" fill="#111827" text-anchor="middle">{esc(ax_t.get_title())}</text>')

    pad = 0.30

    def svg_block(ax, x, y, w, h, title, sub="", fc=C_BOX_BG, ec=C_BOX_EC,
                  lw=1.25, title_fs=8.5, sub_fs=7.0, tier="", study_id="", tip="", f6_map=""):
        x0, y0 = d2svg(ax, x - pad, y + h + pad)
        x1, y1 = d2svg(ax, x + w + pad, y - pad)
        bw, bh = x1 - x0, y1 - y0
        cx = 0.5 * (x0 + x1)
        attrs = ['class="f6-node"']
        if tier:
            attrs.append(f'data-f6-tier="{tier}"')
        if f6_map:
            attrs.append(f'data-f6-map="{f6_map}"')
        if study_id:
            attrs.append(f'data-id="{study_id}"')
        g = [f'<g {" ".join(attrs)}>']
        if tip:
            g.append(f'<title>{esc(tip)}</title>')
        g.append(f'<rect class="f6-box" x="{x0:.1f}" y="{y0:.1f}" width="{bw:.1f}" height="{bh:.1f}" rx="8.5" '
                 f'fill="{fc}" stroke="{ec}" stroke-width="{SW(lw)}"/>')
        if sub:
            _, ty1 = d2svg(ax, x + w / 2, y + h * 0.62)
            _, ty2 = d2svg(ax, x + w / 2, y + h * 0.28)
            g.append(f'<text x="{cx:.1f}" y="{ty1:.1f}" font-size="{title_fs * PT:.2f}" '
                     f'font-weight="bold" fill="{C_INK}" text-anchor="middle" dominant-baseline="central">{esc(title)}</text>')
            g.append(f'<text x="{cx:.1f}" y="{ty2:.1f}" font-size="{sub_fs * PT:.2f}" '
                     f'fill="{C_MUTED}" text-anchor="middle" dominant-baseline="central">{esc(sub)}</text>')
        else:
            _, tyc = d2svg(ax, x + w / 2, y + h / 2)
            g.append(f'<text x="{cx:.1f}" y="{tyc:.1f}" font-size="{title_fs * PT:.2f}" '
                     f'font-weight="bold" fill="{C_INK}" text-anchor="middle" dominant-baseline="central">{esc(title)}</text>')
        g.append("</g>")
        s.extend(g)

    def svg_arr(ax, p1, p2):
        x1, y1 = d2svg(ax, p1[0], p1[1])
        x2, y2 = d2svg(ax, p2[0], p2[1])
        dx, dy = x2 - x1, y2 - y1
        ln = math.hypot(dx, dy) or 1.0
        ux, uy = dx / ln, dy / ln
        sx2, sy2 = x2 - 6.0 * ux, y2 - 6.0 * uy
        hl, hw = 7.8, 3.6
        p_a = f"{x2:.1f},{y2:.1f}"
        p_b = f"{x2 - hl * ux - hw * uy:.1f},{y2 - hl * uy + hw * ux:.1f}"
        p_c = f"{x2 - hl * ux + hw * uy:.1f},{y2 - hl * uy - hw * ux:.1f}"
        s.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{sx2:.1f}" y2="{sy2:.1f}" '
                 f'stroke="{C_BOX_EC}" stroke-width="{SW(1.35)}"/>')
        s.append(f'<polygon points="{p_a} {p_b} {p_c}" fill="{C_BOX_EC}"/>')

    def svg_poly(ax, pts, color=C_BOX_EC, lw=1.35, dash=None):
        p_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in (d2svg(ax, px_, py_) for px_, py_ in pts))
        dash_a = f' stroke-dasharray="{dash}"' if dash else ""
        s.append(f'<polyline points="{p_str}" fill="none" stroke="{color}" '
                 f'stroke-width="{SW(lw)}"{dash_a}/>')

    def hatch_poly_lines(px, py, step=6.2):
        u = px + py
        v = px - py
        n_v = len(px)
        u_min, u_max = float(np.min(u)), float(np.max(u))
        u0 = math.ceil(u_min / step) * step
        segs = []
        while u0 <= u_max:
            vhits = []
            for idx_e in range(n_v):
                u1, v1 = u[idx_e], v[idx_e]
                u2, v2 = u[(idx_e + 1) % n_v], v[(idx_e + 1) % n_v]
                if (u1 <= u0 < u2) or (u2 <= u0 < u1):
                    t = (u0 - u1) / (u2 - u1)
                    vhits.append(v1 + t * (v2 - v1))
            vhits.sort()
            for k_h in range(0, len(vhits) - 1, 2):
                va, vb = vhits[k_h], vhits[k_h + 1]
                segs.append((0.5 * (u0 + va), 0.5 * (u0 - va), 0.5 * (u0 + vb), 0.5 * (u0 - vb)))
            u0 += step
        return segs

    # ---- Panel (a): Brain-to-text decoding pipeline (CTC-TLG) ----
    yb, hb = 11.5, 13.5
    yc = yb + hb / 2  # 18.25
    y_v1, y_v2 = 23.4, 13.1
    svg_block(ax_a, 0.4, 19.6, 19.8, 7.6, "Threshold Crossings", "20-ms bins",
              title_fs=8.0, sub_fs=6.7,
              tip="Intracortical neural features: 20-ms binned threshold crossings (TX), blockwise z-scored.")
    svg_block(ax_a, 0.4, 9.3, 19.8, 7.6, "Spike Band Power", "20-ms bins",
              title_fs=8.0, sub_fs=6.7,
              tip="Intracortical neural features: 20-ms binned spike-band power (SBP), blockwise z-scored.")
    # Merge both input streams into the per-day calibration layer
    x_in_r = 0.4 + 19.8 + pad
    svg_poly(ax_a, [(x_in_r, y_v1), (21.2, y_v1), (21.2, y_v2), (x_in_r, y_v2)])
    svg_arr(ax_a, (21.2, yc), (22.8 - pad, yc))

    svg_block(ax_a, 22.8, yb, 17.4, hb, "Daily Calibration", "Linear alignment",
              tip="Per-day calibration layer: linear (affine) alignment W_d\u00b7x_t + b_d absorbing cross-day recording nonstationarity.")
    svg_arr(ax_a, (40.2 + pad, yc), (42.8 - pad, yc))

    # Colored by worst-case tier: CTC RNN -> Tier 2 (amber); WFST & LLM -> Tier 3 (terracotta)
    svg_block(ax_a, 42.8, yb, 13.8, hb, "CTC RNN", "Phoneme probs (T)",
              fc=C_T2_BG, ec=C_T2_EC, tier="2", f6_map="c",
              tip="Tier 2 bottleneck: emits frame posteriors over a language-specific discrete phoneme inventory T (e.g., 39 English ARPAbet phonemes + SIL + blank). Tonal or sub-word languages require tonal T or sub-word token redesign.")
    svg_arr(ax_a, (56.6 + pad, yc), (59.2 - pad, yc))

    svg_block(ax_a, 59.2, yb, 16.6, hb, "WFST (T\u2218L\u2218G)", "Lexicon L + n-gram G",
              fc=C_T3_BG, ec=C_T3_EC, tier="3", f6_map="c",
              tip="Tier 3 & memory bottleneck: composes token FST T, pronunciation lexicon L, and n-gram grammar G into TLG.fst (~300 GB unpruned RAM; 19.6 GB mmap). Cannot apply to languages that are unwritten or lack sufficient text to train language models; fails when greedy PER > 25%.")
    svg_arr(ax_a, (75.8 + pad, yc), (78.4 - pad, yc))

    svg_block(ax_a, 78.4, yb, 11.6, hb, "LLM", "Text rescorer",
              fc=C_T3_BG, ec=C_T3_EC, tier="3", f6_map="c",
              tip="Tier 3 bottleneck: second-pass causal LLM rescorer (OPT-6.7B / Qwen3-8B, 13 GB VRAM). Requires massive written pretraining corpora — unavailable for unwritten languages or those with insufficient text.")
    svg_arr(ax_a, (90.0 + pad, yc), (92.8 - pad, yc))
    tx_txt, ty_txt = d2svg(ax_a, 95.6, yc)
    s.append(f'<text x="{tx_txt:.1f}" y="{ty_txt:.1f}" font-size="{9.0 * PT:.2f}" '
             f'font-weight="bold" fill="{C_INK}" text-anchor="middle" dominant-baseline="central">Text</text>')

    # ---- Panel (b): Brain-to-voice decoding pipeline (v1.0 & v2.0) ----
    svg_block(ax_b, 0.4, 19.6, 19.8, 7.6, "Threshold Crossings", "10-ms bins",
              title_fs=8.0, sub_fs=6.7, study_id="wairagkar2025",
              tip="10-ms binned threshold crossings + spike-band power from ventral precentral gyrus (vPCG, 256 channels, T15).")
    svg_block(ax_b, 0.4, 9.3, 19.8, 7.6, "Spike Band Power", "10-ms bins",
              title_fs=8.0, sub_fs=6.7, study_id="wairagkar2025",
              tip="10-ms binned threshold crossings + spike-band power from ventral precentral gyrus (vPCG, 256 channels, T15).")
    # Combine both features and feed separately into v1.0 (top) and v2.0 (bottom)
    x_in_b = 0.4 + 19.8 + pad
    svg_poly(ax_b, [(x_in_b, y_v1), (21.1, y_v1), (21.1, y_v2), (x_in_b, y_v2)])
    svg_poly(ax_b, [(21.1, yc), (21.7, yc)])
    svg_poly(ax_b, [(21.7, y_v2), (21.7, y_v1)])
    svg_arr(ax_b, (21.7, y_v1), (22.8 - pad, y_v1))
    svg_arr(ax_b, (21.7, y_v2), (22.8 - pad, y_v2))

    # Top row: v1.0 (2025) separate pipeline
    # Acoustic Head = Tier 2 worst-case (needs recorded audio for DTW training when TTS is absent)
    # LPCNet Vocoder = Tier 1 worst-case (universal vocal-tract waveform synthesis)
    svg_block(ax_b, 22.8, 19.6, 18.8, 7.6, "v1.0 Voiceformer", "2-block causal (600 ms)",
              title_fs=8.0, sub_fs=6.7, study_id="wairagkar2025",
              tip="Wairagkar et al., 2025 (Brain-to-Voice v1.0): 2-block causal Voiceformer encoder (600-ms receptive field) mapping 10-ms neural bins to acoustic features in 1.5 GB VRAM (click to inspect study).")
    svg_arr(ax_b, (41.6 + pad, y_v1), (44.5 - pad, y_v1))
    svg_block(ax_b, 44.5, 19.6, 21.2, 7.6, "20-D Acoustic Head", "18 Bark + 2 F0 pitch",
              fc=C_T2_BG, ec=C_T2_EC, title_fs=8.0, sub_fs=6.7, tier="2", f6_map="d",
              study_id="wairagkar2025",
              tip="Tier 2 bottleneck: continuous 20-D LPCNet regression head (18 Bark bands + 2 F0). When the language lacks TTS, training needs recorded prompt audio for DTW alignment (v1.0: 43.8% listener WER; Mel Pearson r = 0.69 in our reproduction).")
    svg_arr(ax_b, (65.7 + pad, y_v1), (68.5 - pad, y_v1))
    svg_block(ax_b, 68.5, 19.6, 17.0, 7.6, "LPCNet Vocoder", "Causal synthesis",
              fc=C_T1_BG, ec=C_T1_EC, title_fs=8.0, sub_fs=6.7, tier="1", f6_map="d",
              study_id="wairagkar2025",
              tip="Causal LPCNet vocoder: synthesizes continuous speech waveform directly from 20-D Bark + F0 features in <10 ms with no dictionary L, no WFST G, and no LLM.")
    svg_arr(ax_b, (85.5 + pad, y_v1), (88.0 - pad, y_v1))
    tx_v1, ty_v1 = d2svg(ax_b, 91.0, y_v1)
    s.append(f'<text x="{tx_v1:.1f}" y="{ty_v1:.1f}" font-size="{9.0 * PT:.2f}" '
             f'font-weight="bold" fill="{C_INK}" text-anchor="middle" dominant-baseline="central">Voice</text>')

    # Dashed divider between 1.0 (top) and 2.0 (bottom) pipelines
    svg_poly(ax_b, [(22.2, yc), (99.2, yc)], color=C_MUTED, lw=0.75, dash="4.5,2.8")
    tx_10, ty_10 = d2svg(ax_b, 97.2, yc + 1.55)
    s.append(f'<text x="{tx_10:.1f}" y="{ty_10:.1f}" font-size="{6.8 * PT:.2f}" '
             f'font-weight="bold" fill="{C_MUTED}" text-anchor="middle" dominant-baseline="bottom">1.0</text>')
    tx_20, ty_20 = d2svg(ax_b, 97.2, yc - 1.55)
    s.append(f'<text x="{tx_20:.1f}" y="{ty_20:.1f}" font-size="{6.8 * PT:.2f}" '
             f'font-weight="bold" fill="{C_MUTED}" text-anchor="middle" dominant-baseline="top">2.0</text>')

    # Bottom row: v2.0 (2026) separate pipeline
    svg_block(ax_b, 22.8, 9.3, 18.8, 7.6, "v2.0 Transformer", "8-layer causal masked",
              title_fs=8.0, sub_fs=6.7, study_id="wairagkar2026",
              tip="Wairagkar et al., 2026 (Brain2voice 2.0): 8-layer causal masked Transformer encoder (click to inspect study).")
    svg_arr(ax_b, (41.6 + pad, y_v2), (44.5 - pad, y_v2))
    svg_block(ax_b, 44.5, 9.3, 21.2, 7.6, "Acoustic + GAN", "+ RVQ / CTC / SSL aux",
              fc=C_T2_BG, ec=C_T2_EC, title_fs=8.0, sub_fs=6.7, tier="2", f6_map="d",
              study_id="wairagkar2026",
              tip="Tier 2 bottleneck: 20-D continuous LPCNet + 4-scale 1D-conv GAN + auxiliary RVQ token, phoneme CTC, and SSL training heads — cutting human listener WER from 43.8% to 5.24%. Auxiliary targets still need audio or text resources that unserved languages lack.")
    svg_arr(ax_b, (65.7 + pad, y_v2), (68.5 - pad, y_v2))
    svg_block(ax_b, 68.5, 9.3, 17.0, 7.6, "LPCNet Vocoder", "Causal synthesis",
              fc=C_T1_BG, ec=C_T1_EC, title_fs=8.0, sub_fs=6.7, tier="1", f6_map="d",
              study_id="wairagkar2026",
              tip="Causal LPCNet vocoder: synthesizes continuous speech waveform directly from 20-D Bark + F0 features in <10 ms with no dictionary L, no WFST G, and no LLM.")
    svg_arr(ax_b, (85.5 + pad, y_v2), (88.0 - pad, y_v2))
    tx_v2, ty_v2 = d2svg(ax_b, 91.0, y_v2)
    s.append(f'<text x="{tx_v2:.1f}" y="{ty_v2:.1f}" font-size="{9.0 * PT:.2f}" '
             f'font-weight="bold" fill="{C_INK}" text-anchor="middle" dominant-baseline="central">Voice</text>')

    # ---- Panels (c) & (d): Robinson World Maps ----
    countries = json.loads(zlib.decompress(base64.b64decode(_WORLD_110M_B64)).decode("utf-8"))
    tier_fill = {1: C_T1_MAP, 2: C_T2_MAP, 3: C_T3_MAP}
    indigenous_belts = [
        ("Arctic & Subarctic North America (Inuktitut, Yupik, Athabaskan, Cree-Innu — polysynthetic / oral)",
         [-168, -140, -105, -75, -55, -60, -80, -115, -145, -168],
         [58, 56, 54, 53, 54, 74, 76, 74, 71, 68]),
        ("SW North America & Mesoamerican indigenous belt to Panama (Navajo, Uto-Aztecan, Otomanguean, Mayan)",
         [-112, -104, -87, -80, -83, -95, -107, -112],
         [36, 35, 21, 9, 8, 15, 24, 33]),
        ("Amazonia, Guiana Interior, Andean Cordillera & Gran Chaco (~300 languages, 50+ families — oral / unwritten)",
         [-79, -70, -60, -52, -49, -54, -64, -71, -74, -81],
         [0, 2.5, 4, 2, -2, -25, -28, -39, -16, -4]),
        ("Sub-Saharan Equatorial, East African Rift & Kalahari / Southern African belt (Bantu, Nilotic, Khoisan)",
         [-15, 10, 38, 37, 26, 14, 11, 5, -12],
         [11, 11, 4, -15, -25, -23, -6, 4, 6]),
        ("Hindu Kush, Himalaya, Central-East Indian Adivasi & SE Asian Massif / Zomia (Tibeto-Burman, Austroasiatic)",
         [72, 85, 98, 106, 105, 86, 81, 75, 71],
         [35, 30, 28, 23, 19, 19, 22, 28, 32]),
        ("Outer Insular SE Asia, New Guinea & Melanesia (Austronesian, Papuan — ~1,000+ languages, predominantly oral)",
         [111, 128, 152, 168, 166, 142, 118, 110],
         [4, 3, -2, -13, -18, -10, -9, -2]),
        ("Central, Northern & Western Outback Australia (Top End, Kimberley, Central Desert)",
         [120, 142, 139, 126, 119],
         [-11, -12, -26, -27, -20]),
        ("Aotearoa / New Zealand (Māori indigenous belt)",
         [172, 179, 178, 171],
         [-34, -37, -42, -40]),
        ("Northern Eurasia: Sápmi, Northern Ural, Siberia & Russian Far East (Sámi, Paleo-Siberian, Tungusic)",
         [16, 45, 95, 175, 178, 135, 85, 35, 15],
         [65, 64, 59, 60, 71, 73, 73, 70, 69]),
    ]

    ctc_tier_desc = {
        1: "Applies as-is (Tier 1): high-resource language with an established phoneme inventory T, written lexicon L, and text LM/LLM.",
        2: "Needs adaptation (Tier 2): tonal or sub-word language where word-level L∘G suffers homophony or OOV explosion; requires tonal T or sub-word L_sub∘G_sub.",
        3: "Cannot apply (Tier 3): unwritten or insufficient text to train language models (no standardized dictionary L or text LM/LLM).",
    }
    b2v_tier_desc = {
        1: "Applies as-is (Tier 1): mainstream TTS targets exist; synthesizes continuous vocal-tract acoustics + F0 pitch contour directly.",
        2: "Needs adaptation (Tier 2): lacks Text-To-Speech, so needs recorded audio for acoustic-target training.",
        3: "Cannot apply (Tier 3): none — no spoken language is blocked outright.",
    }

    def draw_svg_world_map(map_cx, map_cy, map_sx, map_sy, is_b2v):
        def proj_svg(lon_arr, lat_arr):
            rx, ry = _robinson_xy(lon_arr, lat_arr)
            dx = map_cx + map_sx * rx
            dy = map_cy + map_sy * ry
            pts = np.column_stack([dx, dy])
            px_py = ax_cd.transData.transform(pts)
            return SX(px_py[:, 0]), SY(px_py[:, 1])

        lats_edge = np.linspace(-60.0, 84.0, 90)
        lons_top = np.linspace(-180.0, 180.0, 90)
        lons_bot = np.linspace(180.0, -180.0, 90)
        lon_b = np.concatenate([
            np.full_like(lats_edge, -180.0),
            lons_top,
            np.full_like(lats_edge, 180.0),
            lons_bot,
        ])
        lat_b = np.concatenate([
            lats_edge,
            np.full_like(lons_top, 84.0),
            lats_edge[::-1],
            np.full_like(lons_bot, -60.0),
        ])
        bx, by = proj_svg(lon_b, lat_b)
        b_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(bx, by))
        s.append(f'<polygon points="{b_pts}" fill="{C_OCEAN}" stroke="#B8C4CE" stroke-width="{SW(1.15)}"/>')

        for glat in (-45, -23.5, 0, 23.5, 45, 66.5):
            glons = np.linspace(-180.0, 180.0, 75)
            gx, gy = proj_svg(glons, np.full_like(glons, glat))
            g_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(gx, gy))
            dash = "" if glat == 0 else ' stroke-dasharray="4.5,2.5"'
            s.append(f'<polyline points="{g_pts}" fill="none" stroke="{C_GRAT}" stroke-width="{SW(0.5)}"{dash}/>')
        for glon in (-120, -60, 0, 60, 120):
            glats = np.linspace(-60.0, 84.0, 60)
            gx, gy = proj_svg(np.full_like(glats, glon), glats)
            g_pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(gx, gy))
            s.append(f'<polyline points="{g_pts}" fill="none" stroke="{C_GRAT}" stroke-width="{SW(0.5)}" stroke-dasharray="1.8,2.2"/>')

        clip_pts = []
        for c_rec in countries:
            a3 = c_rec["a3"]
            name = c_rec["name"]
            t_ctc = _ctc_tier(a3)
            t_b2v = _b2v_tier(a3)
            tier = t_b2v if is_b2v else t_ctc
            fc = tier_fill[tier]
            panel_tag = "d" if is_b2v else "c"
            # 2026-11-12 official-language round (user: "鼠标指针放在某个国家上时，
            # 列出其官方语言"): the country's official language(s) ride directly
            # under its name so the tier reading is self-evident (e.g. Lao ->
            # Tier 3, unwritten-language cascade). Long lists cap at 8 entries.
            lang_line = ""
            langs = _OFFICIAL_LANGUAGES.get(a3)
            if langs:
                shown = "; ".join(langs[:8])
                extra = len(langs) - 8
                if extra > 0:
                    shown += f" +{extra} more"
                lang_line = f"\u2022 Official language(s): {shown}\n"
            tip_str = (
                f"{name} ({a3})\n"
                f"{lang_line}"
                f"\u2022 Brain-to-text (Panel c): {ctc_tier_desc[t_ctc]}\n"
                f"\u2022 Brain-to-voice (Panel d): {b2v_tier_desc[t_b2v]}"
            )
            s.append(f'<g class="f6-country" data-a3="{esc(a3)}" data-name="{esc(name)}" '
                     f'data-f6-tier="{tier}" data-f6-map="{panel_tag}" data-ctc-tier="{t_ctc}" '
                     f'data-b2v-tier="{t_b2v}" data-map="{panel_tag}">')
            s.append(f'<title>{esc(tip_str)}</title>')
            for ring in c_rec["rings"]:
                arr = np.asarray(ring, dtype=float)
                lat_c = np.clip(arr[:, 1], -60.0, 84.0)
                px, py = proj_svg(arr[:, 0], lat_c)
                p_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(px, py))
                s.append(f'<polygon points="{p_str}" fill="{fc}" stroke="#FFFFFF" stroke-width="{SW(0.42)}"/>')
                if a3 not in _CTC_TIER3_A3:
                    clip_pts.append(p_str)
            s.append("</g>")

        # Land clip (all countries except Tier-3 ones) so indigenous-belt fill
        # and hatching never spill into the ocean, mirroring main_fig_7.
        clip_id = f"f6-clip-{'d' if is_b2v else 'c'}"
        s.append(f'<defs><clipPath id="{clip_id}">')
        for cp in clip_pts:
            s.append(f'<polygon points="{cp}"/>')
        s.append('</clipPath></defs>')

        belt_fc = C_T2_MAP if is_b2v else C_T3_MAP
        for b_name, lons, lats in indigenous_belts:
            px, py = proj_svg(lons, lats)
            p_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(px, py))
            b_tip = (
                f"Indigenous belt: {b_name}\n"
                f"\u2022 Brain-to-text (Panel c): Cannot apply (Tier 3) — unwritten or insufficient text to train language models.\n"
                f"\u2022 Brain-to-voice (Panel d): Needs adaptation (Tier 2) — lacks Text-To-Speech, so needs recorded audio."
            )
            s.append(f'<g class="f6-belt" data-f6-tier="belt" data-f6-map="{"d" if is_b2v else "c"}">')
            s.append(f'<title>{esc(b_tip)}</title>')
            s.append(f'<g clip-path="url(#{clip_id})">')
            s.append(f'<polygon points="{p_str}" fill="{belt_fc}" fill-opacity="0.38" stroke="none"/>')
            for hx1, hy1, hx2, hy2 in hatch_poly_lines(px, py, step=6.0):
                s.append(f'<line x1="{hx1:.1f}" y1="{hy1:.1f}" x2="{hx2:.1f}" y2="{hy2:.1f}" '
                         f'stroke="{belt_fc}" stroke-width="{SW(1.05)}"/>')
            s.append(f'<polygon points="{p_str}" fill="none" stroke="{belt_fc}" '
                     f'stroke-width="{SW(0.95)}" stroke-dasharray="4.5,2.5"/>')
            s.append('</g>')
            s.append(f'<polygon points="{p_str}" fill="#ffffff" fill-opacity="0.001" stroke="none" pointer-events="all"/>')
            s.append("</g>")

    draw_svg_world_map(44.5, 40.5, 16.6, 18.5, is_b2v=False)
    draw_svg_world_map(139.5, 40.5, 16.6, 18.5, is_b2v=True)

    # Two separate clickable legends, one under Panel (c) and one under Panel (d)
    leg6_specs = [
        ("c", 235.0, [
            ("1", C_T1_MAP, C_T1_EC, C_T1_BG, "Applies as-is (Tier 1)"),
            ("2", C_T2_MAP, C_T2_EC, C_T2_BG, "Needs adaptation: tonal or sub-word (Tier 2)"),
            ("3", C_T3_MAP, C_T3_EC, C_T3_BG, "Cannot apply: unwritten or insufficient text to train language models (Tier 3)"),
            ("belt", C_T3_MAP, C_T3_MAP, C_T3_BG, "Indigenous belts: unwritten or insufficient text to train language models (Tier 3)"),
        ]),
        ("d", 1250.0, [
            ("1", C_T1_MAP, C_T1_EC, C_T1_BG, "Applies as-is (Tier 1)"),
            ("2", C_T2_MAP, C_T2_EC, C_T2_BG, "Needs adaptation: lacks Text-To-Speech, so needs recorded audio (Tier 2)"),
            ("3", C_T3_MAP, C_T3_EC, C_T3_BG, "Cannot apply: none (Tier 3)"),
            ("belt", C_T2_MAP, C_T2_MAP, C_T2_BG, "Indigenous belts: lacks Text-To-Speech, so needs recorded audio (Tier 2)"),
        ]),
    ]
    leg6_y0, leg6_dy = vh - 128.0, 29.0
    for leg_map, lx0, items in leg6_specs:
        for idx_l, (tier_k, fc_l, ec_l, hatch_bg, lab_l) in enumerate(items):
            ry_l = leg6_y0 + idx_l * leg6_dy
            panel_letter = "c" if leg_map == "c" else "d"
            s.append(f'<g class="f6-leg-item" data-f6-tier="{tier_k}" data-f6-map="{leg_map}" style="cursor:pointer">')
            s.append(f'<title>Click to highlight {esc(lab_l)} on the Panel ({panel_letter}) map '
                     f'and its pipeline modules above</title>')
            if tier_k == "belt":
                s.append(f'<rect x="{lx0:.1f}" y="{ry_l - 6.0:.1f}" width="18" height="11" '
                         f'fill="{hatch_bg}" stroke="{ec_l}" stroke-width="{SW(0.9)}" stroke-dasharray="3.5,2.0"/>')
                for hx1, hy1, hx2, hy2 in hatch_poly_lines(
                    np.array([lx0, lx0 + 18.0, lx0 + 18.0, lx0]),
                    np.array([ry_l - 6.0, ry_l - 6.0, ry_l + 5.0, ry_l + 5.0]),
                    step=4.8,
                ):
                    s.append(f'<line x1="{hx1:.1f}" y1="{hy1:.1f}" x2="{hx2:.1f}" y2="{hy2:.1f}" '
                             f'stroke="{ec_l}" stroke-width="{SW(0.95)}"/>')
            else:
                s.append(f'<rect x="{lx0:.1f}" y="{ry_l - 6.0:.1f}" width="18" height="11" '
                         f'fill="{fc_l}" stroke="{ec_l}" stroke-width="{SW(0.9)}"/>')
            s.append(f'<text x="{lx0 + 24.0:.1f}" y="{ry_l:.1f}" font-size="{8.2 * PT:.2f}" '
                     f'fill="#1f2937" dominant-baseline="central">{esc(lab_l)}</text>')
            s.append("</g>")

    plt.close(fig)
    return "\n".join(s), vw, vh


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__ · Interactive Edition</title>
<style>
  :root {
    --ink: #1f2937; --muted: #6b7280; --faint: #9aa3af;
    --line: #e5e7eb; --canvas: #f5f6f8; --card: #ffffff;
    --accent: #0f4c81; --accent-soft: rgba(15, 76, 129, .08);
    --repro: #3C5488; --repro-soft: rgba(60, 84, 136, .10);
  }
  * { box-sizing: border-box; }
  html { scroll-behavior: smooth; }
  body { margin: 0; background: var(--canvas); color: var(--ink);
         font: 14px/1.45 "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif;
         -webkit-font-smoothing: antialiased; }
  .wrap { max-width: 1600px; margin: 0 auto; padding: 0 28px 80px; }

  /* ---------- header ---------- */
  header { position: sticky; top: 0; z-index: 50; background: rgba(245, 246, 248, .92);
           backdrop-filter: blur(6px); border-bottom: 1px solid var(--line);
           margin: 0 -28px; padding: 12px 28px; }
  .head-row { display: flex; flex-wrap: wrap; align-items: baseline; gap: 8px 16px; max-width: 1600px; margin: 0 auto; }
  h1 { font-size: 17px; font-weight: 650; margin: 0; letter-spacing: .01em; }
  .head-tools { margin-left: auto; display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
  .stat { background: var(--card); border: 1px solid var(--line); border-radius: 999px;
          padding: 3px 12px; font-size: 12px; color: var(--muted); white-space: nowrap; }
  .stat b { color: var(--ink); font-weight: 650; }
  #search { border: 1px solid var(--line); border-radius: 999px; padding: 5px 14px;
            font-size: 12.5px; width: 240px; background: var(--card); color: var(--ink); outline: none; }
  #search:focus { border-color: var(--accent); }
  #reset { border: 1px solid var(--line); background: var(--card); border-radius: 999px;
           padding: 5px 14px; font-size: 12.5px; color: var(--muted); cursor: pointer; }
  #reset:hover { color: var(--ink); border-color: var(--faint); }

  /* ---------- filter bar ---------- */
  #panel { background: var(--card); border: 1px solid var(--line); border-radius: 14px;
           margin: 18px 0; padding: 14px 18px 10px; }
  .groups { display: flex; flex-wrap: wrap; gap: 10px 30px; }
  .fgroup { display: flex; flex-direction: column; gap: 5px; }
  .fg-title { font-size: 10.5px; font-weight: 700; letter-spacing: .08em;
              text-transform: uppercase; color: var(--faint); }
  .chips { display: flex; flex-wrap: wrap; gap: 5px; }
  .chip { display: inline-flex; align-items: center; gap: 6px; border: 1px solid var(--line);
          background: #fafbfc; color: var(--muted); border-radius: 999px;
          padding: 2.5px 11px; font-size: 12px; cursor: pointer; user-select: none;
          transition: all .12s ease; }
  .chip:hover { border-color: var(--faint); color: var(--ink); }
  .chip.on { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); font-weight: 600; }
  .chip .sw { width: 9px; height: 9px; border-radius: 50%; flex: none; }
  .chip .mg { font-size: 11px; flex: none; opacity: .85; }
  .date-range { display: inline-flex; align-items: center; gap: 6px; border: 1px solid var(--line);
                background: #fafbfc; border-radius: 999px; padding: 2px 11px; font-size: 12px;
                color: var(--muted); transition: all .12s ease; }
  .date-range:hover { border-color: var(--faint); }
  .date-range:focus-within { border-color: var(--accent); color: var(--ink); }
  .date-range.filtered { background: var(--accent-soft); border-color: var(--accent); color: var(--accent); }
  .date-range input[type="date"] { border: none; background: transparent; color: var(--ink);
                                   font: inherit; font-size: 12px; padding: 0; outline: none;
                                   cursor: pointer; }
  .date-range.filtered input[type="date"] { color: var(--accent); font-weight: 600; }
  .date-sep { color: var(--faint); user-select: none; }
  #date-clear { border: none; background: none; color: var(--muted); font-size: 14px;
                line-height: 1; padding: 0 2px; cursor: pointer; display: none; }
  #date-clear:hover { color: var(--ink); }
  .date-range.filtered #date-clear { display: inline-block; color: var(--accent); }

  /* ---------- view mode & quick toggles ---------- */
  .bottom-row { display: flex; flex-wrap: wrap; gap: 12px; align-items: center;
                border-top: 1px solid var(--line); margin-top: 12px; padding-top: 10px; }
  .mode-lbl { font-size: 10.5px; font-weight: 700; letter-spacing: .08em;
              text-transform: uppercase; color: var(--faint); }
  #modebar { display: inline-flex; border: 1px solid var(--line); border-radius: 999px; overflow: hidden; }
  #modebar button { border: none; background: transparent; padding: 4px 14px; font-size: 12px;
                    color: var(--muted); cursor: pointer; }
  #modebar button + button { border-left: 1px solid var(--line); }
  #modebar button.on { background: var(--accent); color: #fff; font-weight: 600; }
  #pinned-only-btn { border: 1px solid rgba(0, 107, 89, .38); background: rgba(0, 160, 135, .08);
                     color: #005f4f; border-radius: 999px; padding: 4px 14px; font-size: 12px;
                     font-weight: 600; cursor: pointer; transition: all .12s ease; }
  #pinned-only-btn:hover { border-color: #006b59; background: rgba(0, 160, 135, .15); }
  #pinned-only-btn.on { background: #006b59; border-color: #006b59; color: #fff; }
  #repro-only-btn { border: 1px solid rgba(60, 84, 136, .42); background: var(--repro-soft);
                    color: #2c406b; border-radius: 999px; padding: 4px 14px; font-size: 12px;
                    font-weight: 600; cursor: pointer; transition: all .12s ease; }
  #repro-only-btn:hover { border-color: var(--repro); background: rgba(60, 84, 136, .18); }
  #repro-only-btn.on { background: var(--repro); border-color: var(--repro); color: #fff; }
  .ext-links { margin-left: auto; display: flex; flex-wrap: wrap; gap: 8px; }
  .nav-pill { display: inline-flex; align-items: center; gap: 4px; border: 1px solid var(--line);
              background: #fafbfc; color: var(--accent); border-radius: 999px; padding: 4px 13px;
              font-size: 12px; font-weight: 600; text-decoration: none; transition: all .12s ease; }
  .nav-pill:hover { border-color: var(--accent); background: var(--accent-soft); }

  /* ---------- figure boxes ---------- */
  #benchbox, #fig6box, #pmbox, #mmbox { position: relative; background: var(--card); border: 1px solid var(--line);
                        border-radius: 14px; padding: 10px 10px 6px; }
  #benchbox:fullscreen, #benchbox:-webkit-full-screen, #benchbox.is-fullscreen,
  #fig6box:fullscreen, #fig6box:-webkit-full-screen, #fig6box.is-fullscreen,
  #pmbox:fullscreen, #pmbox:-webkit-full-screen, #pmbox.is-fullscreen,
  #mmbox:fullscreen, #mmbox:-webkit-full-screen, #mmbox.is-fullscreen {
    background: var(--card); border: none; border-radius: 0; padding: 14px 18px;
    display: flex; flex-direction: column; align-items: center; justify-content: flex-start;
    overflow: auto; width: 100vw; height: 100vh;
  }
  #benchbox.is-fullscreen, #fig6box.is-fullscreen, #pmbox.is-fullscreen, #mmbox.is-fullscreen { position: fixed; inset: 0; z-index: 9999; }
  #f2-svg, #f6-svg { width: 100%; height: auto; display: block; border-radius: 8px; }
  /* 2026-11-12 Insights round (user: "算了，还是新开一个section吧，就叫 Insights";
     "四个 panel 在网页中打开看上去很大，和其他图片不太适应，不要动 pdf 和 png，
     在 html 中处理一下"): the 4-panel phase/memory companion left the Overview
     box for its own "Insights" section and is capped at its natural CSS size
     (12.6 in x 96 dpi = 1210 px) and centered, so its panels no longer render
     larger than the Figure 2 / Figure 6 panels. The .pdf/.png outputs are
     untouched; full screen lifts the cap.
     2026-11-13 split round (user: "把 Insights 拆成两节吧，a b 是 insights，
     c d 是节约"): the companion is split into two 1x2 SVGs — #pm-svg (a, b)
     stays in "Insights", #mm-svg (c, d) goes to the new "Memory Savings"
     section; both keep the 1210 px cap. */
  #pm-svg, #mm-svg { width: 100%; max-width: 1210px; height: auto; display: block; margin: 0 auto; border-radius: 8px; }
  #benchbox:fullscreen #f2-svg, #benchbox:-webkit-full-screen #f2-svg, #benchbox.is-fullscreen #f2-svg,
  #fig6box:fullscreen #f6-svg, #fig6box:-webkit-full-screen #f6-svg, #fig6box.is-fullscreen #f6-svg {
    width: 100%; height: auto; max-width: 100%;
  }
  #pmbox:fullscreen #pm-svg, #pmbox:-webkit-full-screen #pm-svg, #pmbox.is-fullscreen #pm-svg,
  #mmbox:fullscreen #mm-svg, #mmbox:-webkit-full-screen #mm-svg, #mmbox.is-fullscreen #mm-svg {
    width: 100%; height: auto; max-width: 100%;
  }
  #f2-fs-btn, #f6-fs-btn, #pm-fs-btn, #mm-fs-btn { position: absolute; top: 14px; right: 16px; z-index: 10;
                           display: inline-flex; align-items: center; gap: 6px;
                           border: 1px solid var(--line); background: rgba(255, 255, 255, .92);
                           backdrop-filter: blur(4px); border-radius: 999px; padding: 5px 12px;
                           font-size: 12px; font-weight: 500; color: var(--muted); cursor: pointer;
                           box-shadow: 0 1px 4px rgba(15, 36, 64, .06); transition: all .12s ease; }
  #f2-fs-btn:hover, #f6-fs-btn:hover, #pm-fs-btn:hover, #mm-fs-btn:hover { color: var(--ink); border-color: var(--faint);
                                       box-shadow: 0 2px 8px rgba(15, 36, 64, .10); }
  #f2-fs-btn svg, #f6-fs-btn svg, #pm-fs-btn svg, #mm-fs-btn svg { width: 13px; height: 13px; stroke: currentColor; stroke-width: 1.8;
                                   fill: none; stroke-linecap: round; stroke-linejoin: round; flex: none; }
  #f2-svg text, #f6-svg text { font-family: "Segoe UI", "DejaVu Sans", Arial, sans-serif; }

  /* ---------- Figure 2 interactive SVG classes ---------- */
  .f2-slot { cursor: pointer; transition: opacity .15s ease; }
  .f2-slot .f2-hit { fill: transparent; }
  .f2-slot .f2-bar { transition: stroke-width .12s ease, fill-opacity .12s ease; }
  .f2-slot:hover .f2-bar, .f2-slot.hovered .f2-bar, .f2-slot.selected .f2-bar {
    stroke-width: 2.3px; fill-opacity: 0.58;
  }
  .f2-slot:hover .f2-lbl, .f2-slot.hovered .f2-lbl, .f2-slot.selected .f2-lbl,
  .f2-slot:hover .f2-val, .f2-slot.hovered .f2-val, .f2-slot.selected .f2-val {
    fill: var(--accent); font-weight: 700;
  }
  .f2-drop { transition: opacity .15s ease; }
  .f2-sess { cursor: pointer; transition: opacity .15s ease, transform .12s ease; }
  .f2-sess:hover circle, .f2-sess.hovered circle, .f2-sess.selected circle,
  .f2-sess:hover rect, .f2-sess.hovered rect, .f2-sess.selected rect,
  .f2-sess:hover polygon, .f2-sess.hovered polygon, .f2-sess.selected polygon {
    stroke-width: 2.8px; opacity: 1;
  }
  .f2-sess-leg:hover text { fill: var(--accent); font-weight: 700; }
  .lg-row text, #lg-repro-toggle text { transition: fill .12s ease; }
  .lg-row:hover text, #lg-repro-toggle:hover text { fill: var(--accent); font-weight: 600; }
  .lg-row.off text { fill: #c3c9d1; text-decoration: line-through; text-decoration-color: #c3c9d1; }
  .lg-row.off rect { opacity: .25; }
  #lg-repro-toggle.on text { fill: var(--repro); font-weight: 700; text-decoration: underline; }
  #lg-repro-toggle.on rect { fill-opacity: 0.72; stroke-width: 2.2px; }

  /* ---------- Figure 6 interactive SVG classes ---------- */
  .f6-node { cursor: pointer; transition: opacity .15s ease; }
  .f6-node .f6-box { transition: stroke-width .12s ease, filter .12s ease; }
  .f6-node:hover .f6-box, .f6-node.hovered .f6-box { stroke-width: 2.6px; filter: drop-shadow(0 2px 6px rgba(15,36,64,.14)); }
  .f6-country { cursor: pointer; transition: opacity .15s ease; }
  .f6-country polygon { transition: stroke .1s ease, stroke-width .1s ease, fill-opacity .12s ease; }
  .f6-country:hover polygon, .f6-country.hovered polygon {
    stroke: #0f172a; stroke-width: 1.8px;
  }
  .f6-belt { cursor: pointer; transition: opacity .15s ease; }
  .f6-belt:hover polygon, .f6-belt.hovered polygon { stroke-width: 2.2px; }
  .f6-leg-item text { transition: fill .12s ease; }
  .f6-leg-item:hover text { fill: var(--accent); font-weight: 600; }
  .f6-leg-item.active text { fill: var(--accent); font-weight: 700; text-decoration: underline; }
  .f6-leg-item.active rect { stroke-width: 2.3px; }
  .f6-dimmed { opacity: 0.16 !important; }

  /* ---------- CTC-TLG 3-limits diagnostic strip ---------- */
  .limits-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
                 gap: 12px; margin: 10px 4px 6px; padding-top: 10px; border-top: 1px solid var(--line); }
  .limit-card { background: #f8f9fb; border: 1px solid var(--line); border-left: 4px solid var(--repro);
                border-radius: 10px; padding: 11px 14px; font-size: 12.5px; line-height: 1.45; color: var(--ink); }
  .limit-card.t2 { border-left-color: #7D6618; }
  .limit-card.t3 { border-left-color: #6E3B23; }
  .limit-card .lc-head { font-size: 11px; font-weight: 700; letter-spacing: .06em;
                         text-transform: uppercase; color: var(--muted); margin-bottom: 4px;
                         display: flex; align-items: center; justify-content: space-between; }
  .limit-card .lc-badge { font-size: 10.5px; font-weight: 700; padding: 1px 8px;
                          border-radius: 999px; background: rgba(60, 84, 136, .12); color: var(--repro); }
  .limit-card.t2 .lc-badge { background: rgba(223, 194, 102, .28); color: #6b5510; }
  .limit-card.t3 .lc-badge { background: rgba(176, 108, 73, .22); color: #6E3B23; }
  .limit-card b { font-weight: 650; }

  /* ---------- floating SVG tooltip ---------- */
  #svg-tip { position: fixed; z-index: 10000; pointer-events: none; display: none;
             max-width: 390px; background: rgba(17, 24, 39, .95); color: #f9fafb;
             border-radius: 10px; padding: 8px 12px; font-size: 12px; line-height: 1.42;
             box-shadow: 0 6px 20px rgba(0, 0, 0, .22); white-space: pre-line; }

  /* ---------- section headers ---------- */
  .sec { display: flex; align-items: baseline; gap: 12px; margin: 30px 2px 12px; scroll-margin-top: 68px; }
  .sec h2 { font-size: 16px; font-weight: 650; margin: 0; letter-spacing: .01em; }
  .sec .count { color: var(--muted); font-size: 12.5px; }
  .sec .rule { flex: 1; border-top: 1px solid var(--line); transform: translateY(-4px); }
  .sec-btn { border: 1px solid var(--line); background: var(--card); border-radius: 999px;
             padding: 3px 12px; font-size: 12px; color: var(--muted); cursor: pointer; }
  .sec-btn:hover { color: var(--ink); border-color: var(--faint); }

  /* ---------- Table 1 (8-Study Reproduction Ledger) ---------- */
  #reprobox { background: var(--card); border: 1px solid var(--line); border-radius: 14px;
              padding: 14px 18px 12px; overflow-x: auto; }
  .cohort-sub { font-size: 12px; color: var(--muted); margin: 0 0 10px; line-height: 1.45; }
  .cohort-sub b { color: var(--ink); font-weight: 650; }
  .cohort-tbl { width: 100%; border-collapse: collapse; font-size: 12.5px; line-height: 1.38; }
  .cohort-tbl th { text-align: left; font-size: 10.8px; font-weight: 700; letter-spacing: .05em;
                   text-transform: uppercase; color: var(--muted); padding: 7px 10px;
                   border-top: 1.5px solid var(--ink); border-bottom: 1px solid var(--ink);
                   background: #fafbfc; white-space: nowrap; }
  .cohort-tbl td { padding: 7px 10px; border-bottom: 1px solid var(--line); vertical-align: top;
                   color: var(--ink); }
  .cohort-tbl tbody tr:last-child td { border-bottom: 1.5px solid var(--ink); }
  .repro-row { cursor: pointer; transition: background .12s ease, opacity .15s ease; }
  .repro-row:hover { background: #f4f7fb; }
  .repro-row.selected { background: var(--accent-soft); }
  .repro-row.selected td:first-child { color: var(--accent); font-weight: 700;
                                       box-shadow: inset 3px 0 0 var(--accent); }
  .repro-row.off { opacity: .25; }
  .cohort-tbl a { color: var(--accent); text-decoration: none; font-weight: 600; }
  .cohort-tbl a:hover { text-decoration: underline; }
  .cohort-foot { font-size: 11.5px; color: var(--muted); margin: 9px 0 0; line-height: 1.45; }
  .v-badge { display: inline-block; font-size: 10.5px; font-weight: 700; padding: 1.5px 8px;
             border-radius: 999px; white-space: nowrap; }
  .v-badge.ok { background: rgba(0, 160, 135, .13); color: #005f4f; }
  .v-badge.gap { background: rgba(192, 127, 0, .15); color: #7a4f00; }

  /* ---------- spotlight ---------- */
  #spotlight { position: relative; border: 1.5px solid var(--accent); border-radius: 16px;
               background: #fdfeff; padding: 22px 26px 16px; margin: 24px 0 22px; scroll-margin-top: 68px; }
  #spotlight::before { content: "SELECTED STUDY"; position: absolute; top: -9px; left: 22px;
                       background: var(--accent); color: #fff; font-size: 9.5px; font-weight: 700;
                       letter-spacing: .12em; padding: 2px 10px; border-radius: 999px; }
  #sp-close { position: absolute; top: 10px; right: 12px; border: none; background: none;
              font-size: 22px; line-height: 1; color: var(--faint); cursor: pointer; }
  #sp-close:hover { color: var(--ink); }
  .sp-nav { display: flex; gap: 10px; align-items: center; margin-top: 14px;
            border-top: 1px dashed var(--line); padding-top: 10px; }
  .sp-nav button { border: 1px solid var(--line); background: var(--card); border-radius: 999px;
                   padding: 4px 14px; font-size: 12px; color: var(--muted); cursor: pointer; }
  .sp-nav button:hover:not(:disabled) { color: var(--accent); border-color: var(--accent); }
  .sp-nav button:disabled { opacity: .35; cursor: default; }
  #sp-pos { font-size: 12px; color: var(--faint); margin-left: auto; }
  .repro-box { margin-top: 12px; border: 1px solid rgba(60, 84, 136, .32);
               background: linear-gradient(180deg, rgba(60, 84, 136, .06) 0%, #ffffff 48%);
               border-radius: 11px; padding: 11px 14px; }
  .repro-box-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px;
                    font-size: 11px; font-weight: 700; letter-spacing: .06em;
                    text-transform: uppercase; color: var(--repro); margin-bottom: 8px; }
  .repro-note { font-size: 12.2px; color: var(--muted); margin-top: 8px; line-height: 1.45; }

  /* ---------- study cards ---------- */
  #cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); gap: 16px; }
  .cards-row-break { grid-column: 1 / -1; height: 0; border-top: 1px dashed var(--line); margin: 2px 0; }
  .study-card { --spine: #9aa3af; background: var(--card); border: 1px solid var(--line);
                border-left: 4px solid var(--spine); border-radius: 12px; padding: 14px 16px 12px;
                cursor: pointer; transition: box-shadow .15s ease, border-color .15s ease, transform .15s ease; }
  .study-card:hover { box-shadow: 0 4px 18px rgba(15, 36, 64, .10); transform: translateY(-1px); }
  .study-card.selected { border-color: var(--accent); border-left-color: var(--accent);
                         box-shadow: 0 0 0 2px var(--accent-soft); }
  .study-card.pinned { border-color: rgba(0, 107, 89, .38); border-left: 4px solid #006b59;
                       background: linear-gradient(180deg, rgba(0, 160, 135, .045) 0%, #ffffff 42%); }
  .study-card.repro-card:not(.pinned) { border-color: rgba(60, 84, 136, .35); border-left: 4px solid var(--repro); }
  .sc-pin { display: inline-flex; align-items: center; gap: 3px; background: rgba(0, 160, 135, .13);
            color: #005f4f; font-size: 10.5px; font-weight: 700; letter-spacing: .04em;
            padding: 1px 7px; border-radius: 999px; white-space: nowrap; }
  .sc-repro-pill { display: inline-flex; align-items: center; gap: 3px; background: var(--repro-soft);
                   color: var(--repro); font-size: 10.5px; font-weight: 700; letter-spacing: .04em;
                   padding: 1px 7px; border-radius: 999px; white-space: nowrap; }
  .sc-head { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; }
  .sc-dot { width: 10px; height: 10px; border-radius: 50%; flex: none;
            box-shadow: 0 0 0 2px #fff, 0 0 0 3px rgba(0,0,0,.10); }
  .sc-title { font-size: 14.5px; font-weight: 650; }
  .sc-glyph { margin-left: auto; color: var(--muted); font-size: 12px; white-space: nowrap; }
  .sc-date { font-size: 11.5px; color: var(--faint); margin: 2px 0 1px; letter-spacing: .02em; }
  .sc-desc { font-size: 12.5px; color: var(--muted); margin: 2px 0 10px; }
  .sc-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 7px; }
  .sc-cell { background: #f8f9fb; border-radius: 9px; padding: 7px 10px 6px; }
  .sc-cell.span2 { grid-column: span 2; }
  .sc-cell .k { font-size: 9.5px; font-weight: 700; letter-spacing: .07em; color: var(--faint);
                text-transform: uppercase; margin-bottom: 2px; }
  .sc-cell .v { font-size: 12.5px; font-weight: 600; color: var(--ink); overflow-wrap: anywhere; }
  .sc-parts { font-size: 12.5px; color: var(--ink); margin-top: 9px; }
  .sc-parts .plabel { color: var(--faint); font-size: 11px; }
  .sc-links { margin-top: 8px; font-size: 12px; }
  .sc-links a { color: var(--accent); text-decoration: none; margin-right: 14px; font-weight: 600; }
  .sc-links a:hover { text-decoration: underline; }
  #cards-empty { display: none; color: var(--faint); font-size: 13px; padding: 26px 0;
                 text-align: center; }
</style>
</head>
<body>
<div id="svg-tip"></div>
<div class="wrap">
  <header>
    <div class="head-row">
      <h1>__TITLE__</h1>
      <div class="head-tools">
        <span class="stat">Reproduced <b id="vis-repro">8 / 8</b></span>
        <span class="stat">Studies <b id="vis-count">33 / 33</b></span>
        <span class="stat">Participants <b id="vis-people">18 / 18</b></span>
        <input id="search" type="search" placeholder="Search author, participant, pipeline, memory wall…">
        <button id="reset">Reset</button>
      </div>
    </div>
  </header>

  <div id="panel">
    <div class="groups" id="controls"></div>
    <div class="bottom-row">
      <span class="mode-lbl">Filter view</span>
      <div id="modebar">
        <button data-mode="dim" class="on">Dim</button>
        <button data-mode="hide">Hide in place</button>
      </div>
      <button id="pinned-only-btn" type="button" title="Toggle showing only the 2 open-data, online (closed-loop) studies with vocabulary &gt; 71.4k, online WER &lt; 12.8% and at least one vocal-paralysis (△) participant: Card et al. (2024) &amp; Jude et al. (2026a)">★ Open Data, Online, Vocab &gt; 71.4k, WER &lt; 12.8%, Vocal paralysis</button>
      <button id="repro-only-btn" type="button" title="Toggle showing only the 8 open-data studies reproduced end-to-end on our 16 GB VRAM / 64 GB RAM consumer PC">● 8 Reproduced Studies Only</button>
      <div class="ext-links">
        <a href="https://physics-lee.github.io/communication-neuroprostheses-timeline/" target="_blank" rel="noopener" class="nav-pill">Living Timeline ↗</a>
        <a href="https://github.com/Physics-Lee/communication-neuroprostheses-reproduction" target="_blank" rel="noopener" class="nav-pill">GitHub Repo ↗</a>
      </div>
    </div>
  </div>

  <div class="sec" id="bench-sec" style="margin-top:20px">
    <h2>Overview</h2>
    <span class="count" id="bench-count"></span>
    <span class="rule"></span>
    <button id="bench-toggle" class="sec-btn" type="button">Hide</button>
  </div>
  <div id="benchbox">
    <button id="f2-fs-btn" type="button" title="Toggle full screen (F)" aria-label="Toggle full screen">
      <svg id="f2-fs-ico" viewBox="0 0 16 16"><path d="M2 6V2h4M14 6V2h-4M2 10v4h4M14 10v4h-4"/></svg>
      <span id="f2-fs-lbl">Full screen</span>
    </button>
    <svg id="f2-svg" viewBox="0 0 __F2_VW__ __F2_VH__" xmlns="http://www.w3.org/2000/svg">__FIG2_FULL__</svg>
  </div>

  <div id="spotlight" style="display:none">
    <button id="sp-close" title="Close">×</button>
    <div id="sp-body"></div>
    <div class="sp-nav">
      <button id="sp-prev">← Previous</button>
      <button id="sp-next">Next →</button>
      <span id="sp-pos"></span>
    </div>
  </div>

  <div class="sec" id="pm-sec">
    <h2>Insights</h2>
    <span class="count" id="pm-count">Session-level phase diagrams (post-LM WER vs. greedy PER / raw CER)</span>
    <span class="rule"></span>
    <button id="pm-toggle" class="sec-btn" type="button">Hide</button>
  </div>
  <div id="pmbox">
    <button id="pm-fs-btn" type="button" title="Toggle full screen" aria-label="Toggle full screen">
      <svg id="pm-fs-ico" viewBox="0 0 16 16"><path d="M2 6V2h4M14 6V2h-4M2 10v4h4M14 10v4h-4"/></svg>
      <span id="pm-fs-lbl">Full screen</span>
    </button>
    __PM_PHASE_SVG__
  </div>

  <div class="sec" id="mm-sec">
    <h2>Memory Savings</h2>
    <span class="count" id="mm-count">The memory wall: our 64 GB RAM / 16 GB VRAM stack vs. official pipelines</span>
    <span class="rule"></span>
    <button id="mm-toggle" class="sec-btn" type="button">Hide</button>
  </div>
  <div id="mmbox">
    <button id="mm-fs-btn" type="button" title="Toggle full screen" aria-label="Toggle full screen">
      <svg id="mm-fs-ico" viewBox="0 0 16 16"><path d="M2 6V2h4M14 6V2h-4M2 10v4h4M14 10v4h-4"/></svg>
      <span id="mm-fs-lbl">Full screen</span>
    </button>
    __PM_MEMORY_SVG__
  </div>

  <div class="sec" id="fig6-sec">
    <h2>Limits of CTC-TLG</h2>
    <span class="count" id="fig6-count">Click any tier in either map legend (or any country / module) to compare Brain-to-text vs. Brain-to-Voice across ~7,000 spoken languages</span>
    <span class="rule"></span>
    <button id="fig6-toggle" class="sec-btn" type="button">Hide</button>
  </div>
  <div id="fig6box">
    <button id="f6-fs-btn" type="button" title="Toggle full screen" aria-label="Toggle full screen">
      <svg id="f6-fs-ico" viewBox="0 0 16 16"><path d="M2 6V2h4M14 6V2h-4M2 10v4h4M14 10v4h-4"/></svg>
      <span id="f6-fs-lbl">Full screen</span>
    </button>
    <svg id="f6-svg" viewBox="0 0 __F6_VW__ __F6_VH__" xmlns="http://www.w3.org/2000/svg">__FIG6_SVG__</svg>
    <div class="limits-grid" id="ctc-limits-strip">
      <div class="limit-card">
        <div class="lc-head"><span>1. Hardware Memory Wall (Fig. 2g, h)</span><span class="lc-badge">300 GB → 19.6 GB RAM</span></div>
        Composing a 125k-word pronunciation dictionary <b>L</b> with a 5-gram language model <b>G</b> into <code>TLG.fst = T ∘ min(det(L ∘ G))</code> inflates host RAM to <b>300 GB</b> (and second-pass LLM rescoring provisions <b>48–80 GB VRAM</b>). We fit the entire stack on one consumer PC (64 GB RAM, 1× RTX 5060 Ti 16 GB) via on-demand <code>mmap + .idx</code> WFST state tables (<b>19.6 GB RAM</b>) or KenLM tries (<b>1.46 GB RAM</b>) and decoupled FP16/4-bit LLM rescoring (<b>13.0 GB VRAM</b>).
      </div>
      <div class="limit-card t2">
        <div class="lc-head"><span>2. Acoustic Phase Transition (Fig. 2i)</span><span class="lc-badge">PER &lt; 25% Gate</span></div>
        Because English words average <b>3.4 phonemes</b> (<code>WER_no-LM = 1 − (1 − PER)^3.4</code>), the <code>CTC-TLG + LLM</code> cascade beats raw phoneme error (<code>WER &lt; PER</code>) in <b>79% (45/57)</b> of sessions when greedy <code>PER ≤ 25%</code>, but only <b>20% (4/20)</b> when <code>PER &gt; 25%</code>. Above 25% PER (and at ~60% PER in sparse inner speech), 125k WFST beam search hallucinates word insertions that even <code>NB=400</code> OPT/Qwen rescoring cannot rescue.
      </div>
      <div class="limit-card t3">
        <div class="lc-head"><span>3. Global Linguistic Ceiling (Fig. 6a–d)</span><span class="lc-badge">~85% Blocked in Tier 3</span></div>
        <code>CTC-TLG</code> is triple-gated by a discrete phoneme set <b>T</b>, a finite written dictionary <b>L</b>, and a text LM/LLM <b>G</b>: tonal and agglutinative languages require Tier 2 subword/tonal redesign, and <b>~85% of the world’s ~7,000 spoken languages (Tier 3)</b> lack a written lexicon or text LM and are blocked outright. Continuous <b>Brain-to-Voice</b> (Wairagkar et al., 2025, 2026) synthesizes vocal-tract acoustics directly at inference (<b>0% in Tier 3</b>).
      </div>
    </div>
  </div>

  <div class="sec" id="repro-sec">
    <h2>Reproduction Ledger</h2>
    <span class="count" id="repro-count">8 open-data studies reproduced end-to-end (click any row to inspect)</span>
    <span class="rule"></span>
    <button id="repro-toggle" class="sec-btn" type="button">Hide</button>
  </div>
  <div id="reprobox">
    <p class="cohort-sub">
      <b>Hardware target</b>: 1× NVIDIA GeForce RTX 5060 Ti (16 GB VRAM) + 64 GB DDR5 host RAM ·
      <b>★</b> satisfies Open Data, Online Decoding, Vocabulary &gt; 71.4k, Online WER &lt; 12.8% &amp; Vocal Paralysis (△) ·
      Click any row to highlight its bars across Figure 2 and open its reproduction breakdown.
    </p>
    <table class="cohort-tbl" aria-label="Table 1: 8-Study Consumer-GPU Reproduction Ledger">
      <thead>
        <tr>
          <th>Study &amp; Directory</th>
          <th>Modality &amp; Participant(s)</th>
          <th>Decoding Cascade</th>
          <th>Host RAM (Ours vs. Off.)</th>
          <th>GPU VRAM (Ours vs. Off.)</th>
          <th>Acoustic / Synthesis (Ours vs. Paper)</th>
          <th>Full-Pipeline WER (Ours vs. Paper)</th>
          <th>Status</th>
        </tr>
      </thead>
      <tbody id="repro-tbody"></tbody>
    </table>
    <p class="cohort-foot">
      All per-trial decode outputs (<code>*.tsv</code>, <code>*.csv</code>, <code>*.json</code>) are archived under each study’s <code>results/</code> folder and verified automatically by <code>build_session_per_vs_wer.py</code> and <code>render_figure2.py</code>.
    </p>
  </div>

  <div class="sec" id="cards-sec">
    <h2>Study Cards</h2>
    <span class="count" id="cards-count"></span>
    <span class="rule"></span>
    <button id="cards-toggle" class="sec-btn" type="button">Show</button>
  </div>
  <div id="cardsbox" style="display:none">
    <div id="cards"></div>
    <div id="cards-empty">No studies match the current filters.</div>
  </div>
</div>

<script>
'use strict';
const STUDIES = __STUDIES__;
const N = STUDIES.length;
const COLORS = { open:'#00A087', partial:'#c07f00', closed:'#cc3324', unknown:'#7F7F7F', reanalysis:'#7B3294' };
const AVAIL_LBL = { open:'open data', partial:'data on request', closed:'closed data',
                    unknown:'preprint', reanalysis:'reanalysis on open data' };
const MODE_LBL = { online:'online', offline:'offline' };
const ELEC_LBL = { ecog:'ECoG', utah:'Utah array', thin_film:'thin-film', microwire:'microwire' };
const TASK_LBL = { inner:'inner speech ◆', intended:'overt/mouthing/intended ◇', both:'inner ◆ + overt ◇' };
const OUT_LBL = { voice:'brain-to-voice ⬟', text:'brain-to-text ⬠', both:'voice ⬟ + text ⬠' };
const SHAPE_GLYPH = { o:'●', s:'■', h:'⬢', v:'▼' };
const LANE_LBL = { speech:'Speech', typing:'Typing', handwriting:'Handwriting' };
const STAR_STUDY_IDS = new Set(['card2024', 'jude2026a']);
const REPRO_STUDY_IDS = new Set(STUDIES.filter(r => r.is_repro).map(r => r.id));

function esc(s){ return String(s).replace(/[&<>"']/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }

const DIMS = {
  data_availability: ['open','partial','closed','unknown','reanalysis'],
  decoding_mode: ['online','offline'],
  decoded_output: ['voice','text'],
  electrode: ['ecog','utah','thin_film','microwire'],
  speech_ability: ['intact','paralysis'],
  task_instruction: ['inner','intended'],
  lane: ['speech','typing','handwriting'],
};
const sel = {}; for (const d in DIMS) sel[d] = new Set(DIMS[d]);
const ALL_DATES = STUDIES.map(r => r.date).sort();
const MIN_DATE = ALL_DATES[0] || '2021-01-01';
const MAX_DATE = ALL_DATES[ALL_DATES.length - 1] || '2026-12-31';
let searchStr = '', dateFrom = MIN_DATE, dateTo = MAX_DATE, mode = 'dim';
let selectedId = null, pinnedOnly = false, reproOnly = false, activeF6Tier = null, activeF6Map = null;

function matches(rec){
  if (pinnedOnly && !STAR_STUDY_IDS.has(rec.id)) return false;
  if (reproOnly && !REPRO_STUDY_IDS.has(rec.id)) return false;
  if (!sel.data_availability.has(rec.opn)) return false;
  if (!sel.decoding_mode.has(rec.decoding_mode)) return false;
  if (!sel.electrode.has(rec.electrode)) return false;
  if (!sel.lane.has(rec.lane)) return false;
  const dsel = sel.decoded_output;
  if (rec.decoded_output === 'both' ? dsel.size === 0 : !dsel.has(rec.decoded_output)) return false;
  const tsel = sel.task_instruction;
  if (rec.task_instruction === 'both' ? tsel.size === 0 : !tsel.has(rec.task_instruction)) return false;
  if (rec.speech_abilities.length &&
      !rec.speech_abilities.some(a => sel.speech_ability.has(a))) return false;
  const lo = dateFrom && dateTo && dateFrom > dateTo ? dateTo : dateFrom;
  const hi = dateFrom && dateTo && dateFrom > dateTo ? dateFrom : dateTo;
  if (lo && rec.date < lo) return false;
  if (hi && rec.date > hi) return false;
  if (searchStr){
    const repHay = rec.repro ? (rec.repro.dir + ' ' + rec.repro.cascade + ' ' + rec.repro.mitigation + ' ' + rec.repro.notes) : '';
    const hay = (rec.citation + ' ' + rec.id + ' ' + rec.participants.join(' ') + ' ' +
                 rec.descriptor + ' ' + (rec.metrics.sentences || '') + ' ' + repHay).toLowerCase();
    if (!hay.includes(searchStr)) return false;
  }
  return true;
}

function visibleStudies(){ return STUDIES.filter(matches); }

/* ---------- build Table 1 (8-Study Reproduction Ledger) ---------- */
const reproTbody = document.getElementById('repro-tbody');
const REPRO_ORDER = ['willett2021','willett2023','fan2023','card2024','wairagkar2025','kunz2025','jude2026a','jude2026b'];
for (const sid of REPRO_ORDER){
  const rec = STUDIES.find(r => r.id === sid);
  if (!rec || !rec.repro) continue;
  const rp = rec.repro;
  const isStar = STAR_STUDY_IDS.has(sid);
  const starPrefix = isStar ? '★ ' : '';
  const tr = document.createElement('tr');
  tr.className = 'repro-row';
  tr.dataset.id = sid;
  const repoHref = `https://github.com/Physics-Lee/communication-neuroprostheses-reproduction/tree/main/${rp.dir.replace(/\/$/, '')}`;
  const dataHref = rec.data_url || (rec.data_doi ? 'https://doi.org/' + encodeURIComponent(rec.data_doi) : '');
  tr.innerHTML = `
    <td><b>${starPrefix}${esc(rec.citation)}</b><br><a href="${esc(repoHref)}" target="_blank" rel="noopener" onclick="event.stopPropagation()"><code>${esc(rp.dir)}</code> ↗</a></td>
    <td>${esc(LANE_LBL[rec.lane])} · ${rec.participants.map(esc).join(', ')}<br><span style="font-size:11.5px;color:var(--muted)">${esc(rp.pub)}</span></td>
    <td><span style="font-weight:600">${esc(rp.cascade)}</span><br><span style="font-size:11.5px;color:var(--muted)">${esc(rp.mitigation)}</span></td>
    <td style="white-space:nowrap"><b style="color:var(--repro)">${esc(rp.ram_ours)}</b> vs. ${esc(rp.ram_off)}<br><span style="font-size:11px;color:var(--muted)">${esc(rp.ram_note)}</span></td>
    <td style="white-space:nowrap"><b style="color:var(--repro)">${esc(rp.vram_ours)}</b> vs. ${esc(rp.vram_off)}<br><span style="font-size:11px;color:var(--muted)">${esc(rp.vram_note)}</span></td>
    <td><b style="color:var(--repro)">${esc(rp.acoustic_ours)}</b><br><span style="font-size:11.5px;color:var(--muted)">Paper: ${esc(rp.acoustic_paper)}</span></td>
    <td><b style="color:var(--repro)">${esc(rp.wer_ours)}</b><br><span style="font-size:11.5px;color:var(--muted)">Paper: ${esc(rp.wer_paper)}</span></td>
    <td><span class="v-badge ${esc(rp.verdict_cls)}">${esc(rp.verdict)}</span>${dataHref ? `<br><a href="${esc(dataHref)}" target="_blank" rel="noopener" onclick="event.stopPropagation()" style="font-size:11px">Dryad ↗</a>` : ''}</td>
  `;
  tr.addEventListener('click', () => selectStudy(sid, true));
  tr.addEventListener('mouseenter', () => highlightStudy(sid, true));
  tr.addEventListener('mouseleave', () => highlightStudy(sid, false));
  reproTbody.appendChild(tr);
}

/* ---------- study cards & spotlight ---------- */
function cardHtml(rec, includeReproBox){
  const m = rec.metrics;
  const isStar = STAR_STUDY_IDS.has(rec.id);
  const pinBadge = isStar ? `<span class="sc-pin" title="Satisfies Open Data, Online Decoding, Vocab &gt; 71.4k, Online WER &lt; 12.8%, Vocal Paralysis (△)">★ Pinned</span>` : '';
  const reproBadge = rec.is_repro ? `<span class="sc-repro-pill" title="Reproduced end-to-end on our 16 GB VRAM / 64 GB RAM consumer PC">● Reproduced</span>` : '';
  const dataHref = rec.data_url || (rec.data_doi ? 'https://doi.org/' + encodeURIComponent(rec.data_doi) : '');
  const repoHref = rec.repro ? `https://github.com/Physics-Lee/communication-neuroprostheses-reproduction/tree/main/${rec.repro.dir.replace(/\/$/, '')}` : '';
  const links = [
    rec.doi ? `<a href="https://doi.org/${encodeURIComponent(rec.doi)}" target="_blank" rel="noopener">Paper ↗</a>` : '',
    dataHref ? `<a href="${esc(dataHref)}" target="_blank" rel="noopener">Dataset ↗</a>` : '',
    rec.url ? `<a href="${esc(rec.url)}" target="_blank" rel="noopener">Announcement ↗</a>` : '',
    repoHref ? `<a href="${esc(repoHref)}" target="_blank" rel="noopener">Reproduction Code (${esc(rec.repro.dir)}) ↗</a>` : '',
  ].join('');
  let reproHtml = '';
  if (includeReproBox && rec.repro){
    const rp = rec.repro;
    reproHtml = `
      <div class="repro-box">
        <div class="repro-box-head">
          <span>Consumer-GPU Reproduction (${esc(rp.dir)})</span>
          <span class="v-badge ${esc(rp.verdict_cls)}">${esc(rp.verdict)}</span>
          <span style="margin-left:auto;color:var(--muted);font-weight:600;text-transform:none">${esc(rp.cascade)}</span>
        </div>
        <div class="sc-grid">
          <div class="sc-cell"><div class="k">Acoustic / Synthesis (Ours vs. Paper)</div><div class="v" style="color:var(--repro)">${esc(rp.acoustic_ours)} <span style="color:var(--muted);font-weight:500">vs. ${esc(rp.acoustic_paper)}</span></div></div>
          <div class="sc-cell"><div class="k">Full-Pipeline WER (Ours vs. Paper)</div><div class="v" style="color:var(--repro)">${esc(rp.wer_ours)} <span style="color:var(--muted);font-weight:500">vs. ${esc(rp.wer_paper)}</span></div></div>
          <div class="sc-cell"><div class="k">Host RAM (64 GB Ceiling)</div><div class="v">${esc(rp.ram_ours)} vs. ${esc(rp.ram_off)} (${esc(rp.ram_note)})</div></div>
          <div class="sc-cell"><div class="k">GPU VRAM (16 GB RTX 5060 Ti)</div><div class="v">${esc(rp.vram_ours)} vs. ${esc(rp.vram_off)} (${esc(rp.vram_note)})</div></div>
        </div>
        <div class="repro-note"><b>Engineering &amp; Reproduction Notes:</b> ${esc(rp.notes)}</div>
      </div>`;
  }
  return `
    <div class="sc-head"><span class="sc-dot" style="background:${COLORS[rec.opn]}"></span>
      <span class="sc-title">${esc(rec.citation)}</span>${pinBadge}${reproBadge}
      <span class="sc-glyph">${SHAPE_GLYPH[rec.shape]} ${ELEC_LBL[rec.electrode]}</span></div>
    <div class="sc-date">${esc(rec.date)} · ${LANE_LBL[rec.lane]} lane · ${AVAIL_LBL[rec.opn]} · ${MODE_LBL[rec.decoding_mode]}</div>
    <div class="sc-desc">${esc(rec.descriptor)}</div>
    <div class="sc-grid">
      <div class="sc-cell"><div class="k">Vocabulary</div><div class="v">${esc(m.vocab_size || 'n/r')}</div></div>
      <div class="sc-cell"><div class="k">Speed</div><div class="v">${esc(m.speed || 'n/r')}</div></div>
      <div class="sc-cell"><div class="k">Error rate</div><div class="v">${esc(m.wer || 'n/r')}</div></div>
      <div class="sc-cell span2"><div class="k">Sentences / data volume</div><div class="v">${esc(m.sentences || 'n/r')}</div></div>
    </div>
    ${reproHtml}
    <div class="sc-parts"><span class="plabel">PARTICIPANTS (n=${rec.n_count})</span>&ensp;${rec.participants.map(esc).join(' · ')}</div>
    <div class="sc-parts"><span class="plabel">TASK</span>&ensp;${TASK_LBL[rec.task_instruction]} &emsp; <span class="plabel">OUTPUT</span>&ensp;${OUT_LBL[rec.decoded_output]}</div>
    <div class="sc-links">${links}</div>`;
}

const cardsBox = document.getElementById('cards');
function rebuildCards(visStudies){
  cardsBox.innerHTML = '';
  const reproStudies = visStudies.filter(r => r.is_repro);
  const restStudies = visStudies.filter(r => !r.is_repro);
  const appendCard = (rec) => {
    const d = document.createElement('div');
    d.className = 'study-card' + (STAR_STUDY_IDS.has(rec.id) ? ' pinned' : '') + (rec.is_repro ? ' repro-card' : '');
    d.dataset.id = rec.id;
    d.style.setProperty('--spine', COLORS[rec.opn]);
    d.innerHTML = cardHtml(rec, false);
    d.addEventListener('click', () => selectStudy(rec.id, true));
    d.addEventListener('mouseenter', () => highlightStudy(rec.id, true));
    d.addEventListener('mouseleave', () => highlightStudy(rec.id, false));
    cardsBox.appendChild(d);
  };
  for (const r of reproStudies) appendCard(r);
  if (reproStudies.length && restStudies.length){
    const br = document.createElement('div');
    br.className = 'cards-row-break';
    cardsBox.appendChild(br);
  }
  for (const r of restStudies) appendCard(r);
  document.getElementById('cards-empty').style.display = visStudies.length ? 'none' : '';
  document.getElementById('cards-count').textContent =
    visStudies.length + ' of ' + N + ' studies (' + reproStudies.length + ' reproduced)';
}

function highlightStudy(sid, on){
  document.querySelectorAll(`.f2-slot[data-id="${sid}"], .f2-sess[data-id="${sid}"]`).forEach(el => {
    el.classList.toggle('hovered', on);
  });
}

function selectStudy(id, scrollPage){
  selectedId = id;
  document.querySelectorAll('.study-card, .f2-slot, .f2-sess, .repro-row').forEach(c =>
    c.classList.toggle('selected', Boolean(id && c.dataset.id === id)));
  const sp = document.getElementById('spotlight');
  if (!id){ sp.style.display = 'none'; return; }
  const rec = STUDIES.find(r => r.id === id);
  if (!rec){ sp.style.display = 'none'; return; }
  document.getElementById('sp-body').innerHTML = cardHtml(rec, true);
  sp.style.display = '';
  updateSpotlightNav();
  if (scrollPage !== false){
    if (isFs()) exitFs();
    sp.scrollIntoView({behavior:'smooth', block:'nearest'});
  }
}

function updateSpotlightNav(){
  const list = visibleStudies().map(r => r.id);
  const i = list.indexOf(selectedId);
  document.getElementById('sp-pos').textContent =
    (i >= 0 ? (i + 1) + ' of ' + list.length + ' shown' : 'filtered out') +
    ' · ' + (STUDIES.findIndex(r => r.id === selectedId) + 1) + ' of ' + N + ' total';
  document.getElementById('sp-prev').disabled = i <= 0;
  document.getElementById('sp-next').disabled = i < 0 || i >= list.length - 1;
}

function stepSpotlight(delta){
  const list = visibleStudies().map(r => r.id), i = list.indexOf(selectedId);
  if (i >= 0 && i + delta >= 0 && i + delta < list.length)
    selectStudy(list[i + delta], false);
}

document.getElementById('sp-close').addEventListener('click', () => selectStudy(null));
document.getElementById('sp-prev').addEventListener('click', () => stepSpotlight(-1));
document.getElementById('sp-next').addEventListener('click', () => stepSpotlight(1));

/* ---------- apply filters ---------- */
const TOTAL_PIDS = new Set(STUDIES.flatMap(r => r.unique_pids)).size;
function apply(){
  const vis = visibleStudies();
  const visIds = new Set(vis.map(r => r.id));
  const visRepro = vis.filter(r => r.is_repro).length;
  const visPids = new Set();
  for (const r of vis) for (const p of r.unique_pids) visPids.add(p);

  document.getElementById('vis-repro').textContent = visRepro + ' / 8';
  document.getElementById('vis-count').textContent = vis.length + ' / ' + N;
  document.getElementById('vis-people').textContent = visPids.size + ' / ' + TOTAL_PIDS;

  document.querySelectorAll('.f2-slot, .f2-drop, .f2-sess').forEach(el => {
    const sid = el.dataset.id;
    const on = visIds.has(sid);
    if (mode === 'hide' && el.classList.contains('f2-sess')){
      el.style.display = on ? '' : 'none';
      el.style.opacity = '1';
    } else {
      el.style.display = '';
      el.style.opacity = on ? '1' : '0.13';
    }
    el.classList.toggle('selected', Boolean(selectedId && sid === selectedId));
  });

  document.querySelectorAll('.repro-row').forEach(row => {
    const sid = row.dataset.id;
    const on = visIds.has(sid);
    row.classList.toggle('off', !on);
    row.classList.toggle('selected', Boolean(selectedId && sid === selectedId));
  });

  document.getElementById('bench-count').textContent =
    vis.length + ' of ' + N + ' studies · ' + visRepro + ' of 8 open-data studies reproduced on 1× RTX 5060 Ti (16 GB VRAM, 64 GB RAM)';
  document.getElementById('repro-count').textContent =
    visRepro + ' of 8 open-data studies shown (click any row to inspect)';

  rebuildCards(vis);
  if (selectedId) updateSpotlightNav();
}

/* ---------- control panel ---------- */
const GROUPS = [
  {dim:'data_availability', title:'Data availability', opts:[
    {v:'open', label:'open', color:COLORS.open},
    {v:'partial', label:'on request', color:COLORS.partial},
    {v:'closed', label:'closed', color:COLORS.closed},
    {v:'unknown', label:'preprint', color:COLORS.unknown},
    {v:'reanalysis', label:'reanalysis', color:COLORS.reanalysis}]},
  {dim:'decoding_mode', title:'Decoding mode', opts:[
    {v:'online', label:'online', glyph:'●'},
    {v:'offline', label:'offline', glyph:'○'}]},
  {dim:'decoded_output', title:'Decoded output', opts:[
    {v:'voice', label:'brain-to-voice', glyph:'⬟'},
    {v:'text', label:'brain-to-text', glyph:'⬠'}]},
  {dim:'task_instruction', title:'Task instruction', opts:[
    {v:'inner', label:'inner', glyph:'◆'},
    {v:'intended', label:'overt/mouthing/intended', glyph:'◇'}]},
  {dim:'electrode', title:'Electrode', opts:[
    {v:'ecog', label:'ECoG', glyph:'●'},
    {v:'utah', label:'Utah', glyph:'■'},
    {v:'thin_film', label:'thin-film', glyph:'⬢'},
    {v:'microwire', label:'microwire', glyph:'▼'}]},
  {dim:'speech_ability', title:'Speech ability', opts:[
    {v:'intact', label:'can speak naturally', glyph:'▲'},
    {v:'paralysis', label:'vocal paralysis', glyph:'△'}]},
  {dim:'lane', title:'Modality', opts:[
    {v:'speech', label:'speech'},
    {v:'typing', label:'typing'},
    {v:'handwriting', label:'handwriting'}]},
];
const controls = document.getElementById('controls');
for (const grp of GROUPS){
  const g = document.createElement('div'); g.className = 'fgroup';
  const t = document.createElement('div'); t.className = 'fg-title'; t.textContent = grp.title;
  g.appendChild(t);
  const chips = document.createElement('div'); chips.className = 'chips';
  for (const o of grp.opts){
    const chip = document.createElement('button');
    chip.type = 'button'; chip.className = 'chip on';
    chip.dataset.dim = grp.dim; chip.dataset.val = o.v;
    if (o.color){ const sw = document.createElement('span'); sw.className = 'sw';
      sw.style.background = o.color; chip.appendChild(sw); }
    if (o.glyph){ const mg = document.createElement('span'); mg.className = 'mg';
      mg.textContent = o.glyph; chip.appendChild(mg); }
    chip.appendChild(document.createTextNode(o.label));
    chip.addEventListener('click', () =>
      toggleVal(grp.dim, o.v, !sel[grp.dim].has(o.v)));
    chips.appendChild(chip);
  }
  g.appendChild(chips);
  controls.appendChild(g);
}

const dateGrp = document.createElement('div'); dateGrp.className = 'fgroup';
const dateTitle = document.createElement('div'); dateTitle.className = 'fg-title'; dateTitle.textContent = 'Time range';
dateGrp.appendChild(dateTitle);
const dateWrap = document.createElement('div'); dateWrap.className = 'date-range';
dateWrap.innerHTML =
  `<input id="date-from" type="date" aria-label="Start date" title="Start date" min="2021-01-01" max="2027-12-31" value="${MIN_DATE}">` +
  `<span class="date-sep">–</span>` +
  `<input id="date-to" type="date" aria-label="End date" title="End date" min="2021-01-01" max="2027-12-31" value="${MAX_DATE}">` +
  `<button id="date-clear" type="button" title="Reset time range" aria-label="Reset time range">×</button>`;
dateGrp.appendChild(dateWrap);
controls.appendChild(dateGrp);
const dateFromEl = document.getElementById('date-from');
const dateToEl = document.getElementById('date-to');
const dateClearEl = document.getElementById('date-clear');
function onDateChange(){
  dateFrom = dateFromEl.value || '';
  dateTo = dateToEl.value || '';
  syncUI(); apply();
}
dateFromEl.addEventListener('input', onDateChange);
dateFromEl.addEventListener('change', onDateChange);
dateToEl.addEventListener('input', onDateChange);
dateToEl.addEventListener('change', onDateChange);
dateClearEl.addEventListener('click', () => {
  dateFrom = MIN_DATE; dateTo = MAX_DATE;
  dateFromEl.value = MIN_DATE; dateToEl.value = MAX_DATE;
  syncUI(); apply();
});

function syncUI(){
  document.querySelectorAll('.chip[data-dim]').forEach(chip => {
    chip.classList.toggle('on', sel[chip.dataset.dim].has(chip.dataset.val));
  });
  document.querySelectorAll('.lg-row').forEach(row => {
    row.classList.toggle('off', !sel[row.dataset.dim].has(row.dataset.val));
  });
  const isDateFiltered = Boolean((dateFrom && dateFrom > MIN_DATE) || (dateTo && dateTo < MAX_DATE));
  dateWrap.classList.toggle('filtered', isDateFiltered);
  document.getElementById('pinned-only-btn').classList.toggle('on', pinnedOnly);
  document.getElementById('repro-only-btn').classList.toggle('on', reproOnly);
  const lgRepro = document.getElementById('lg-repro-toggle');
  if (lgRepro) lgRepro.classList.toggle('on', reproOnly);
}

function toggleVal(dim, val, on){
  if (on) sel[dim].add(val); else sel[dim].delete(val);
  syncUI(); apply();
}

document.getElementById('pinned-only-btn').addEventListener('click', () => {
  pinnedOnly = !pinnedOnly;
  syncUI(); apply();
});
document.getElementById('repro-only-btn').addEventListener('click', () => {
  reproOnly = !reproOnly;
  syncUI(); apply();
});
const lgReproEl = document.getElementById('lg-repro-toggle');
if (lgReproEl){
  lgReproEl.addEventListener('click', ev => {
    ev.stopPropagation();
    reproOnly = !reproOnly;
    syncUI(); apply();
  });
}

document.querySelectorAll('.lg-row').forEach(row =>
  row.addEventListener('click', ev => {
    ev.stopPropagation();
    const dim = row.dataset.dim, val = row.dataset.val;
    toggleVal(dim, val, !sel[dim].has(val));
  }));

document.querySelectorAll('#modebar button').forEach(b =>
  b.addEventListener('click', () => {
    mode = b.dataset.mode;
    document.querySelectorAll('#modebar button').forEach(x =>
      x.classList.toggle('on', x === b));
    apply();
  }));

document.getElementById('reset').addEventListener('click', () => {
  for (const d in DIMS) sel[d] = new Set(DIMS[d]);
  searchStr = ''; document.getElementById('search').value = '';
  pinnedOnly = false; reproOnly = false;
  dateFrom = MIN_DATE; dateTo = MAX_DATE;
  dateFromEl.value = MIN_DATE; dateToEl.value = MAX_DATE;
  mode = 'dim';
  document.querySelectorAll('#modebar button').forEach(x =>
    x.classList.toggle('on', x.dataset.mode === 'dim'));
  setF6Tier(null);
  selectStudy(null); syncUI(); apply();
});

/* ---------- floating tooltip & Figure 2/6 interactivity ---------- */
const tipEl = document.getElementById('svg-tip');
function showTip(ev, text){
  if (!text) return;
  tipEl.textContent = text;
  tipEl.style.display = 'block';
  moveTip(ev);
}
function moveTip(ev){
  if (tipEl.style.display === 'none') return;
  const pad = 14;
  let x = ev.clientX + pad, y = ev.clientY + pad;
  const w = tipEl.offsetWidth || 320, h = tipEl.offsetHeight || 80;
  if (x + w + 12 > window.innerWidth) x = Math.max(8, ev.clientX - w - pad);
  if (y + h + 12 > window.innerHeight) y = Math.max(8, ev.clientY - h - pad);
  tipEl.style.left = x + 'px';
  tipEl.style.top = y + 'px';
}
function hideTip(){ tipEl.style.display = 'none'; }

document.querySelectorAll('.f2-slot, .f2-sess, .f2-sess-leg').forEach(el => {
  const sid = el.dataset.id;
  const tNode = el.querySelector('title');
  const tipText = tNode ? tNode.textContent : '';
  el.addEventListener('click', ev => {
    ev.stopPropagation();
    selectStudy(sid, true);
  });
  el.addEventListener('mouseenter', ev => {
    highlightStudy(sid, true);
    showTip(ev, tipText);
  });
  el.addEventListener('mousemove', moveTip);
  el.addEventListener('mouseleave', () => {
    highlightStudy(sid, false);
    hideTip();
  });
});

/* ---------- Figure 6 interactivity ---------- */
function setF6Tier(tier, map){
  if (activeF6Tier === tier && activeF6Map === map){
    activeF6Tier = null;
    activeF6Map = null;
  } else {
    activeF6Tier = tier;
    activeF6Map = map;
  }
  document.querySelectorAll('.f6-leg-item').forEach(it => {
    it.classList.toggle('active', it.dataset.f6Tier === activeF6Tier && it.dataset.f6Map === activeF6Map);
  });
  document.querySelectorAll('#f6-svg .f6-country, #f6-svg .f6-belt, #f6-svg .f6-node[data-f6-tier]').forEach(el => {
    if (!activeF6Tier){
      el.classList.remove('f6-dimmed');
    } else {
      const sameMap = (el.dataset.f6Map || el.dataset.map) === activeF6Map;
      el.classList.toggle('f6-dimmed', !(sameMap && el.dataset.f6Tier === activeF6Tier));
    }
  });
}

document.querySelectorAll('.f6-leg-item').forEach(it => {
  it.addEventListener('click', ev => {
    ev.stopPropagation();
    setF6Tier(it.dataset.f6Tier, it.dataset.f6Map);
  });
});

document.querySelectorAll('#f6-svg .f6-country').forEach(cEl => {
  const a3 = cEl.dataset.a3;
  const tNode = cEl.querySelector('title');
  const tipText = tNode ? tNode.textContent : '';
  cEl.addEventListener('mouseenter', ev => {
    document.querySelectorAll(`#f6-svg .f6-country[data-a3="${a3}"]`).forEach(x => x.classList.add('hovered'));
    showTip(ev, tipText);
  });
  cEl.addEventListener('mousemove', moveTip);
  cEl.addEventListener('mouseleave', () => {
    document.querySelectorAll(`#f6-svg .f6-country[data-a3="${a3}"]`).forEach(x => x.classList.remove('hovered'));
    hideTip();
  });
  cEl.addEventListener('click', ev => {
    ev.stopPropagation();
    setF6Tier(cEl.dataset.f6Tier, cEl.dataset.f6Map || cEl.dataset.map);
  });
});

document.querySelectorAll('#f6-svg .f6-belt, #f6-svg .f6-node').forEach(el => {
  const tNode = el.querySelector('title');
  const tipText = tNode ? tNode.textContent : '';
  el.addEventListener('mouseenter', ev => showTip(ev, tipText));
  el.addEventListener('mousemove', moveTip);
  el.addEventListener('mouseleave', hideTip);
  el.addEventListener('click', ev => {
    ev.stopPropagation();
    if (el.dataset.id){
      selectStudy(el.dataset.id, true);
    } else if (el.dataset.f6Tier){
      setF6Tier(el.dataset.f6Tier, el.dataset.f6Map || el.dataset.map);
    }
  });
});

/* ---------- full screen & collapsible sections ---------- */
const benchbox = document.getElementById('benchbox');
const f2FsBtn = document.getElementById('f2-fs-btn');
const f2FsLbl = document.getElementById('f2-fs-lbl');
const f2FsIco = document.getElementById('f2-fs-ico');
const benchToggle = document.getElementById('bench-toggle');

const fig6box = document.getElementById('fig6box');
const f6FsBtn = document.getElementById('f6-fs-btn');
const f6FsLbl = document.getElementById('f6-fs-lbl');
const f6FsIco = document.getElementById('f6-fs-ico');
const fig6Toggle = document.getElementById('fig6-toggle');

const pmbox = document.getElementById('pmbox');
const pmFsBtn = document.getElementById('pm-fs-btn');
const pmFsLbl = document.getElementById('pm-fs-lbl');
const pmFsIco = document.getElementById('pm-fs-ico');
const pmToggle = document.getElementById('pm-toggle');

const mmbox = document.getElementById('mmbox');
const mmFsBtn = document.getElementById('mm-fs-btn');
const mmFsLbl = document.getElementById('mm-fs-lbl');
const mmFsIco = document.getElementById('mm-fs-ico');
const mmToggle = document.getElementById('mm-toggle');

const reprobox = document.getElementById('reprobox');
const reproToggle = document.getElementById('repro-toggle');
const cardsbox = document.getElementById('cardsbox');
const cardsToggle = document.getElementById('cards-toggle');

const FS_ENTER_PATH = 'M2 6V2h4M14 6V2h-4M2 10v4h4M14 10v4h-4';
const FS_EXIT_PATH  = 'M6 2v4H2M10 2v4h4M6 14v-4H2M10 14v-4h4';

function isBoxFs(box){
  return Boolean(document.fullscreenElement === box ||
                 document.webkitFullscreenElement === box ||
                 box.classList.contains('is-fullscreen'));
}
function isFs(){ return isBoxFs(benchbox) || isBoxFs(fig6box) || isBoxFs(pmbox) || isBoxFs(mmbox); }
function syncFsUI(){
  const on2 = isBoxFs(benchbox);
  f2FsLbl.textContent = on2 ? 'Exit full screen' : 'Full screen';
  f2FsIco.innerHTML = `<path d="${on2 ? FS_EXIT_PATH : FS_ENTER_PATH}"/>`;
  const on6 = isBoxFs(fig6box);
  f6FsLbl.textContent = on6 ? 'Exit full screen' : 'Full screen';
  f6FsIco.innerHTML = `<path d="${on6 ? FS_EXIT_PATH : FS_ENTER_PATH}"/>`;
  const onP = isBoxFs(pmbox);
  pmFsLbl.textContent = onP ? 'Exit full screen' : 'Full screen';
  pmFsIco.innerHTML = `<path d="${onP ? FS_EXIT_PATH : FS_ENTER_PATH}"/>`;
  const onM = isBoxFs(mmbox);
  mmFsLbl.textContent = onM ? 'Exit full screen' : 'Full screen';
  mmFsIco.innerHTML = `<path d="${onM ? FS_EXIT_PATH : FS_ENTER_PATH}"/>`;
  document.body.style.overflow =
    (benchbox.classList.contains('is-fullscreen') || fig6box.classList.contains('is-fullscreen') ||
     pmbox.classList.contains('is-fullscreen') || mmbox.classList.contains('is-fullscreen')) ? 'hidden' : '';
}
function exitFs(){
  if ((document.fullscreenElement === benchbox || document.fullscreenElement === fig6box ||
       document.fullscreenElement === pmbox || document.fullscreenElement === mmbox) && document.exitFullscreen)
    document.exitFullscreen().catch(() => {});
  else if ((document.webkitFullscreenElement === benchbox || document.webkitFullscreenElement === fig6box ||
            document.webkitFullscreenElement === pmbox || document.webkitFullscreenElement === mmbox) &&
           document.webkitExitFullscreen)
    document.webkitExitFullscreen();
  benchbox.classList.remove('is-fullscreen');
  fig6box.classList.remove('is-fullscreen');
  pmbox.classList.remove('is-fullscreen');
  mmbox.classList.remove('is-fullscreen');
  syncFsUI();
}
function toggleBoxFs(box){
  if (isBoxFs(box)){ exitFs(); return; }
  if (isFs()) exitFs();
  const req = box.requestFullscreen || box.webkitRequestFullscreen;
  if (req){
    try {
      const p = req.call(box);
      if (p && typeof p.catch === 'function')
        p.catch(() => { box.classList.add('is-fullscreen'); syncFsUI(); });
    } catch (_) {
      box.classList.add('is-fullscreen'); syncFsUI();
    }
  } else {
    box.classList.add('is-fullscreen'); syncFsUI();
  }
}
f2FsBtn.addEventListener('click', ev => { ev.stopPropagation(); toggleBoxFs(benchbox); });
f6FsBtn.addEventListener('click', ev => { ev.stopPropagation(); toggleBoxFs(fig6box); });
pmFsBtn.addEventListener('click', ev => { ev.stopPropagation(); toggleBoxFs(pmbox); });
mmFsBtn.addEventListener('click', ev => { ev.stopPropagation(); toggleBoxFs(mmbox); });
document.addEventListener('fullscreenchange', syncFsUI);
document.addEventListener('webkitfullscreenchange', syncFsUI);

benchToggle.addEventListener('click', () => {
  const hidden = benchbox.style.display === 'none';
  benchbox.style.display = hidden ? '' : 'none';
  benchToggle.textContent = hidden ? 'Hide' : 'Show';
});
fig6Toggle.addEventListener('click', () => {
  const hidden = fig6box.style.display === 'none';
  fig6box.style.display = hidden ? '' : 'none';
  fig6Toggle.textContent = hidden ? 'Hide' : 'Show';
});
pmToggle.addEventListener('click', () => {
  const hidden = pmbox.style.display === 'none';
  pmbox.style.display = hidden ? '' : 'none';
  pmToggle.textContent = hidden ? 'Hide' : 'Show';
});
mmToggle.addEventListener('click', () => {
  const hidden = mmbox.style.display === 'none';
  mmbox.style.display = hidden ? '' : 'none';
  mmToggle.textContent = hidden ? 'Hide' : 'Show';
});
reproToggle.addEventListener('click', () => {
  const hidden = reprobox.style.display === 'none';
  reprobox.style.display = hidden ? '' : 'none';
  reproToggle.textContent = hidden ? 'Hide' : 'Show';
});
cardsToggle.addEventListener('click', () => {
  const hidden = cardsbox.style.display === 'none';
  cardsbox.style.display = hidden ? '' : 'none';
  cardsToggle.textContent = hidden ? 'Hide' : 'Show';
});

document.addEventListener('keydown', e => {
  const tag = (e.target && e.target.tagName) || '';
  if (tag === 'INPUT' || tag === 'TEXTAREA') return;
  if ((e.key === 'f' || e.key === 'F') && !e.ctrlKey && !e.metaKey && !e.altKey){
    e.preventDefault(); toggleBoxFs(benchbox); return;
  }
  if (e.key === 'Escape' && isFs()){
    exitFs(); return;
  }
  if (!selectedId) return;
  if (e.key === 'Escape') selectStudy(null);
  else if (e.key === 'ArrowLeft') stepSpotlight(-1);
  else if (e.key === 'ArrowRight') stepSpotlight(1);
});

let debounce = null;
document.getElementById('search').addEventListener('input', e => {
  clearTimeout(debounce);
  debounce = setTimeout(() => { searchStr = e.target.value.trim().toLowerCase(); apply(); }, 120);
});

syncUI();
apply();
</script>
</body>
</html>
"""


def export_interactive_html(
    out_html,
    f2_svg_tuple,
    verify_overlaps=True,
    pm_svgs=None,
):
    """Write the single-file interactive HTML edition showcasing Figure 2
    (Panels a-f), the phase-diagram panels (a)/(b) in the "Insights"
    section, the memory-wall panels (c)/(d) in their own "Memory Savings"
    section, and Figure 6 (The Limit of CTC-TLG, Panels a-d).

    Parameters
    ----------
    out_html : str
        Destination path for ``communication_neuroprostheses_reproduction.html``.
    f2_svg_tuple : tuple of (str, int, int)
        ``(svg_inner, vw, vh)`` produced by :func:`_build_fig2_full_svg`.
    verify_overlaps : bool, optional
        Whether to run the pairwise text overlap check on Figure 6.
    pm_svgs : dict of str to tuple of (str, int, int), optional
        ``{"phase": (svg, vw, vh), "memory": (svg, vw, vh)}`` produced by
        ``render_phase_memory_figure.build_html_svgs``. When ``None``
        (default) it is built in-process via a deferred import, so CI can
        run ``render_figure2.py`` before ``render_phase_memory_figure.py``
        without stale artifacts.

    Notes
    -----
    2026-11-12 HTML rounds: the companion first continued the Overview box
    under the blue reproduction brace (user: "续着 overview 画"), then moved
    to its own "Insights" section at native size (user: "算了，还是新开一个
    section 吧，就叫 Insights"; "四个 panel 在网页中打开看上去很大，和其他
    图片不太适应") -- the page CSS caps ``#pm-svg`` at 1210 px (12.6 in x
    96 dpi) and centers it, without touching the ``.pdf``/``.png`` outputs.
    2026-11-13 split round (user: "把 Insights 拆成两节吧，a b 是 insights，
    c d 是节约"): the single 4-panel SVG is replaced by two 1x2 SVGs --
    ``#pm-svg`` (panels a, b) stays in "Insights" and ``#mm-svg`` (panels
    c, d) moves to the new "Memory Savings" section, each with its own
    full-screen button and the same 1210 px cap.
    """
    f2_svg_inner, f2_vw, f2_vh = f2_svg_tuple
    if pm_svgs is None:
        # Deferred import: render_phase_memory_figure imports symbols from
        # this module, so a top-level import would be circular.
        from render_phase_memory_figure import build_html_svgs
        pm_svgs = build_html_svgs()
    pm_svg_element = pm_svgs["phase"][0]
    mm_svg_element = pm_svgs["memory"][0]
    f6_svg_inner, f6_vw, f6_vh = _build_fig6_svg(verify_overlaps=verify_overlaps)
    studies_list = json.loads(
        zlib.decompress(base64.b64decode(_STUDIES_B64)).decode("utf-8")
    )
    studies_json = json.dumps(studies_list, ensure_ascii=False).replace("</", "<\\/")
    title_str = (
        "Communication Neuroprostheses, Reproduced on a Consumer PC"
    )
    doc = (
        _HTML_TEMPLATE
        .replace("__TITLE__", html_lib.escape(title_str))
        .replace("__FIG2_FULL__", f2_svg_inner)
        .replace("__F2_VW__", str(f2_vw))
        .replace("__F2_VH__", str(f2_vh))
        .replace("__PM_PHASE_SVG__", pm_svg_element)
        .replace("__PM_MEMORY_SVG__", mm_svg_element)
        .replace("__FIG6_SVG__", f6_svg_inner)
        .replace("__F6_VW__", str(f6_vw))
        .replace("__F6_VH__", str(f6_vh))
        .replace("__STUDIES__", studies_json)
    )
    with open(out_html, "w", encoding="utf-8", newline="\n") as f:
        f.write(doc)

    # Also keep index.html redirect in sync alongside out_html
    out_dir = os.path.dirname(os.path.abspath(out_html))
    html_name = os.path.basename(out_html)
    index_path = os.path.join(out_dir, "index.html")
    index_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="0; url={html_lib.escape(html_name)}">
<title>{html_lib.escape(title_str)} — Interactive Edition</title>
</head>
<body>
<p>Redirecting to the
<a href="{html_lib.escape(html_name)}">interactive reproduction benchmark</a>…</p>
</body>
</html>
"""
    with open(index_path, "w", encoding="utf-8", newline="\n") as idxf:
        idxf.write(index_doc)


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
                    help="Output stem for .png, .pdf, and .html")
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
    if os.path.exists(args.out + ".html"):
        print(f"[ok] Interactive edition (Figure 2 + Figure 6) -> {args.out}.html")
    print("     Overlap check: ALL CLEAN")


if __name__ == "__main__":
    main()
