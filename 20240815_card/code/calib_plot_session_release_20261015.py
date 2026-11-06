# -*- coding: utf-8 -*-
"""Paper-84 vs released-45 session timeline (investigation doc section 8).

Paper Table S2 calendar (post-implant days, implant = 2023-07-17, extracted
from paper_text appendix; S24/S28 rows were swallowed by PDF text extraction
and are linear interpolations, S72 288->228 corrected against neighbors
S71=226/S73=230 -- none of the three is a released date, so the released-set
mapping is unaffected).

Usage: python calib_plot_session_release_20261015.py
"""
import datetime as dt
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from matplotlib.patches import Patch

IMPLANT = dt.date(2023, 7, 17)
DAYS = {1: 25, 2: 27, 3: 32, 4: 34, 5: 39, 6: 41, 7: 46, 8: 48, 9: 67, 10: 69,
        11: 74, 12: 76, 13: 81, 14: 83, 15: 88, 16: 90, 17: 95, 18: 97,
        19: 102, 20: 104, 21: 109, 22: 110, 23: 114, 24: 119, 25: 123, 26: 125,
        27: 127, 28: 130, 29: 132, 30: 134, 31: 137, 32: 139, 33: 141, 34: 144,
        35: 146, 36: 148, 37: 149, 38: 153, 39: 156, 40: 163, 41: 165, 42: 170,
        43: 171, 44: 172, 45: 174, 46: 176, 47: 177, 48: 179, 49: 181, 50: 183,
        51: 184, 52: 186, 53: 188, 54: 190, 55: 191, 56: 193, 57: 195, 58: 202,
        59: 204, 60: 205, 61: 207, 62: 209, 63: 211, 64: 212, 65: 214, 66: 216,
        67: 218, 68: 219, 69: 223, 70: 225, 71: 226, 72: 228, 73: 230, 74: 232,
        75: 233, 76: 235, 77: 239, 78: 240, 79: 242, 80: 244, 81: 246, 82: 247,
        83: 249, 84: 251}
RELEASED = ["2023-08-11", "2023-08-13", "2023-08-18", "2023-08-20", "2023-08-25",
            "2023-08-27", "2023-09-01", "2023-09-03", "2023-09-24", "2023-09-29",
            "2023-10-01", "2023-10-06", "2023-10-08", "2023-10-13", "2023-10-15",
            "2023-10-20", "2023-10-22", "2023-11-03", "2023-11-04", "2023-11-17",
            "2023-11-19", "2023-11-26", "2023-12-03", "2023-12-08", "2023-12-10",
            "2023-12-17", "2023-12-29",
            "2024-02-25", "2024-03-03", "2024-03-08", "2024-03-15", "2024-03-17",
            "2024-04-25", "2024-04-28", "2024-05-10", "2024-06-14", "2024-07-19",
            "2024-07-21", "2024-07-28",
            "2025-01-10", "2025-01-12", "2025-03-14", "2025-03-16", "2025-03-30",
            "2025-04-13"]
OUT = r".\20240815_card\result\figs\summary\card24_paper84_vs_released45_timeline.png"

# Table S2 descriptions: sessions whose description includes "personal use".
# S24/S28 rows were swallowed by PDF extraction (description unknown, shown as
# black ticks); every other unreleased session involves personal use except S9
# ("training data collection and other experiments").
PU = {4, 10, 11, 12, 14, 16, 17, 18, 19, 20, 21, 22, 23, 26, 27, 29, 30, 31,
      32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
      50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
      68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84}

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["axes.unicode_minus"] = False


def d(s):
    return dt.datetime.strptime(s, "%Y-%m-%d")


def main():
    paper_dates = [IMPLANT + dt.timedelta(days=v) for v in DAYS.values()]
    win_end = IMPLANT + dt.timedelta(days=251)          # S84 = 2024-03-23
    rel = [d(s) for s in RELEASED]
    in_win = [t for t in rel if t.date() <= win_end]
    post24 = [t for t in rel if dt.date(2024, 3, 24) <= t.date() <= dt.date(2024, 12, 31)]
    y25 = [t for t in rel if t.date() >= dt.date(2025, 1, 1)]
    assert len(in_win) + len(post24) + len(y25) == 45
    # every in-window release must hit a Table S2 study day
    hit = {IMPLANT + dt.timedelta(days=v) for v in DAYS.values()}  # dates already
    for t in in_win:
        assert t.date() in hit, f"unmapped released day {t}"
    print(f"paper sessions: {len(DAYS)} | released: {len(rel)} "
          f"(in-window {len(in_win)}, post-window 2024 {len(post24)}, 2025 {len(y25)}) "
          f"| in-window unreleased: {84 - len(in_win)}")

    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(15, 7.5), gridspec_kw={"height_ratios": [2.1, 1]},
        sharex=True)

    # ---- panel A: session lanes ----
    ax.axvspan(d("2023-08-11"), win_end, color="#e8f0fe", zorder=0)
    rel_sess = {s for s, v in DAYS.items()
                if (IMPLANT + dt.timedelta(days=v)).strftime("%Y-%m-%d") in RELEASED}
    n_unrel_pu = sum(1 for s in DAYS if s not in rel_sess and s in PU)
    n_unrel_pure = sum(1 for s in DAYS if s not in rel_sess and s not in PU)
    tick_colors = ["#c5c9cd" if s in rel_sess else ("#9334e6" if s in PU else "#202124")
                   for s in DAYS]
    ax.vlines(paper_dates, 1.55, 1.95, colors=tick_colors, lw=1.8, zorder=2)
    ax.scatter(in_win, [1.0] * len(in_win), marker="o", s=52, color="#1e8e3e",
               zorder=3, label=f"已发布·论文窗口内（{len(in_win)}）")
    ax.scatter(post24, [1.0] * len(post24), marker="o", s=52, color="#f9ab00",
               zorder=3, label=f"已发布·2024 窗口后（{len(post24)}）")
    ax.scatter(y25, [1.0] * len(y25), marker="o", s=52, color="#d93025",
               zorder=3, label=f"已发布·2025 追访（{len(y25)}）")
    ax.text(d("2023-09-05"), 1.75, "论文 84 个 session（Table S2 日历；刻度颜色 = 发布 × personal-use 属性）",
            fontsize=10.5, va="center")
    s9 = IMPLANT + dt.timedelta(days=DAYS[9])
    ax.annotate("S9：唯一未发布的纯采集日", xy=(s9, 1.75), xytext=(d("2023-10-05"), 2.08),
                fontsize=9, ha="left", color="#202124",
                arrowprops=dict(arrowstyle="->", color="#202124", lw=1))
    ax.text(d("2024-04-05"), 2.05, "论文未覆盖的时段（2024 下半年录没录，公开信息不可知）",
            fontsize=9, va="center", ha="left", color="#5f6368")
    for s, lab, yy in [(2, "S2 = 08.13\n锚点 WER 9.8", 0.55), (15, "S15 = 10.13\n锚点 WER 2.5%", 0.55)]:
        t = IMPLANT + dt.timedelta(days=DAYS[s])
        ax.annotate(lab, xy=(t, 1.0), xytext=(t, yy), fontsize=9, ha="center",
                    arrowprops=dict(arrowstyle="->", color="#333", lw=1))
    ax.annotate("旧错标：08.18 = S3（纯训练日，无锚点）", xy=(d("2023-08-18"), 1.0),
                xytext=(d("2023-09-10"), 0.15), fontsize=9, ha="left", color="#b31412",
                arrowprops=dict(arrowstyle="->", color="#b31412", lw=1))
    ax.axvline(win_end, color="#5f6368", ls="--", lw=1.2)
    ax.text(win_end, 2.02, "论文收数据 2024-03-23（S84）", fontsize=9, ha="right", color="#5f6368")
    ax.set_yticks([])
    ax.set_ylim(0, 2.25)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.64), fontsize=9, framealpha=0.95, handles=[
        plt.Line2D([], [], color="#c5c9cd", lw=1.8, marker="|", ms=9, linestyle="none",
                   label="已发布（上层圆点，32）"),
        plt.Line2D([], [], color="#9334e6", lw=3, marker="|", ms=9, linestyle="none",
                   label=f"未发布·含 personal use（{n_unrel_pu}）"),
        plt.Line2D([], [], color="#202124", lw=3, marker="|", ms=9, linestyle="none",
                   label=f"未发布·纯采集/描述不可提取（{n_unrel_pure}：S9+S24/S28）"),
        plt.Line2D([], [], color="#1e8e3e", marker="o", linestyle="none", ms=7,
                   label="已发布·论文窗口内（32）"),
        plt.Line2D([], [], color="#f9ab00", marker="o", linestyle="none", ms=7,
                   label="已发布·2024 窗口后（7）"),
        plt.Line2D([], [], color="#d93025", marker="o", linestyle="none", ms=7,
                   label="已发布·2025 追访（6）"),
    ])
    ax.set_title("Card 2024：论文 84 个 session vs 公开 45 个（T15，植入 2023-07-17，开台 = 植入后 25 天）",
                 fontsize=13)

    # ---- panel B: cumulative counts ----
    pd_sorted = sorted(paper_dates)  # dates already
    rel_sorted = sorted(t.date() for t in rel)
    xs, ys = [d("2023-08-11")], [0]
    for i, t in enumerate(pd_sorted, 1):
        xs += [dt.datetime.combine(t, dt.time(12)), dt.datetime.combine(t, dt.time(12))]
        ys += [ys[-1], i]
    ax2.plot(xs[1:], ys[1:], color="#9aa0a6", lw=2, label="论文累计 84")
    xs, ys = [d("2023-08-11")], [0]
    for i, t in enumerate(rel_sorted, 1):
        xs += [dt.datetime.combine(t, dt.time(12)), dt.datetime.combine(t, dt.time(12))]
        ys += [ys[-1], i]
    ax2.plot(xs[1:], ys[1:], color="#1a73e8", lw=2, label="公开累计 45")
    ax2.fill_between([d("2023-08-11"), d("2025-04-20")], 0, 90,
                     where=[True, True], color="none")
    ax2.axvspan(d("2023-08-11"), win_end, color="#e8f0fe", zorder=0)
    ax2.annotate("窗口内缺口 52 个\n（每日重训的池子缺这部分）",
                 xy=(d("2023-12-15"), 30), xytext=(d("2024-04-20"), 40),
                 fontsize=10, arrowprops=dict(arrowstyle="->", color="#333"))
    ax2.annotate("", xy=(d("2024-03-01"), 84), xytext=(d("2024-03-01"), 32),
                 arrowprops=dict(arrowstyle="<->", color="#d93025", lw=1.6))
    ax2.text(d("2024-03-05"), 58, "84 vs 32", color="#d93025", fontsize=10)
    ax2.set_ylabel("累计 session 数")
    ax2.set_ylim(0, 92)
    ax2.grid(alpha=0.25)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))

    fig.text(0.99, 0.005, "注：S24/S28 为插值行且描述行提取失败（按未知着色）、S72 已修为 228（PDF 提取坏值），均非已发布日期，不影响映射；"
             "规律：11 个纯采集日发布 10 个（仅漏 S9），71 个含 personal-use 日仅发布 22 个。证据链见 docs/20261014_calibration_investigation.md §八",
             ha="right", fontsize=8, color="#5f6368")
    fig.tight_layout()
    fig.savefig(OUT, dpi=150)

    # in-figure self-check: sample the rendered pixel at each tick's center
    # (y=1.75) and assert it matches the intended release x personal-use class
    fig.dpi = 150
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].astype(int)
    H = buf.shape[0]
    ref = {"rel": (197, 201, 205), "pu": (147, 52, 230), "pure": (32, 34, 36)}
    bad = []
    for s, t in zip(DAYS.keys(), paper_dates):
        want = "rel" if s in rel_sess else ("pu" if s in PU else "pure")
        x, y = ax.transData.transform((mdates.date2num(t), 1.75))
        r, c = int(H - y), int(x)
        if not (0 <= r < H and 0 <= c < buf.shape[1]):
            raise SystemExit(f"tick S{s} outside canvas: r={r} c={c} buf={buf.shape}")
        column = buf[max(0, r - 30):r + 30, max(0, c - 1):c + 2].reshape(-1, 3)
        dist = int(np.abs(column - np.array(ref[want])).sum(1).min())
        if dist > 20:
            bad.append((s, want, dist))
    if bad:
        raise SystemExit(f"TICK COLOR CHECK FAILED (no core pixel of intended color): {bad}")
    print(f"tick-color self-check passed: 84/84 "
          f"(rel {len(rel_sess)} / pu-unreleased {n_unrel_pu} / pure-unreleased {n_unrel_pure})")
    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
