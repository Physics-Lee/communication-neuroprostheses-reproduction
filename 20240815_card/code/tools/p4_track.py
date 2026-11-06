# -*- coding: utf-8 -*-
"""P4 overnight tracker: compare the new from-scratch run against run_120k.

Parses both trainer logs (val PER trajectory + train loss + batch rates),
prints a side-by-side table at matching optimizer steps, estimates the new
run's ETA from its own measured rate, and writes an overlay figure.

Safe to re-run at any time during training (parses whatever exists).

Usage: python p4_track.py
"""
import re
import time
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = r".\20240815_card"
OLD_LOG = REPO + r"\result\logs\train_btot25_120k.log"
NEW_LOG = REPO + r"\result\logs\train_p4_20261013.log"
OUT_PNG = REPO + r"\result\figs\summary\p4_trajectory_vs_run120k.png"
OUT_CSV = REPO + r"\result\p4_vs_run120k_trajectory.csv"

RE_VAL = re.compile(r"Val batch (\d+): PER \(avg\): ([\d.]+) CTC Loss \(avg\): ([\d.]+)")
RE_TRAIN = re.compile(r"Train batch (\d+): loss: ([\d.]+) grad norm: ([\d.]+) time: ([\d.]+)")
RE_TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")


def parse(path):
    val, tr, last_ts, start_ts = [], [], None, None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = RE_TS.match(line)
        if m:
            t = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
            last_ts = t
            if start_ts is None:
                start_ts = t
        m = RE_VAL.search(line)
        if m:
            val.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
            continue
        m = RE_TRAIN.search(line)
        if m:
            tr.append((int(m.group(1)), float(m.group(2)), float(m.group(4))))
    return val, tr, start_ts, last_ts


def main():
    ov, otr, o_start, o_end = parse(OLD_LOG)
    nv, ntr, n_start, n_end = parse(NEW_LOG)
    print(f"run_120k : {len(ov)} val points, {len(otr)} train log lines "
          f"({o_start} -> {o_end}, 1263.1 min total)")
    if not nv:
        print("new run: nothing parsed yet")
        return
    print(f"p4 run   : {len(nv)} val points, {len(ntr)} train log lines "
          f"(started {n_start})")

    # ETA from the new run's own train-batch timestamps (first vs last)
    if len(ntr) >= 2 and n_start and n_end:
        done_micro = ntr[-1][0]
        rate = (n_end - n_start).total_seconds() / max(1, done_micro)
        eta_min = (240000 - done_micro) * rate / 60
        eta = n_end.timestamp() + eta_min * 60 if hasattr(n_end, "timestamp") else None
        print(f"rate: {rate:.3f} s/micro-batch over {done_micro} micro-batches "
              f"-> ETA {eta_min/60:.1f} h more"
              f" (done ~{datetime.fromtimestamp(n_end.timestamp() + eta_min*60):%F %T})"
              if rate else "")
        # steady-state rate: use only the last 5000 logged batches
        if len(ntr) > 5200:
            (b1, _, _), (b2, _, _) = ntr[-5000], ntr[-1]
            print(f"recent rate (last 5000 log points): ignored (log lines are "
                  f"sampled); using global rate above")

    # side-by-side at matching val steps
    o_by_step = {b // 2: (p, l) for b, p, l in ov}
    n_by_step = {b // 2: (p, l) for b, p, l in nv}
    steps = sorted(set(o_by_step) & set(n_by_step))
    print(f"\n{'opt_step':>8} {'run120k PER':>12} {'p4 PER':>10} {'diff':>8}")
    rows = []
    for s in steps[-25:]:
        d = 100 * (n_by_step[s][0] - o_by_step[s][0])
        rows.append((s, 100 * o_by_step[s][0], 100 * n_by_step[s][0], d))
        print(f"{s:>8} {100*o_by_step[s][0]:>11.2f}% {100*n_by_step[s][0]:>9.2f}% {d:>+7.2f}")
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        f.write("opt_step,run120k_val_PER_pct,p4_val_PER_pct,diff_pct\n")
        for s in steps:
            f.write(f"{s},{100*o_by_step[s][0]:.4f},{100*n_by_step[s][0]:.4f},"
                    f"{100*(n_by_step[s][0]-o_by_step[s][0]):.4f}\n")

    # overlay plot
    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.plot([b / 2 for b, _, _ in ov], [100 * p for _, p, _ in ov], "-o", ms=3,
            color="#9aa5b5", label=f"run_120k (bcirun, 1263 min)")
    ax.plot([b / 2 for b, _, _ in nv], [100 * p for _, p, _ in nv], "-o", ms=3,
            color="#2457c5", label="p4 retrain (2026-10-13)")
    ax.set_xlabel("optimizer step"); ax.set_ylabel("val greedy PER (%)")
    ax.set_title("P4 from-scratch retrain vs run_120k: val PER trajectory")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PNG, dpi=150)
    print(f"\nwrote {OUT_CSV}\nwrote {OUT_PNG}")


if __name__ == "__main__":
    main()
