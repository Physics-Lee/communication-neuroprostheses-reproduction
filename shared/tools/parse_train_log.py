# -*- coding: utf-8 -*-
"""Post-hoc parser for the nejm-brain-to-text trainer log.

Extracts train loss / grad-norm and val PER curves (+ per-day PER at the
latest val step) from training_log, writes:
- curves.csv        (batch, split, metric, value)
- training_curves.png
Safe to re-run while training is in progress (parses whatever exists).

Usage: parse_train_log.py <log_path> <out_dir>
"""
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

RE_TRAIN = re.compile(r"Train batch (\d+): loss: ([\d.]+) grad norm: ([\d.]+) time: ([\d.]+)")
RE_VAL = re.compile(r"Val batch (\d+): PER \(avg\): ([\d.]+) CTC Loss \(avg\): ([\d.]+) time: ([\d.]+)")
RE_DAY = re.compile(r"(t12\.\d{4}\.\d{2}\.\d{2}) val PER: ([\d.]+)")
RE_BEST = re.compile(r"New best test PER ([\d.]+) --> ([\d.]+)")


def main():
    log_path, out_dir = sys.argv[1], sys.argv[2]

    train_rows, val_rows, best_rows = [], [], []
    day_per_at_batch = defaultdict(dict)  # batch -> {day: per}
    cur_batch = None
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = RE_VAL.match(line.split(": ", 1)[-1].strip()) if ": " in line else None
            m = RE_VAL.search(line)
            if m:
                cur_batch = int(m.group(1))
                val_rows.append((cur_batch, float(m.group(2)), float(m.group(3))))
                continue
            m = RE_TRAIN.search(line)
            if m:
                train_rows.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
                continue
            m = RE_BEST.search(line)
            if m:
                best_rows.append((float(m.group(1)), float(m.group(2))))
                continue
            m = RE_DAY.search(line)
            if m and cur_batch is not None:
                day_per_at_batch[cur_batch][m.group(1)] = float(m.group(2))

    tr = pd.DataFrame(train_rows, columns=["batch", "train_loss", "grad_norm"])
    va = pd.DataFrame(val_rows, columns=["batch", "val_PER", "val_ctc_loss"])
    print(f"parsed: {len(tr)} train points, {len(va)} val points, "
          f"{len(day_per_at_batch)} day-level val steps")

    # curves.csv
    curves = pd.concat([
        tr.melt(id_vars="batch", value_vars=["train_loss", "grad_norm"],
                var_name="metric", value_name="value").assign(split="train"),
        va.melt(id_vars="batch", value_vars=["val_PER", "val_ctc_loss"],
                var_name="metric", value_name="value").assign(split="val"),
    ])
    curves.to_csv(f"{out_dir}/curves.csv", index=False)

    # per-day PER at the latest val step
    if day_per_at_batch:
        last_batch = max(day_per_at_batch)
        day_df = (pd.DataFrame.from_dict(day_per_at_batch[last_batch], orient="index",
                                         columns=["val_PER"])
                  .sort_values("val_PER"))
        day_df.index.name = "session"
        day_df["val_PER_pct"] = (100 * day_df["val_PER"]).round(2)
        day_df.to_csv(f"{out_dir}/day_per_last_valstep.csv")

    # summary print
    if len(va):
        b = va.loc[va["val_PER"].idxmin()]
        print(f"best val PER so far: {100*b.val_PER:.2f}% @ batch {int(b.batch)}")
        print("val PER trajectory:")
        for _, r in va.iterrows():
            print(f"  batch {int(r.batch):>6d}: PER {100*r.val_PER:6.2f}%  "
                  f"CTC loss {r.val_ctc_loss:7.2f}")
    if day_per_at_batch:
        print(f"\nper-day val PER at batch {last_batch} "
              f"(best 5 / worst 5):")
        print(day_df.head(5).to_string())
        print("...")
        print(day_df.tail(5).to_string())

    # plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    axes[0].plot(tr["batch"], tr["train_loss"], lw=0.8, color="#2457c5")
    axes[0].set_title("train CTC loss"); axes[0].set_xlabel("batch")
    axes[1].plot(va["batch"], 100 * va["val_PER"], marker="o", ms=4, color="#17683c")
    if len(va):
        axes[1].axhline(19.7, ls="--", c="#a83232", lw=1)
        axes[1].text(va["batch"].iloc[0], 19.9, "Willett 2023 paper raw PER 19.7%",
                     fontsize=8, color="#a83232")
        axes[1].scatter([b.batch], [100 * b.val_PER], marker="*", s=120,
                        color="#9a5b00", zorder=5, label=f"best {100*b.val_PER:.2f}%")
        axes[1].legend(fontsize=8)
    axes[1].set_title("validation greedy PER (%)"); axes[1].set_xlabel("batch")
    axes[2].plot(va["batch"], va["val_ctc_loss"], marker="o", ms=4, color="#5c667a")
    axes[2].set_title("validation CTC loss"); axes[2].set_xlabel("batch")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("BtoT24 GRU training (run_60k) — parsed from training_log", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/training_curves.png", dpi=150, bbox_inches="tight")
    print(f"\nwrote: {out_dir}/curves.csv, {out_dir}/training_curves.png"
          + (f", {out_dir}/day_per_last_valstep.csv" if day_per_at_batch else ""))


if __name__ == "__main__":
    main()
