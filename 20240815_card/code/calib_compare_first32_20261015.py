# -*- coding: utf-8 -*-
"""Paper online vs our offline, session by session, for the first 32 paper
sessions (investigation doc section 8 follow-up).

Paper Table S3 reports online PER/WER (with 95% CI) for 14 selected sessions.
S69/S80 (2024.02.25/03.17) are released as train-only days (no val split), so
the pairable set is exactly S2..S17 -- all within the first 32 sessions.

Ours: run_p4 greedy PER (val CSV) + 5-gram and +OPT WER (p4 tsvs), the same
alignment as calib_p0_wer_baseline.py. NOTE the two sides are different
protocols: paper = that day's online system on its own eval sentences; ours =
offline 45-day-trained model on the released val split of that day.

Usage: python calib_compare_first32_20261015.py
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = r".\20240815_card"
P4 = os.path.join(REPO, r"data\20241009_card_BtoT25\run_p4_20261013")
OUT_CSV = os.path.join(REPO, r"result\calib_20261015_paper_vs_ours_first32.csv")
OUT_FIG = os.path.join(REPO, r"result\figs\summary\card24_first32_paper_vs_ours.png")

sys.path.insert(0, os.path.join(REPO, "code"))
from calib_p0_wer_baseline import word_ed  # noqa: E402

# session, post-implant day, PER (lo, hi), WER (lo, hi)  -- Table S3, 125k vocab
T3 = [
    (2, 27, 15.36, 12.05, 18.87, 9.84, 4.10, 16.04),
    (4, 34, 12.82, 11.15, 14.52, 13.61, 10.28, 17.12),
    (7, 46, 9.36, 7.80, 10.98, 4.61, 2.61, 6.95),
    (8, 48, 11.74, 10.09, 13.39, 5.40, 3.52, 7.57),
    (10, 69, 12.45, 10.57, 14.40, 6.15, 3.94, 8.66),
    (11, 74, 10.17, 8.51, 11.91, 3.89, 1.92, 6.22),
    (12, 76, 10.36, 8.99, 11.77, 4.96, 3.31, 6.87),
    (13, 81, 10.58, 9.16, 12.06, 4.06, 2.67, 5.66),
    (14, 83, 10.18, 8.85, 11.57, 4.37, 2.99, 5.86),
    (15, 88, 6.93, 5.44, 8.45, 2.54, 0.99, 4.45),
    (16, 90, 7.27, 6.19, 8.41, 3.25, 2.00, 4.66),
    (17, 95, 7.76, 5.85, 9.76, 0.99, 0.00, 2.30),
]
IMPLANT = np.datetime64("2023-07-17")


def main():
    csv = pd.read_csv(os.path.join(P4, "val_greedy_per_trial.csv"))
    fin = pd.read_csv(os.path.join(P4, "p4_fixed_final_full.tsv"), sep="\t", encoding="utf-8")
    opt = pd.read_csv(os.path.join(P4, "p4_fixed_opt_fx.tsv"), sep="\t", encoding="utf-8")
    sessions = csv["session"].tolist()

    rows = []
    for s, day, per, plo, phi, wer, wlo, whi in T3:
        date = IMPLANT + np.timedelta64(day, "D")
        name = "t15." + str(date)[:10].replace("-", ".")
        g = csv[csv.session == name]
        idx = [i for i, x in enumerate(sessions) if x == name]
        e5 = r5 = eo = 0
        for i in idx:
            a, b = word_ed(fin["ref"][i], fin["hyp"][i])
            e5 += a
            r5 += b
            c, _ = word_ed(opt["ref"][i], opt["hyp"][i])
            eo += c
        our_per = 100 * g.edit_distance.sum() / g.true_len.sum()
        rows.append({"S": s, "date": str(date)[:10], "session": name,
                     "paper_PER": per, "paper_PER_lo": plo, "paper_PER_hi": phi,
                     "our_PER": round(our_per, 2),
                     "paper_WER": wer, "paper_WER_lo": wlo, "paper_WER_hi": whi,
                     "our_WER_5gram": round(100 * e5 / r5, 2),
                     "our_WER_OPT": round(100 * eo / r5, 2),
                     "n_val_trials": len(g)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    print(df.to_string(index=False))
    print(f"\nsaved: {OUT_CSV}")
    print("\nmedians: paper PER %.2f vs our PER %.2f | paper WER %.2f vs our 5-gram %.2f / +OPT %.2f"
          % (df.paper_PER.median(), df.our_PER.median(),
             df.paper_WER.median(), df.our_WER_5gram.median(), df.our_WER_OPT.median()))
    per_win = int((df.our_PER < df.paper_PER).sum())
    wer_win = int((df.our_WER_OPT < df.paper_WER).sum())
    print("sessions where ours beats paper: PER %d/12, +OPT WER %d/12" % (per_win, wer_win))

    matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    matplotlib.rcParams["axes.unicode_minus"] = False
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    x = df.S.values

    a1.errorbar(x, df.paper_PER, yerr=[df.paper_PER - df.paper_PER_lo,
                                       df.paper_PER_hi - df.paper_PER],
                fmt="o-", color="#1a73e8", lw=1.8, capsize=3, label="论文在线系统（当日校准后）")
    a1.plot(x, df.our_PER, "s--", color="#d93025", lw=1.8, label="我们离线（run_p4 greedy）")
    a1.set_ylabel("raw PER (%)")
    a1.set_title("前 32 个 session 里的可比对子集：论文在线 vs 我们离线（Table S3 的 S2–S17，12/12 已发布）", fontsize=12)
    a1.grid(alpha=0.25)
    a1.legend(fontsize=10)

    a2.errorbar(x, df.paper_WER, yerr=[df.paper_WER - df.paper_WER_lo,
                                       df.paper_WER_hi - df.paper_WER],
                fmt="o-", color="#1a73e8", lw=1.8, capsize=3, label="论文在线系统")
    a2.plot(x, df.our_WER_5gram, "^:", color="#9aa0a6", lw=1.6, label="我们 5-gram 解码")
    a2.plot(x, df.our_WER_OPT, "s--", color="#d93025", lw=1.8, label="我们 +OPT 重打分")
    a2.set_ylabel("WER (%)")
    a2.set_xlabel("论文 session 编号（Table S3 评测日）")
    a2.set_xticks(x)
    a2.grid(alpha=0.25)
    a2.legend(fontsize=10)
    for ax in (a1, a2):
        for s, lab in [(2, "S2"), (15, "S15（2.5%）"), (17, "S17（0.99%）")]:
            ax.annotate(lab, xy=(s, ax.get_ylim()[0]), fontsize=9, ha="center",
                        xytext=(0, 4), textcoords="offset points", color="#5f6368")
    fig.text(0.99, 0.005, "口径差异：论文=当日在线评测句（当日校准后）；我们=45 天混训离线模型 + 发布 val 切分。"
             "S69/S80 也有论文数字但发布为 train-only 天，无 val 不可比。", ha="right", fontsize=8, color="#5f6368")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    print(f"saved: {OUT_FIG}")


if __name__ == "__main__":
    main()
