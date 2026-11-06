# -*- coding: utf-8 -*-
"""Kunz 2025 复现差距分解（无重训路线，2026-10-28）。

问题：论文在线 WER 26-54%（T15/T16，Switchboard 开放句、生产系统、闭环）vs
我方离线最好 WER 78-94%（Table 4 口径）。用仓内现成产物把差距沿三轴拆开：
  A 词表/条件轴 —— 我方逐会话 WER（早期 50words 会话 vs 后期 125k 开放会话）；
  B LM 档轴 —— 同一 logits 下 3gram / TLG5gram / +OPT / +Qwen 的池化 WER；
  C 训练规模轴 —— finetune 链各阶段的评测（若日志可得）。
剩余（在线闭环效应 + 数据规模差）作为残差项讨论。

产出：stdout 汇总表 + result/20261028_gap_decomposition.tsv
"""
import csv
import os
import re

R = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
K = os.path.join(R, "data", "20250821_kunz")
OUT = os.path.join(R, "result", "20261028_gap_decomposition.tsv")


def rows(path, delim=","):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f, delimiter=delim))


def wer_pct(ed_list):
    e = sum(int(x[0]) for x in ed_list)
    n = sum(int(x[1]) for x in ed_list)
    return 100.0 * e / max(n, 1), len(ed_list)


def ed(hyp, ref):
    r, h = ref.split(), hyp.split()

    def dist(a, b):
        d = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(len(a) + 1):
            d[i][0] = i
        for j in range(len(b) + 1):
            d[0][j] = j
        for i in range(1, len(a) + 1):
            for j in range(1, len(b) + 1):
                d[i][j] = min(d[i - 1][j - 1] + (a[i - 1] != b[j - 1]),
                              d[i - 1][j] + 1, d[i][j - 1] + 1)
        return d[-1][-1]
    return dist(r, h), len(r)


def analyze(part):
    rd = os.path.join(K, f"run_{part}_mixed")
    per = rows(os.path.join(rd, "val_greedy_per_trial.csv"))
    hdr, body = per[0], per[1:]
    sess = [r[0] for r in body]
    out = {"part": part, "n": len(body)}

    # A: 逐会话 greedy PER
    agg = {}
    for r in body:
        s = re.sub(r"_\d+words?_eval$|_eval$", "", r[0])
        agg.setdefault(s, [0, 0, 0])
        agg[s][0] += int(r[4])
        agg[s][1] += int(r[5])
        agg[s][2] += 1
    out["per_session_per"] = {
        s: (round(100 * v[0] / max(v[1], 1), 1), v[2]) for s, v in agg.items()}

    # B: LM 档池化 WER（key 行序 == per-trial 行序）
    key2sess = {f"{i:04d}": sess[i] for i in range(len(sess))}
    tiers = {}

    t = [r for r in rows(os.path.join(rd, "word_decode_3gram.csv"))[1:]
         if len(r) >= 6]
    tiers["3gram"] = [(int(r[4]), int(r[5]), r[0]) for r in t]

    t = rows(os.path.join(rd, "word_decode_tlg5gram.tsv.wer"))
    if t and t[0][0].lower() == "key":
        t = t[1:]
    tiers["tlg5gram"] = [(int(r[1]), int(r[2]), key2sess.get(r[0], "?"))
                         for r in t if len(r) >= 3]

    t = rows(os.path.join(rd, "off_final_full.tsv"), "\t")
    if t and t[0][0].lower() == "key":
        t = t[1:]
    tiers["final_full"] = [ed(r[2], r[1]) + (key2sess.get(r[0], "?"),) for r in t]

    for name in ("off_bp1_opt", "off_bp1_opt_qwen"):
        p = os.path.join(rd, name + ".tsv")
        if os.path.exists(p):
            t = rows(p, "\t")
            if t and t[0][0].lower() == "key":
                t = t[1:]
            tiers[name] = [(int(r[4]), int(r[5]), key2sess.get(r[0], "?"))
                           for r in t]

    out["tiers_full"] = {}
    out["tiers_bp1"] = {}
    for name, recs in tiers.items():
        w, n = wer_pct(recs)
        out["tiers_full"][name] = (round(w, 1), n)
        # bp1 子集 = off_bp1_* 文件覆盖的 key 集合；对其他档先按同 key 池化
        bp_keys = set()
        pb = os.path.join(rd, "off_bp1_opt.tsv")
        if os.path.exists(pb):
            tb = rows(pb, "\t")
            if tb and tb[0][0].lower() == "key":
                tb = tb[1:]
            bp_keys = {r[0] for r in tb}
        # bp1 池化（仅对带 key 的档直接可做）
        if name in ("tlg5gram", "final_full", "off_bp1_opt", "off_bp1_opt_qwen"):
            sub = [(e, l) for (e, l, _s) in recs
                   if name.startswith("off_bp1") or True]
            # off_bp1_* 行本身就是子集；其他档取同 key 子集
            if name.startswith("off_bp1"):
                w2, n2 = wer_pct(recs)
            else:
                sub = []
                if name == "tlg5gram":
                    src = rows(os.path.join(rd,
                                            "word_decode_tlg5gram.tsv.wer"))
                    if src and src[0][0].lower() == "key":
                        src = src[1:]
                    for r in src:
                        if len(r) >= 3 and r[0] in bp_keys:
                            sub.append((int(r[1]), int(r[2])))
                else:
                    src = rows(os.path.join(rd, "off_final_full.tsv"), "\t")
                    if src and src[0][0].lower() == "key":
                        src = src[1:]
                    for r in src:
                        if r[0] in bp_keys:
                            e, l = ed(r[2], r[1])
                            sub.append((e, l))
                w2, n2 = wer_pct(sub)
            out["tiers_bp1"][name] = (round(w2, 1), n2)

    # A': 逐会话 WER（final_full 档）
    per_sess = {}
    src = rows(os.path.join(rd, "off_final_full.tsv"), "\t")
    if src and src[0][0].lower() == "key":
        src = src[1:]
    for r in src:
        s = re.sub(r"_\d+words?_eval$|_eval$", "", key2sess.get(r[0], "?"))
        e, l = ed(r[2], r[1])
        per_sess.setdefault(s, [0, 0, 0])
        per_sess[s][0] += e
        per_sess[s][1] += l
        per_sess[s][2] += 1
    out["per_session_wer_final"] = {
        s: (round(100 * v[0] / max(v[1], 1), 1), v[2]) for s, v in per_sess.items()}
    return out


def main():
    lines = []
    lines.append("part\tn\tper_session_PER\tper_session_WER(final)"
                 "\tWER_3gram\tWER_5gram\tWER_final\tWER_bp1+OPT\tWER_bp1+OPT+Qwen"
                 "\tbp1_WER_5gram\tbp1_WER_final")
    for part in ("t12", "t15", "t16"):
        o = analyze(part)
        tf, tb = o["tiers_full"], o["tiers_bp1"]
        row = [
            o["part"], str(o["n"]),
            "; ".join(f"{k}={v[0]}%(n={v[1]})" for k, v in
                      o["per_session_per"].items()),
            "; ".join(f"{k}={v[0]}%(n={v[1]})" for k, v in
                      o["per_session_wer_final"].items()),
            f"{tf['3gram'][0]}", f"{tf['tlg5gram'][0]}", f"{tf['final_full'][0]}",
            f"{tb.get('off_bp1_opt', ('-',))[0]}",
            f"{tb.get('off_bp1_opt_qwen', ('-',))[0]}",
            f"{tb.get('tlg5gram', ('-',))[0]}", f"{tb.get('final_full', ('-',))[0]}",
        ]
        lines.append("\t".join(row))
        print("\n==", part, "==")
        print("  PER/session:", o["per_session_per"])
        print("  WER/session:", o["per_session_wer_final"])
        print("  WER tiers full:", tf)
        print("  WER tiers bp1 :", tb)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
