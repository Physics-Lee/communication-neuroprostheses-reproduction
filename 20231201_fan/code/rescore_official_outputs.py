# -*- coding: utf-8 -*-
"""Fan 2023 (CORP) tier-1 reproduction: rescore the OFFICIAL online decoded
outputs shipped in the Dryad release (online_evaluation_data/{recalibration,
no_recalibration}/mat/*.mat: sentences = ground truth, rnn_decoded = raw RNN
beam, lm_decoded = post-LM output), independent of any model execution.

Anchors (paper): Table 2 online WER 6.16 (CORP / recalibration arm) vs 26.51
(no recalibration); abstract "average accuracy 93.84% +/- 2.28 (mean +/- SD)".

WER normalization: '>' word separators -> spaces, casefold, strip
punctuation (same convention as the survey repo's make_fig8_data.py).

Writes result/rescore_official_outputs.json + prints a summary table.
"""
import io
import json
import re
import zipfile

import numpy as np
import scipy.io as sio

ZIP = ("./20231201_fan/data/raw/"
       "20231201_fan/CORP_data_release.zip")
OUT = ("./20231201_fan/result/"
       "rescore_official_outputs.json")
TOK = re.compile(r"[^a-z0-9']+")


def words_of(s):
    out = []
    for w in str(s).replace(">", " ").casefold().split():
        w = TOK.sub("", w)
        if w:
            out.append(w)
    return out


def wer(ref, hyp):
    a, b = words_of(ref), words_of(hyp)
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n], m


def main():
    z = zipfile.ZipFile(ZIP)
    report = {"arms": {}}
    for arm in ["recalibration", "no_recalibration"]:
        mats = sorted(x for x in z.namelist()
                      if x.endswith(".mat") and f"/{arm}/" in x)
        per_session = []
        tot_e = tot_r = 0
        for name in mats:
            m = sio.loadmat(io.BytesIO(z.read(name)), squeeze_me=True,
                            struct_as_record=False)
            gt = np.atleast_1d(m["sentences"])
            rnn = np.atleast_1d(m["rnn_decoded"])
            lm = np.atleast_1d(m["lm_decoded"])
            assert len(gt) == len(rnn) == len(lm)
            e_rnn = e_lm = n_ref = exact = 0
            for g, r, l in zip(gt, rnn, lm):
                er, nr = wer(g, r)
                el, _ = wer(g, l)
                e_rnn += er
                e_lm += el
                n_ref += nr
                exact += int(words_of(g) == words_of(l))
            per_session.append({
                "session": name.split("/")[-1].replace(".mat", ""),
                "n_sentences": len(gt),
                "n_ref_words": n_ref,
                "wer_rnn_pct": round(100 * e_rnn / n_ref, 2),
                "wer_lm_pct": round(100 * e_lm / n_ref, 2),
                "sentence_exact_match_pct": round(100 * exact / len(gt), 2),
                "session_accuracy_pct": round(100 - 100 * e_lm / n_ref, 2),
            })
            tot_e += e_lm
            tot_r += n_ref
        accs = [s["session_accuracy_pct"] for s in per_session]
        report["arms"][arm] = {
            "sessions": len(per_session),
            "pooled_wer_lm_pct": round(100 * tot_e / tot_r, 2),
            "mean_session_wer_pct": round(float(np.mean(
                [s["wer_lm_pct"] for s in per_session])), 2),
            "mean_session_accuracy_pct": round(float(np.mean(accs)), 2),
            "sd_session_accuracy_pct": round(float(np.std(accs, ddof=1)), 2),
            "mean_session_wer_rnn_pct": round(float(np.mean(
                [s["wer_rnn_pct"] for s in per_session])), 2),
            "per_session": per_session,
        }
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    for arm, d in report["arms"].items():
        print(f"== {arm}: {d['sessions']} sessions | pooled LM WER "
              f"{d['pooled_wer_lm_pct']}% | mean session WER "
              f"{d['mean_session_wer_pct']}% | acc {d['mean_session_accuracy_pct']}"
              f" +/- {d['sd_session_accuracy_pct']}% | raw RNN mean WER "
              f"{d['mean_session_wer_rnn_pct']}%")
        for s in d["per_session"]:
            print(f"   {s['session']}  n={s['n_sentences']:3d}  "
                  f"LM WER {s['wer_lm_pct']:6.2f}%  "
                  f"raw WER {s['wer_rnn_pct']:6.2f}%  "
                  f"exact {s['sentence_exact_match_pct']:5.1f}%")


if __name__ == "__main__":
    main()
