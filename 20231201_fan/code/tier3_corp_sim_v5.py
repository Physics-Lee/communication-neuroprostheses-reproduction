# -*- coding: utf-8 -*-
"""Fan 2023 (CORP) tier-3 v5: official pseudo-labels + CORP-scale update
budget.

v4b (48 steps/session at lr 2e-5) left every session's input layer
half-converged, and the polluted layer state compounded across sessions
(late sessions started at 160-180% CER). The train-budget diagnostic
(diag_train_budget.py) showed 300-500 steps at lr 5e-5 on session 3
monotonically pulls every probe trial down (170->75, 129->60 ...). v5
therefore spends 500 steps per session (70% official pseudo-label trials,
30% seed replay), then hands the converged layer to the next session.
"""
import json
import os
import sys

import numpy as np

NSD_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../open_source_code_for_reference/20231201_fan_CORP/NeuralDecoder"))
sys.path.insert(0, NSD_ROOT)

import tensorflow as tf  # noqa: E402
from tier3_corp_sim_official_labels import (  # noqa: E402
    make_args, gather_trials, load_official_labels, gauss_smooth_np,
    forward_logits, greedy_decode, editdist, ctc_loss, EVAL_SESSIONS)
from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder  # noqa: E402

OUT = ("./20231201_fan/result/"
       "tier3_corp_sim_v5.json")
N_PRE = 6
N_POST = 6
STEPS = 500
LR = 5e-5
REPLAY_FRAC = 0.3


def cer(nsd, li, feat, label):
    dec = greedy_decode(forward_logits(nsd, li, feat))
    ed, ref = editdist(dec, [int(x) for x in label if x > 0])
    return 100 * ed / max(ref, 1)


def main():
    args = make_args()
    nsd = NeuralSequenceDecoder(args=args)
    seed_bank = []
    for si in range(0, 11, 2):
        f, l, _ = gather_trials(nsd, si)
        for k in range(0, len(f), 4):
            seed_bank.append((f[k], [int(x) for x in l[k] if x > 0]))
    print(f"seed bank {len(seed_bank)}", flush=True)
    report = {"sessions": [], "protocol": {
        "pseudo": "official lm_decoded labels", "steps": STEPS, "lr": LR,
        "replay_frac": REPLAY_FRAC}}
    incoming = 10
    for si, sess in enumerate(EVAL_SESSIONS):
        li = 11 + si
        feats, labels, sents = gather_trials(nsd, 11 + si)
        official = load_official_labels(sess)
        usable = [(feats[k], official[sents[k]]) for k in range(len(feats))
                  if sents[k] in official and len(official[sents[k]]) > 0]
        n = len(feats)
        pre = float(np.mean([cer(nsd, incoming, feats[k], labels[k])
                             for k in range(min(N_PRE, n))]))
        layer = nsd.inputLayers[li]
        opt = tf.keras.optimizers.Adam(LR)
        mid = None
        for step in range(STEPS):
            if np.random.rand() < REPLAY_FRAC and seed_bank:
                xf, lab = seed_bank[np.random.randint(len(seed_bank))]
            else:
                xf, lab = usable[np.random.randint(len(usable))]
            x = gauss_smooth_np(xf)
            with tf.GradientTape() as tape:
                xin = tf.constant(x[np.newaxis], dtype=tf.float32)
                y = layer(xin, training=True)
                logits = nsd.model(y, training=False)
                loss = ctc_loss(lab, logits[0],
                                tf.cast(tf.shape(logits)[1], tf.int32))
            grads = tape.gradient(loss, layer.trainable_variables)
            if grads and all(g is not None for g in grads):
                opt.apply_gradients(zip(grads, layer.trainable_variables))
            if step == STEPS // 2:
                mid = float(np.mean([cer(nsd, li, feats[k], labels[k])
                                     for k in range(min(N_PRE, n))]))
        post = float(np.mean([cer(nsd, li, feats[k], labels[k])
                              for k in range(max(0, n - N_POST), n)]))
        report["sessions"].append({
            "session": sess, "n": n, "usable": len(usable),
            "pre": round(pre, 1), "mid": round(mid, 1) if mid else None,
            "post": round(post, 1)})
        print(f"{sess}: n={n} usable={len(usable)} pre={pre:.1f} "
              f"mid={mid if mid else -1:.1f} post={post:.1f}", flush=True)
        incoming = li
    pres = [s["pre"] for s in report["sessions"]]
    posts = [s["post"] for s in report["sessions"]]
    report["summary"] = {
        "mean_pre": round(float(np.mean(pres)), 1),
        "mean_post": round(float(np.mean(posts)), 1),
        "static_ref": 50.85,
        "v4b_ref": {"mean_pre": 91.3, "mean_post": 186.8}}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("SUMMARY:", report["summary"])


if __name__ == "__main__":
    main()
