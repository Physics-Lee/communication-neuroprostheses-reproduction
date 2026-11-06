# -*- coding: utf-8 -*-
"""Fan 2023 (CORP) tier-3 v7: v6 protocol with a leakage-free evaluation.

v6 (full decoder, batch recalibration) reached mean CER 26.9 -> 6.2%, but
its post measure covers the session's final 6 trials which were part of
the training pool (same as v5). v7 keeps everything else identical
(official lm_decoded pseudo-labels, full decoder = session input layer +
shared backend, 500 steps/session, lr 5e-5, global-norm clip 10, 70/30
pseudo/replay) and only changes the split: the final 6 trials of every
session NEVER enter the training pool; post = CER on those held-out
trials. Also adds per-session del+gc against the ~2 GB/session memory
climb seen in v6.
"""
import gc
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
    gather_trials, load_official_labels, gauss_smooth_np,
    forward_logits, ctc_loss, EVAL_SESSIONS)
from tier3_corp_sim_v5 import cer  # noqa: E402
from tier3_corp_sim_v6 import make_args  # noqa: E402
from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder  # noqa: E402

OUT = ("./20231201_fan/result/"
       "tier3_corp_sim_v7.json")
STEPS = 500
LR = 5e-5
REPLAY_FRAC = 0.3
N_PRE = 6
N_HOLD = 6
GRAD_CLIP = 10.0


def main():
    args = make_args()
    nsd = NeuralSequenceDecoder(args=args)
    seed_bank = []
    for si in range(0, 11, 2):
        f, l, _ = gather_trials(nsd, si)
        for k in range(0, len(f), 4):
            seed_bank.append((f[k], [int(x) for x in l[k] if x > 0]))
    print(f"seed bank {len(seed_bank)}", flush=True)
    _ = forward_logits(nsd, 10, seed_bank[0][0])  # warm-up: build backend
    backend_vars = list(nsd.model.trainable_variables)
    print(f"backend vars: {len(backend_vars)}", flush=True)
    assert len(backend_vars) > 0
    report = {"sessions": [], "protocol": {
        "pseudo": "official lm_decoded labels",
        "scope": "FULL decoder (per-session input layer + shared backend)",
        "eval": "final 6 trials per session NEVER trained on (leakage-free)",
        "steps": STEPS, "lr": LR, "replay_frac": REPLAY_FRAC,
        "grad_global_norm_clip": GRAD_CLIP,
        "backend_vars": len(backend_vars)}}
    incoming = 10
    for si, sess in enumerate(EVAL_SESSIONS):
        li = 11 + si
        feats, labels, sents = gather_trials(nsd, 11 + si)
        official = load_official_labels(sess)
        n = len(feats)
        hold = list(range(max(0, n - N_HOLD), n))
        pool = [(feats[k], official[sents[k]]) for k in range(n - N_HOLD)
                if sents[k] in official and len(official[sents[k]]) > 0]
        assert len(pool) > 0, f"{sess}: empty training pool"
        pre = float(np.mean([cer(nsd, incoming, feats[k], labels[k])
                             for k in range(min(N_PRE, n))]))
        layer = nsd.inputLayers[li]
        trainable = list(layer.trainable_variables) + backend_vars
        opt = tf.keras.optimizers.Adam(LR)
        mid = None
        for step in range(STEPS):
            if np.random.rand() < REPLAY_FRAC and seed_bank:
                xf, lab = seed_bank[np.random.randint(len(seed_bank))]
            else:
                xf, lab = pool[np.random.randint(len(pool))]
            x = gauss_smooth_np(xf)
            with tf.GradientTape() as tape:
                xin = tf.constant(x[np.newaxis], dtype=tf.float32)
                y = layer(xin, training=True)
                logits = nsd.model(y, training=True)
                loss = ctc_loss(lab, logits[0],
                                tf.cast(tf.shape(logits)[1], tf.int32))
            if np.isnan(float(loss.numpy())):
                print(f"{sess}: NaN loss at step {step}, aborting session",
                      flush=True)
                break
            grads = tape.gradient(loss, trainable)
            if grads and all(g is not None for g in grads):
                grads, _ = tf.clip_by_global_norm(grads, GRAD_CLIP)
                opt.apply_gradients(zip(grads, trainable))
            if step == STEPS // 2:
                mid = float(np.mean([cer(nsd, li, feats[k], labels[k])
                                     for k in hold]))
        post = float(np.mean([cer(nsd, li, feats[k], labels[k])
                              for k in hold]))
        report["sessions"].append({
            "session": sess, "n": n, "pool": len(pool),
            "pre": round(pre, 1), "mid": round(mid, 1) if mid else None,
            "post": round(post, 1)})
        print(f"{sess}: n={n} pool={len(pool)} pre={pre:.1f} "
              f"mid={mid if mid else -1:.1f} post={post:.1f}", flush=True)
        incoming = li
        del feats, labels, sents, pool, hold
        gc.collect()
    pres = [s["pre"] for s in report["sessions"]]
    posts = [s["post"] for s in report["sessions"]]
    report["summary"] = {
        "mean_pre": round(float(np.mean(pres)), 1),
        "mean_post": round(float(np.mean(posts)), 1),
        "v6_ref": {"mean_pre": 26.9, "mean_post": 6.2, "eval": "batch"},
        "static_ref": 50.85}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("SUMMARY:", report["summary"])


if __name__ == "__main__":
    main()
