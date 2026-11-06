# -*- coding: utf-8 -*-
"""Fan 2023 (CORP) tier-3 v6: v5 protocol with the FULL decoder
trainable (backend unfrozen), matching the paper's online update scope.
v5 (input layer only) reached mean CER 51.4 -> 40.2%; the paper's online
system updates the whole decoder trial-by-trial (raw rescoring ~22%).

v6 take-2 (2026-10-27): take-1 was killed 45 min in -- its log printed
"trainable vars: 11", i.e. the backend contributed ZERO variables. Two
bugs: (1) the trainable list was snapshotted before any forward pass, so
the lazily-built Keras backend had no weights yet; (2) the snapshot pinned
session-1's input layer (inputLayers[11]) forever, so sessions 2-10 would
have trained nothing at all (grads all None). Fixed here: warm up one
forward_logits call, then snapshot backend vars once; rebuild the
trainable list per session (session input layer + backend); gradients
clipped to global norm 10 (the official trainer's gradClipValue) so
full-decoder CTC steps cannot blow up in one step.
Everything else identical to v5: official lm_decoded pseudo-labels,
STEPS=500/session, lr 5e-5, 70/30 pseudo/replay split, per-session Adam.
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
    gather_trials, load_official_labels, gauss_smooth_np,
    forward_logits, greedy_decode, editdist, ctc_loss, EVAL_SESSIONS)
from tier3_corp_sim_v5 import cer  # noqa: E402
from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

DATA = ("./20231201_fan/data/20231201_fan/"
        "CORP_data_release/")
OUT = ("./20231201_fan/result/"
       "tier3_corp_sim_v6.json")
STEPS = 500
LR = 5e-5
REPLAY_FRAC = 0.3
N_PRE = 6
N_POST = 6
GRAD_CLIP = 10.0


def make_args():
    from tier3_corp_sim_official_labels import (  # noqa: F401
        SEED_SESSIONS, MODEL_YAML)
    n_ev = len(EVAL_SESSIONS)
    return OmegaConf.create({
        "mode": "inference", "gpuNumber": "0",
        "outputDir": "./20231201_fan/tmp_run6",
        "loadDir": DATA + "pretrained_seed_model",
        "loadCheckpointIdx": 12500,
        "smoothInputs": 1, "smoothKernelSD": 2,
        "learnRateStart": 0.01, "learnRateEnd": 0.0,
        "learnRateDecaySteps": 100000, "learnRatePower": 1.0,
        "trainableInput": 1,
        "trainableBackend": 1,  # v6: full decoder trains, like the paper
        "seed": -1, "batchesPerSave": 0, "batchesPerVal": 50,
        "nBatchesToTrain": 100000, "batchSize": 16, "inputScale": 1.0,
        "inferenceOutputFileName": "", "inputLayerForInference": None,
        "gradClipValue": 10, "lossType": "ctc", "normLayer": False,
        "randomWalkAxis": -1, "warmUpSteps": 0,
        "model": MODEL_YAML,
        "dataset": {
            "name": "handwriting",
            "sessions": SEED_SESSIONS + EVAL_SESSIONS,
            "dataDir": ([DATA + "seed_model_training_data/tfrecords"] * 11 +
                        [DATA + "online_evaluation_data/recalibration/"
                         "tfrecords"] * n_ev),
            "datasetToLayerMap": list(range(11)) + list(range(11, 11 + n_ev)),
            "copyInputLayer": {str(11 + i): str(10 + i)
                               for i in range(n_ev)},
            "datasetProbability": [1] * (11 + n_ev),
            "datasetProbabilityVal": [0] * (11 + n_ev),
            "testOnTrain": True,
            "syntheticDataDir": None, "syntheticMixingRate": 0.0,
            "whiteNoiseSD": 0.0, "constantOffsetSD": 0.0,
            "randomWalkSD": 0.0, "staticGainSD": 0.0,
            "nInputFeatures": 192, "timeWarpFactor": 0.0,
            "nClasses": 31, "maxSeqElements": 500, "bufferSize": 192,
        },
    })


def main():
    args = make_args()
    nsd = NeuralSequenceDecoder(args=args)
    seed_bank = []
    for si in range(0, 11, 2):
        f, l, _ = gather_trials(nsd, si)
        for k in range(0, len(f), 4):
            seed_bank.append((f[k], [int(x) for x in l[k] if x > 0]))
    print(f"seed bank {len(seed_bank)}", flush=True)
    # Warm-up forward: Keras defers backend weight creation to the first
    # call, so the backend trainable list may only be snapshotted after
    # this (take-1 snapshotted too early and got an empty list).
    _ = forward_logits(nsd, 10, seed_bank[0][0])
    backend_vars = list(nsd.model.trainable_variables)
    print(f"backend vars: {len(backend_vars)}", flush=True)
    assert len(backend_vars) > 0, "backend weights never materialized"
    n_in = len(nsd.inputLayers[11].trainable_variables)
    print(f"input-layer vars (layer 11): {n_in}", flush=True)
    report = {"sessions": [], "protocol": {
        "pseudo": "official lm_decoded labels",
        "scope": "FULL decoder (per-session input layer + shared backend), "
                 "like the paper",
        "steps": STEPS, "lr": LR, "replay_frac": REPLAY_FRAC,
        "grad_global_norm_clip": GRAD_CLIP,
        "backend_vars": len(backend_vars), "input_layer_vars": n_in}}
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
        trainable = list(layer.trainable_variables) + backend_vars
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
        "v5_ref": {"mean_pre": 51.4, "mean_post": 40.2},
        "static_ref": 50.85}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("SUMMARY:", report["summary"])


if __name__ == "__main__":
    main()
