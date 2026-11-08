# -*- coding: utf-8 -*-
"""Session-3-only training-budget experiment: can a CORP-scale update
budget (hundreds of steps, not dozens) pull a 60%-drifted input layer
back, using the OFFICIAL pseudo-labels? Success here -> v5 full run."""
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

STEPS = 500
LR = 5e-5
EVAL_IDS = [0, 5, 10, 20, 30]


def cer(nsd, li, feat, label):
    """Character error rate of one trial's greedy decode, in percent.

    Parameters
    ----------
    nsd : NeuralSequenceDecoder
        Decoder whose input layer and frozen backend run the forward pass.
    li : int
        Index of the input layer to push the features through.
    feat : numpy.ndarray
        Raw (unsmoothed) neural feature matrix for one trial, shape
        (n_time_steps, n_features).
    label : numpy.ndarray
        Ground-truth 1-based label ids of the trial; entries <= 0 are
        ignored when forming the reference sequence.

    Returns
    -------
    float
        100 * edit distance / reference length, with the denominator
        guarded at 1.
    """
    dec = greedy_decode(forward_logits(nsd, li, feat))
    ed, ref = editdist(dec, [int(x) for x in label if x > 0])
    return 100 * ed / max(ref, 1)


def main():
    """Run the session-3-only training-budget experiment.

    Builds the official decoder, gathers session 3 (t5.2022.10.18)
    trials, joins them to the official pseudo-labels, prints pre-update
    CERs on the fixed probe trials, then fine-tunes only that session's
    input layer for STEPS Adam steps (lr LR) on pseudo-label trials and
    prints probe CERs every 100 steps. Success here justified v5's
    500-step full-run budget.
    """
    args = make_args()
    nsd = NeuralSequenceDecoder(args=args)
    sess_i = 2
    feats, labels, sents = gather_trials(nsd, 11 + sess_i)
    official = load_official_labels(EVAL_SESSIONS[sess_i])
    usable = [(feats[k], official[sents[k]]) for k in range(len(feats))
              if sents[k] in official and len(official[sents[k]]) > 0]
    layer = nsd.inputLayers[11 + sess_i]
    pre = [round(cer(nsd, 11 + sess_i, feats[k], labels[k]), 1)
           for k in EVAL_IDS]
    print("pre CERs:", pre, flush=True)
    opt = tf.keras.optimizers.Adam(LR)
    for step in range(STEPS):
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
        if step % 100 == 0 or step == STEPS - 1:
            c = [round(cer(nsd, 11 + sess_i, feats[k], labels[k]), 1)
                 for k in EVAL_IDS]
            print(f"step {step}: loss {float(loss):.1f} CERs {c}",
                  flush=True)


if __name__ == "__main__":
    main()
