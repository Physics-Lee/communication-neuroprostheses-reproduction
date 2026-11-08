# -*- coding: utf-8 -*-
"""One-step training diagnostic for the CORP simulation: is the gradient
actually flowing into the input layer, does the loss move, how big is the
weight change? Run on session 3 (t5.2022.10.18) where every version
diverges."""
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
    forward_logits, greedy_decode, editdist, ctc_loss, LR, EVAL_SESSIONS)
from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder  # noqa: E402


def cer(nsd, layer_idx, feat, label):
    """Greedy-decode one trial and score it against the ground truth.

    Parameters
    ----------
    nsd : NeuralSequenceDecoder
        Decoder whose input layer and frozen backend run the forward pass.
    layer_idx : int
        Index of the input layer to push the features through.
    feat : numpy.ndarray
        Raw (unsmoothed) neural feature matrix for one trial, shape
        (n_time_steps, n_features).
    label : numpy.ndarray
        Ground-truth 1-based label ids of the trial; entries <= 0 are
        ignored when forming the reference sequence.

    Returns
    -------
    tuple of (float, list of int)
        Character error rate in percent and the decoded 1-based id
        sequence.
    """
    dec = greedy_decode(forward_logits(nsd, layer_idx, feat))
    ed, ref = editdist(dec, [int(x) for x in label if x > 0])
    return 100 * ed / max(ref, 1), dec


def main():
    """Run the 30-step single-trial training diagnostic on session 3.

    Takes trial 0 of session 3 (t5.2022.10.18) with its official
    pseudo-label, prints the input layer's trainable state and the
    starting CER/decode, then applies 30 Adam steps on that single trial
    -- reporting loss, gradient norm, whether gradients were applied, and
    CER every 5 steps -- and finally checks the same trial (overfit) and
    a neighbor trial to separate learning from drift damage.
    """
    args = make_args()
    nsd = NeuralSequenceDecoder(args=args)
    sess_i = 2  # t5.2022.10.18
    ds_idx = 11 + sess_i
    layer_idx = 11 + sess_i
    feats, labels, sents = gather_trials(nsd, ds_idx)
    official = load_official_labels(EVAL_SESSIONS[sess_i])
    layer = nsd.inputLayers[layer_idx]
    print("layer trainable:", layer.trainable,
          "| n vars:", len(layer.trainable_variables))
    opt = tf.keras.optimizers.Adam(LR)
    k = 0
    lab1b = official[sents[k]]
    x = gauss_smooth_np(feats[k])
    c0, dec0 = cer(nsd, layer_idx, feats[k], labels[k])
    print(f"trial 0 start CER {c0:.1f}% dec={dec0[:20]} label="
          f"{[int(v) for v in labels[k] if v > 0][:20]}")
    for step in range(30):
        with tf.GradientTape() as tape:
            xin = tf.constant(x[np.newaxis], dtype=tf.float32)
            y = layer(xin, training=True)
            logits = nsd.model(y, training=False)
            loss = ctc_loss(lab1b, logits[0],
                            tf.cast(tf.shape(logits)[1], tf.int32))
        grads = tape.gradient(loss, layer.trainable_variables)
        gnorm = float(np.sqrt(sum(float(tf.reduce_sum(g ** 2)) for g in grads
                                  if g is not None))) if grads else 0
        ok = grads and all(g is not None for g in grads)
        if ok:
            opt.apply_gradients(zip(grads, layer.trainable_variables))
        if step % 5 == 0 or step == 29:
            c, dec = cer(nsd, layer_idx, feats[k], labels[k])
            print(f"step {step}: loss {float(loss):.2f} gnorm {gnorm:.4f} "
                  f"applied={bool(ok)} CER {c:.1f}%")
    # overfit check on the SAME trial
    c, dec = cer(nsd, layer_idx, feats[k], labels[k])
    print(f"after 30 steps on ONE trial: CER {c:.1f}%")
    c2, _ = cer(nsd, layer_idx, feats[1], labels[1])
    print(f"neighbor trial 1 CER {c2:.1f}% (drift damage check)")


if __name__ == "__main__":
    main()
