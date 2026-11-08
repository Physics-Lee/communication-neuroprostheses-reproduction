# -*- coding: utf-8 -*-
"""Fan 2023 (CORP) tier-3 v4: simulate the online recalibration dynamic
using the OFFICIAL pseudo-labels -- the LM-corrected online decode outputs
(lm_decoded) that the Dryad evaluation mats recorded for every trial, i.e.
the exact training signal the paper's CORP consumed online (WER 4-10%,
vs our greedy tier-3 v1-v3 labels that collapsed beyond ~60% drift).

Trials are joined to tfrecord neural features by sentence text (verified
40/40 on t5.2022.09.29; mat and tfrecord orders differ). Everything else
follows v3: fresh input layer per session (copy of the incoming layer),
backend frozen, block-wise fine-tuning, seed replay, divergence rollback.

Success criterion: post-CER per session stays in the low teens or below
across ALL sessions including the late ones where the static decoder
sits at 50%+ -- i.e. the CORP dynamic actually holds, matching the
paper's stable online accuracy (93.84% over 15 released/unreleased
sessions).
"""
import io
import json
import os
import sys
import zipfile

import numpy as np

NSD_ROOT = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "../open_source_code_for_reference/20231201_fan_CORP/NeuralDecoder"))
sys.path.insert(0, NSD_ROOT)
DATA = ("./20231201_fan/data/20231201_fan/"
        "CORP_data_release/")
ZIP = ("./20231201_fan/data/raw/"
       "20231201_fan/CORP_data_release.zip")
OUT = ("./20231201_fan/result/"
       "tier3_corp_sim_official_labels.json")

SEED_SESSIONS = [
    "t5.2022.05.18", "t5.2022.05.23", "t5.2022.05.25", "t5.2022.06.01",
    "t5.2022.06.03", "t5.2022.06.06", "t5.2022.06.08", "t5.2022.06.13",
    "t5.2022.06.15", "t5.2022.06.22", "t5.2022.09.01"]
EVAL_SESSIONS = [
    "t5.2022.09.29", "t5.2022.10.06", "t5.2022.10.18", "t5.2022.10.25",
    "t5.2022.10.27", "t5.2022.11.01", "t5.2022.11.03", "t5.2022.12.08",
    "t5.2022.12.15", "t5.2023.02.28"]

MODEL_YAML = {
    "nUnits": 512, "subsampleFactor": 1, "weightReg": 1e-5, "actReg": 0.0,
    "bidirectional": False, "dropout": 0.4, "trainable": True, "nLayers": 2,
    "stack_kwargs": {"kernel_size": 1, "strides": 2},
}

N_PRE = 6
N_POST = 6
BLOCK = 10
TRAIN_STEPS = 12
LR = 2e-5
SEED_REPLAY = 8
ROLLBACK = True

from neuralDecoder.datasets.handwritingDataset import CHAR_DEF  # noqa: E402
import tensorflow as tf  # noqa: E402
from neuralDecoder.neuralSequenceDecoder import NeuralSequenceDecoder  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402
import scipy.io as sio  # noqa: E402

CH2ID = {c: i + 1 for i, c in enumerate(CHAR_DEF)}


def norm(s):
    """Normalize a sentence string for mat-to-tfrecord joining.

    Replaces '>' word separators with spaces, strips NUL characters, and
    collapses all whitespace runs to single spaces.

    Parameters
    ----------
    s : str
        Raw sentence text from a .mat field or tfrecord transcription.

    Returns
    -------
    str
        Whitespace-normalized sentence text.
    """
    return ' '.join(str(s).replace('>', ' ').replace(chr(0), '').split())


def text_to_ids(text):
    """Map a sentence to the official 1-based character-id vocabulary.

    Spaces are converted to the '>' separator token; characters absent
    from ``CH2ID`` are silently dropped.

    Parameters
    ----------
    text : str
        Normalized sentence text.

    Returns
    -------
    list of int
        1-based character ids (possibly empty if nothing maps).
    """
    ids = []
    for ch in text:
        if ch == ' ':
            ch = '>'
        if ch in CH2ID:
            ids.append(CH2ID[ch])
    return ids


def load_official_labels(sess):
    """{norm(true sentence): lm_decoded label ids} for one session.

    Reads the session's mat from the Dryad zip (recalibration arm) and
    pairs each ground-truth sentence with the label-id encoding of its
    LM-corrected online decode -- the exact pseudo-label signal the
    paper's CORP consumed online.

    Parameters
    ----------
    sess : str
        Session name, e.g. 't5.2022.09.29'; selects
        ``online_evaluation_data/recalibration/mat/<sess>.mat`` inside
        the zip.

    Returns
    -------
    dict
        Mapping from normalized ground-truth sentence text to the
        1-based label id list of the official ``lm_decoded`` output.
    """
    z = zipfile.ZipFile(ZIP)
    m = sio.loadmat(io.BytesIO(z.read(
        f"CORP_data_release/online_evaluation_data/recalibration/mat/"
        f"{sess}.mat")), squeeze_me=True, struct_as_record=False)
    sents = np.atleast_1d(m["sentences"])
    lm = np.atleast_1d(m["lm_decoded"])
    table = {}
    for s, l in zip(sents, lm):
        table[norm(s)] = text_to_ids(norm(l))
    return table


def make_args():
    """Build the v4 simulation's OmegaConf argument block.

    Registers all 11 seed sessions plus the 10 evaluation sessions; each
    evaluation dataset gets its own input layer initialized as a copy of
    the previous session's layer (``copyInputLayer``), and everything is
    served through the package's testOnTrain loader with the backend
    frozen (``trainableBackend: 0``).

    Returns
    -------
    omegaconf.DictConf
        Fully-populated inference-mode config for
        ``NeuralSequenceDecoder`` (checkpoint ckpt-12500 of the official
        pretrained seed model).
    """
    n_ev = len(EVAL_SESSIONS)
    return OmegaConf.create({
        "mode": "inference", "gpuNumber": "0",
        "outputDir": "./20231201_fan/tmp_run4",
        "loadDir": DATA + "pretrained_seed_model",
        "loadCheckpointIdx": 12500,
        "smoothInputs": 1, "smoothKernelSD": 2,
        "learnRateStart": 0.01, "learnRateEnd": 0.0,
        "learnRateDecaySteps": 100000, "learnRatePower": 1.0,
        "trainableInput": 1, "trainableBackend": 0,
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


def gather_trials(nsd, sess_idx):
    """Collect per-trial features, labels, and sentence text for one dataset.

    Iterates the package's validation dataset for session index
    ``sess_idx``, unpacks each batch into per-trial numpy arrays, trims
    features to their true ``nTimeSteps``, and decodes the byte-string
    transcriptions to normalized text.

    Parameters
    ----------
    nsd : NeuralSequenceDecoder
        Decoder whose ``tfValDatasets`` holds the loaded trials.
    sess_idx : int
        Dataset index into ``nsd.tfValDatasets`` (0-10 = seed sessions,
        11-20 = evaluation sessions).

    Returns
    -------
    tuple of (list, list, list)
        Per-trial feature matrices of shape (n_time_steps, 192), 1-based
        ground-truth label id arrays, and normalized sentence strings
        (usable as keys into ``load_official_labels`` tables).
    """
    feats, labels, sents = [], [], []
    for data in nsd.tfValDatasets[sess_idx]:
        f = data["inputFeatures"].numpy()
        lab = data["seqClassIDs"].numpy()
        nt = data["nTimeSteps"].numpy()
        t = data["transcription"].numpy()
        for k in range(len(f)):
            feats.append(f[k][:nt[k]])
            labels.append(lab[k])
            sents.append(norm(b''.join(np.atleast_1d(t[k].flatten()))
                              .decode('utf-8', errors='replace')))
    return feats, labels, sents


_GAUSS = None

def gauss_smooth_np(x, sd=2.0):
    """Gaussian-smooth a feature matrix along time, in numpy.

    Numpy mirror of the package's TF input smoothing: the kernel is built
    once from a unit impulse and cached in the module global ``_GAUSS``,
    rows are edge-padded, and each feature column is convolved so the
    output length matches the input.

    Parameters
    ----------
    x : numpy.ndarray
        Feature matrix of shape (n_time_steps, n_features).
    sd : float, optional
        Standard deviation of the Gaussian kernel (default 2.0, matching
        the package's ``smoothKernelSD``). Only the first call's value
        shapes the cached kernel.

    Returns
    -------
    numpy.ndarray
        Smoothed feature matrix, same shape as ``x``.
    """
    global _GAUSS
    if _GAUSS is None:
        inp = np.zeros(100); inp[50] = 1
        from scipy.ndimage import gaussian_filter1d
        g = gaussian_filter1d(inp, sd)
        g = g[g > 0.01]
        _GAUSS = g / g.sum()
    k = len(_GAUSS) // 2
    pad = np.pad(x, ((k, k), (0, 0)), mode="edge")
    return np.stack([np.convolve(pad[:, c], _GAUSS, mode="valid")
                     for c in range(x.shape[1])], axis=1)


def forward_logits(nsd, layer_idx, feat):
    """Run one trial through smoothing, an input layer, and the frozen backend.

    Parameters
    ----------
    nsd : NeuralSequenceDecoder
        Decoder providing ``inputLayers`` and the backend ``model``.
    layer_idx : int
        Index of the input layer to apply.
    feat : numpy.ndarray
        Raw (unsmoothed) per-trial feature matrix of shape
        (n_time_steps, 192).

    Returns
    -------
    numpy.ndarray
        Frame-wise class logits of shape (n_frames, 32) -- 31 character
        classes plus the blank class at index 31.
    """
    x = gauss_smooth_np(feat)
    x = tf.constant(x[np.newaxis, ...], dtype=tf.float32)
    y = nsd.inputLayers[layer_idx](x, training=False)
    return nsd.model(y, training=False)[0].numpy()


def greedy_decode(logits):
    """CTC-greedy-decode frame logits into a 1-based id sequence.

    Collapses repeated frames and drops blank frames (class index 31),
    then shifts surviving classes to the dataset's 1-based convention.

    Parameters
    ----------
    logits : numpy.ndarray
        Frame-wise class logits of shape (n_frames, 32).

    Returns
    -------
    list of int
        Decoded character ids (1-based); may be empty.
    """
    ids = np.argmax(logits, axis=-1)
    seq = []
    prev = -1
    for i in ids:
        if i != prev and i != 31:
            seq.append(int(i) + 1)
        prev = i
    return seq


def ctc_loss(labels1b, logits, logit_len):
    """Compute the CTC loss for one trial.

    labels arrive 1-based (the dataset convention); the official
    _trainStep subtracts 1 before ctc_loss_v2 with blank_index=-1 --
    v1-v4a of this simulation fed 1-based labels directly, shifting every
    character by one class (the real cause of the 'self-training
    collapse', found 2026-10-26 evening).

    Parameters
    ----------
    labels1b : sequence of int
        1-based target character ids for the single trial.
    logits : numpy.ndarray
        Frame-wise class logits of shape (n_frames, 32) for one trial.
    logit_len : tensorflow.Tensor
        Scalar int32 count of valid frames in ``logits``.

    Returns
    -------
    tensorflow.Tensor
        Scalar mean CTC loss (0-based labels, blank at the last class).
    """
    values = [int(v) - 1 for v in labels1b]
    sparse = tf.sparse.SparseTensor(
        indices=[[0, i] for i in range(len(values))],
        values=values,
        dense_shape=[1, len(values)])
    loss = tf.compat.v1.nn.ctc_loss_v2(
        sparse, logits[tf.newaxis, :logit_len], None,
        [logit_len], logits_time_major=False, blank_index=-1)
    return tf.reduce_mean(loss)


def editdist(a, b):
    """Levenshtein edit distance between two id sequences.

    Uses a rolling-row dynamic-programming table (O(len(b)) memory).

    Parameters
    ----------
    a : sequence
        Reference id sequence.
    b : sequence
        Hypothesis id sequence.

    Returns
    -------
    tuple of (int, int)
        Edit distance between ``a`` and ``b``, and ``len(a)`` (the
        reference length, for CER normalization).
    """
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                        prev + (a[i - 1] != b[j - 1]))
            prev = cur
    return dp[n], m


def cer_over(nsd, layer_idx, feats, labels, idxs):
    """Pooled character error rate over selected trials.

    Greedy-decodes each selected trial through the given input layer and
    pools edit distances and reference lengths before dividing.

    Parameters
    ----------
    nsd : NeuralSequenceDecoder
        Decoder used for the forward passes.
    layer_idx : int
        Index of the input layer to evaluate through.
    feats : list of numpy.ndarray
        Per-trial raw feature matrices.
    labels : list of numpy.ndarray
        Per-trial 1-based ground-truth label id sequences.
    idxs : sequence of int
        Trial indices to include in the pooled score.

    Returns
    -------
    float
        100 * summed edit distance / summed reference length
        (denominator guarded at 1).
    """
    e = t = 0
    for k in idxs:
        dec = greedy_decode(forward_logits(nsd, layer_idx, feats[k]))
        ed, ref = editdist(dec, [int(x) for x in labels[k] if x > 0])
        e += ed
        t += ref
    return 100 * e / max(t, 1)


def main():
    """Run the tier-3 v4 CORP simulation with official pseudo-labels.

    Builds the decoder, fills the seed replay bank, then walks the 10
    evaluation sessions in order: a fresh input layer per session (a copy
    of the incoming layer), block-wise fine-tuning on official
    lm_decoded pseudo-labels interleaved with seed replay, divergence
    rollback per block, and pre/post CER per session. Writes the JSON
    report to OUT and prints per-session logs plus the summary.
    """
    os.makedirs("./20231201_fan/tmp_run4",
                exist_ok=True)
    args = make_args()
    nsd = NeuralSequenceDecoder(args=args)
    report = {"sessions": [], "protocol": {
        "pseudo": "OFFICIAL lm_decoded labels from the Dryad mats (the "
                  "exact training signal the paper's CORP consumed online)",
        "join": "sentence text (verified set-equal per session)",
        "trainable": "current session input layer only",
        "block": BLOCK, "steps": TRAIN_STEPS, "lr": LR,
        "seed_replay": SEED_REPLAY, "rollback": ROLLBACK}}
    seed_bank = []
    for si in range(0, 11, 2):
        f, l, _ = gather_trials(nsd, si)
        for k in range(0, len(f), 4):
            seed_bank.append((f[k], [int(x) for x in l[k] if x > 0]))
    print(f"seed replay bank: {len(seed_bank)}", flush=True)
    incoming_layer = 10
    for si, sess in enumerate(EVAL_SESSIONS):
        ds_idx = 11 + si
        layer_idx = 11 + si
        feats, labels, sents = gather_trials(nsd, ds_idx)
        official = load_official_labels(sess)
        pseudo_by_trial = []
        hit = 0
        for s in sents:
            lab = official.get(s)
            if lab is None:
                lab = []
            else:
                hit += 1
            pseudo_by_trial.append(lab)
        n = len(feats)
        pre_cer = cer_over(nsd, incoming_layer, feats, labels,
                           list(range(0, min(N_PRE, n))))
        layer = nsd.inputLayers[layer_idx]
        opt = tf.keras.optimizers.Adam(LR)
        block_report = []
        for b0 in range(0, n, BLOCK):
            block = list(range(b0, min(b0 + BLOCK, n)))
            entry_w = [w.copy() for w in layer.get_weights()] \
                if ROLLBACK else None
            entry_cer = cer_over(nsd, layer_idx, feats, labels, block[:2])
            usable = [k for k in block if len(pseudo_by_trial[k]) > 0]
            for _ in range(TRAIN_STEPS):
                if np.random.rand() < 0.5 and seed_bank:
                    xf, lab = seed_bank[np.random.randint(len(seed_bank))]
                elif usable:
                    k = usable[np.random.randint(len(usable))]
                    xf, lab = feats[k], pseudo_by_trial[k]
                else:
                    continue
                x = gauss_smooth_np(xf)
                with tf.GradientTape() as tape:
                    xin = tf.constant(x[np.newaxis], dtype=tf.float32)
                    y = layer(xin, training=True)
                    logits = nsd.model(y, training=False)
                    loss = ctc_loss(
                        lab, logits[0],
                        tf.cast(tf.shape(logits)[1], tf.int32))
                grads = tape.gradient(loss, layer.trainable_variables)
                if grads and all(g is not None for g in grads):
                    opt.apply_gradients(zip(grads, layer.trainable_variables))
            exit_cer = cer_over(nsd, layer_idx, feats, labels, block[-2:])
            if ROLLBACK and exit_cer > entry_cer + 5.0:
                layer.set_weights(entry_w)
                block_report.append(round(entry_cer, 1))
            else:
                block_report.append(round(exit_cer, 1))
        post_cer = cer_over(nsd, layer_idx, feats, labels,
                            list(range(max(0, n - N_POST), n)))
        report["sessions"].append({
            "session": sess, "n_trials": n, "label_join_hits": hit,
            "pre_cer_pct": round(pre_cer, 1),
            "post_cer_pct": round(post_cer, 1),
            "block_last2_cer": block_report})
        print(f"{sess}: n={n} join={hit} pre={pre_cer:.1f} "
              f"post={post_cer:.1f} blocks={block_report}", flush=True)
        incoming_layer = layer_idx
    pres = [s["pre_cer_pct"] for s in report["sessions"]]
    posts = [s["post_cer_pct"] for s in report["sessions"]]
    report["summary"] = {
        "mean_pre_cer": round(float(np.mean(pres)), 1),
        "mean_post_cer": round(float(np.mean(posts)), 1),
        "static_reference_cer": 50.85,
        "greedy_label_v3_reference": "diverged at session 3"}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("SUMMARY:", report["summary"])


if __name__ == "__main__":
    main()
