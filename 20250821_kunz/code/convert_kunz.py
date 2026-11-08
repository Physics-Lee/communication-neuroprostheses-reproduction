# -*- coding: utf-8 -*-
"""Convert Kunz inner-speech sentenceData (.mat per session/behavior/vocab)
into the nejm-brain-to-text hdf5 layout (same contract as convert_btot24.py).

Source structure per sentenceData readme:
- sentenceFeatures : N x 1, per-trial (T, 512); FIRST half binnedTX, LAST half spikePow
- sentenceText     : N x 1 sentences
- blockNum         : T x 1 per-time-step block id; trial block looked up at goTrialEpochs[i,0]
- binSize          : 10 (T15), 20 (T12, T16), 33 (t16.2024.04.29 eval only)

Canonicalization to the 20 ms grid used by the BtoT24/25 pipeline:
- 10 ms  -> pair compression (TX summed, SBP averaged; trailing odd bin trimmed),
            same semantics as anarthria-speech-bci's bin_compression_factor=2
- 20 ms  -> unchanged
- 33 ms  -> per-channel linear interpolation onto the 20 ms grid (z-score afterwards
            absorbs any per-channel scale factor)
Feature order is rewritten to [spikePow, TX] to match convert_btot24.py.
Per-block z-scoring (blockNum) as in convert_btot24.py.
Plain .mat (open-loop blocks) -> data_train.hdf5; _eval.mat (closed-loop) -> data_val.hdf5.
"""
import csv
import glob
import os
import sys

import h5py
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, r".\20240815_card\nejm-brain-to-text")                # nejm_b2txt_utils
sys.path.insert(0, r".\20240815_card\nejm-brain-to-text\model_training") # evaluate_model_helpers
from nejm_b2txt_utils.general_utils import sentence_to_phonemes  # noqa: E402
from evaluate_model_helpers import LOGIT_TO_PHONEME                # noqa: E402
from g2p_en import G2p                                             # noqa: E402

SRC = r".\20250821_kunz\data\20250821_kunz\sentenceData"
OUT = r".\20250821_kunz\data\20250821_kunz\hdf5"
CSV_OUT = r".\20250821_kunz\data\20250821_kunz\kunz_description.csv"
MAX_SEQ = 500

G2P = G2p()
_phoneme_cache = {}


def phoneme_ids(sentence):
    """Convert a sentence to phoneme id sequence via grapheme-to-phoneme.

    Uses a module-level cache so repeated sentences are phonemized once.

    Parameters
    ----------
    sentence : str
        Input sentence text.

    Returns
    -------
    tuple of (list of int, list of str)
        Phoneme class ids (40 marks `"SIL"`, others indexed into
        ``LOGIT_TO_PHONEME``) and the phoneme string sequence.
    """
    s = sentence.strip()
    if s not in _phoneme_cache:
        ph, _ = sentence_to_phonemes(s, g2p_instance=G2P)
        _phoneme_cache[s] = ph
    ph = _phoneme_cache[s]
    return [40 if p == "SIL" else LOGIT_TO_PHONEME.index(p) for p in ph], ph


def to_str(x):
    """Recursively unwrap loadmat cell contents into a python str.

    Parameters
    ----------
    x : numpy.ndarray or object
        Value loaded by ``scipy.io.loadmat`` (possibly nested arrays,
        char codes, or unicode cells).

    Returns
    -------
    str
        The decoded string content.
    """
    while isinstance(x, np.ndarray):
        if x.dtype.kind in "US":
            return "".join(x.ravel().tolist())
        if x.size == 1:
            x = x.ravel()[0]
            continue
        return "".join(chr(int(c)) for c in np.ravel(x) if c > 0)
    return str(x)


def compress_to_20ms(tx, sbp, bin_ms):
    """tx, sbp: (T, 256). Returns (T', 256) each on a 20 ms grid.

    Canonicalizes binned features to the 20 ms grid: 10 ms bins are
    pair-compressed (TX summed, SBP averaged, trailing odd bin trimmed),
    20 ms bins pass through unchanged, and 33.33 ms bins are per-channel
    linearly interpolated onto the 20 ms grid.

    Parameters
    ----------
    tx : numpy.ndarray
        Shape (T, 256) threshold-crossing feature matrix at the source
        bin size.
    sbp : numpy.ndarray
        Shape (T, 256) spike-power feature matrix at the source bin size.
    bin_ms : float
        Source bin size in milliseconds (10, 20, or 33).

    Returns
    -------
    tuple of (numpy.ndarray, numpy.ndarray)
        The TX and SBP matrices, each shape (T', 256), resampled to the
        20 ms grid.

    Raises
    ------
    ValueError
        If ``bin_ms`` is not one of the handled values (10, 20, 33).
    """
    if bin_ms == 20:
        return tx, sbp
    if bin_ms == 10:
        t = (tx.shape[0] // 2) * 2
        tx2 = tx[:t].reshape(t // 2, 2, tx.shape[1]).sum(axis=1)
        sbp2 = sbp[:t].reshape(t // 2, 2, sbp.shape[1]).mean(axis=1)
        return tx2, sbp2
    if bin_ms == 33:
        # 33.33 ms bins -> 20 ms grid via per-channel linear interpolation
        T = tx.shape[0]
        src_t = np.arange(T) * (100.0 / 3.0)
        dst_t = np.arange(0, src_t[-1], 20.0)
        out = []
        for arr in (tx, sbp):
            interp = np.empty((len(dst_t), arr.shape[1]), dtype=arr.dtype)
            for c in range(arr.shape[1]):
                interp[:, c] = np.interp(dst_t, src_t, arr[:, c])
            out.append(interp)
        return out[0], out[1]
    raise ValueError(f"unhandled binSize {bin_ms}")


def convert_mat(mat_path, out_h5, session, split):
    """Convert one Kunz sentenceData .mat into the nejm-brain-to-text hdf5.

    For each trial: splits sentenceFeatures into TX/SBP halves, resamples
    to 20 ms, reorders features to [spikePow, TX], phonemizes the
    sentence, drops trials failing the official CTC length filter or with
    non-finite features, z-scores features per block, and writes one hdf5
    group per trial.

    Parameters
    ----------
    mat_path : str
        Path to the source sentenceData ``.mat`` file.
    out_h5 : str
        Path of the hdf5 file to create.
    session : str
        Session name (e.g. ``t15.2024.03.03`` or ``..._eval``), stored as
        a trial attribute.
    split : str
        Split label, `"Train"` or `"Val"`, recorded in the CSV rows.

    Returns
    -------
    tuple of (list of tuple, int)
        CSV rows ``(session, block_num, n_phonemes, split)`` and the
        number of trials written.
    """
    m = loadmat(mat_path)
    n = m["sentenceText"].shape[0]
    bin_ms = float(m["binSize"].ravel()[0])
    ge = m["goTrialEpochs"]
    bn = m["blockNum"].ravel().astype(int)

    feats, phones_per_trial, blocks, sents = [], [], [], []
    dropped = 0
    for i in range(n):
        sf = np.asarray(m["sentenceFeatures"][i, 0], dtype=np.float32)
        assert sf.ndim == 2 and sf.shape[1] == 512, f"trial {i}: shape {sf.shape}"
        tx, sbp = sf[:, :256], sf[:, 256:]
        tx, sbp = compress_to_20ms(tx, sbp, bin_ms)
        f = np.concatenate([sbp, tx], axis=1)  # [spikePow, TX] like convert_btot24

        sent = to_str(m["sentenceText"][i]).strip()
        ids, ph = phoneme_ids(sent)
        assert all(v is not None for v in ids), f"unknown phoneme in trial {i}: {ph}"
        # official length filter (same rule as anarthria-speech-bci mat_to_tfrecord):
        # frames must exceed 4*nphonemes+10, else patched CTC input_len < target_len
        # (one t16.2024.12.18 trial hits this; weaker T>=nphon+1 let it crash training)
        if f.shape[0] <= 4 * len(ids) + 10 or not np.isfinite(f).all():
            dropped += 1
            continue
        feats.append(f)
        phones_per_trial.append(ids)
        blocks.append(int(bn[int(ge[i, 0]) - 1]))
        sents.append(sent)
    n = len(feats)
    blocks = np.asarray(blocks)

    # blockwise z-scoring: stats over all trials of the same block in this file
    uniq = np.unique(blocks)
    for b in uniq:
        sel = [i for i in range(n) if blocks[i] == b]
        stack = np.concatenate([feats[i] for i in sel], axis=0)
        mu = stack.mean(axis=0, keepdims=True)
        sd = stack.std(axis=0, keepdims=True)
        sd[sd == 0] = 1.0
        for i in sel:
            feats[i] = (feats[i] - mu) / sd

    csv_rows = []
    with h5py.File(out_h5, "w") as f:
        for i in range(n):
            g = f.create_group(f"trial_{i:04d}")
            g.create_dataset("input_features", data=feats[i].astype(np.float32))
            seq = np.zeros(MAX_SEQ, dtype=np.int32)
            seq[: len(phones_per_trial[i])] = phones_per_trial[i]
            g.create_dataset("seq_class_ids", data=seq)
            tr = np.zeros(MAX_SEQ, dtype=np.int32)
            codes = [ord(c) for c in sents[i]]
            tr[: len(codes)] = codes
            g.create_dataset("transcription", data=tr)
            g.attrs["n_time_steps"] = feats[i].shape[0]
            g.attrs["seq_len"] = len(phones_per_trial[i])
            g.attrs["sentence_label"] = sents[i]
            g.attrs["session"] = session
            g.attrs["block_num"] = int(blocks[i])
            g.attrs["trial_num"] = i
            g.attrs["bin_size_ms_source"] = bin_ms
            csv_rows.append((session, int(blocks[i]), len(phones_per_trial[i]), split))
    print(f"{out_h5}: {n} trials, {len(uniq)} blocks, "
          f"{sum(len(p) for p in phones_per_trial)} phonemes "
          f"(binSize {bin_ms} -> 20ms, dropped {dropped})", flush=True)
    return csv_rows, n


def main():
    """Convert every sentenceData .mat into hdf5 plus a description CSV.

    Iterates over ``SRC/*.mat`` in sorted order, routes each session to a
    Train/Val split by filename suffix, and writes the per-trial
    ``kunz_description.csv`` with date, block, phoneme count, corpus, and
    split columns.

    Returns
    -------
    None
        Converted hdf5 files, the description CSV, and summary counts are
        printed to stdout.
    """
    os.makedirs(OUT, exist_ok=True)
    all_rows = []
    totals = {}
    for mp in sorted(glob.glob(os.path.join(SRC, "*.mat"))):
        session = os.path.basename(mp).replace(".mat", "")
        is_eval = session.endswith("_eval")
        split = "Val" if is_eval else "Train"
        sdir = os.path.join(OUT, session)
        os.makedirs(sdir, exist_ok=True)
        fname = "data_val.hdf5" if is_eval else "data_train.hdf5"
        rows, n = convert_mat(mp, os.path.join(sdir, fname), session, split)
        all_rows.extend(rows)
        totals[session] = (split, n)

    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Post-implant day", "Block number",
                    "Number of sentences", "Corpus", "Split"])
        for session, blk, nph, split in all_rows:
            parts = session.split("_")[0].split(".")  # t15.2024.03.03 -> date parts
            date = "-".join(parts[1:])
            behavior = "Imagined" if "_imagined_" in session else "Attempted"
            vocab = "50words" if "50words" in session else "125kwords"
            corpus = f"Kunz{behavior}{vocab}"
            w.writerow([date, 0, blk, nph, corpus, split])
    print(f"\ntotals: {totals}")
    print(f"unique sentences phonemized: {len(_phoneme_cache)}")
    print(f"csv: {CSV_OUT}")


if __name__ == "__main__":
    main()
