# -*- coding: utf-8 -*-
"""Convert jude 20260428 (T17 anarthric speech) RedisMat .mat blocks into the
nejm-brain-to-text hdf5 layout (same contract as convert_btot24.py).

Per-block extraction follows the official anarthria-speech-bci mat_to_tfrecord.py:
- trial window: go_cue_nsp_neural_time -> trial_end_nsp_neural_time, located on
  binned_neural_nsp_timestamp (10 ms bins)
- reject trials flagged trial_paused_by_CNRA / trial_timed_out
- features: [TX, SBP] in source; rewritten here to [SBP, TX] (convert_btot24 order)
- 10 ms -> 20 ms via bin pair compression (TX summed, SBP averaged; trailing odd
  bin trimmed) = official bin_compression_factor=2 semantics
- drop trials with compressed_T <= 4*n_phonemes + 10 (official length filter)
Channels vary by day (256 on 04.15-04.17, 384 otherwise): padded to 768 features.
Per-block z-scoring as in convert_btot24.py.

Sentence-block selection + vocab/open-closed labels are hardcoded from the dataset
README session table. Val split = last sentence block of each day (BtoT24 rule).
Excluded: 03.25 (isolated phonemes/orofacial), 03.26 blocks 10-22 (single words),
05.23 (phased task), block 0 DO_NOTHING blocks.
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

SRC = r".\20260428_jude\data\20260428_jude\Data\t17"
OUT = r".\20260428_jude\data\20260428_jude\hdf5"
CSV_OUT = r".\20260428_jude\data\20260428_jude\jude_description.csv"
MAX_SEQ = 500
N_FEATS = 768  # 384 channels x 2 feature types (256-channel days are zero-padded)

# day -> {block: (vocab, loop)} from the README session table (Sentence Speech part)
SENTENCE_BLOCKS = {
    "t17.2024.03.26": {23: ("50w", "OL"), 24: ("50w", "OL"), 25: ("50w", "OL"),
                       26: ("50w", "OL"), 27: ("50w", "CL")},
    "t17.2024.04.02": {1: ("50w", "OL"), 2: ("50w", "OL"), 3: ("50w", "OL"),
                       4: ("50w", "OL"), 5: ("50w", "CL"), 6: ("125k", "OL"),
                       7: ("125k", "OL"), 8: ("125k", "OL"), 9: ("125k", "OL"),
                       10: ("125k", "OL"), 11: ("125k", "OL"), 12: ("125k", "OL"),
                       13: ("125k", "CL")},
    "t17.2024.04.15": {b: ("125k", "OL") for b in range(5, 14)},
    "t17.2024.04.16": {b: ("125k", "OL") for b in range(9, 14)},
    "t17.2024.04.17": {**{b: ("125k", "OL") for b in range(1, 7)},
                       7: ("125k", "CL"), 8: ("125k", "CL"), 10: ("125k", "CL")},
    "t17.2024.05.06": {b: ("125k", "OL") for b in [1, 2, 4, 5, 6, 7, 8, 9, 10]},
    "t17.2024.05.07": {**{b: ("125k", "OL") for b in range(1, 6)},
                       **{b: ("125k", "CL") for b in range(7, 12)}},
    "t17.2024.05.13": {11: ("125k", "OL"), 12: ("125k", "OL")},
    "t17.2024.05.14": {1: ("125k", "OL"), 2: ("125k", "OL"), 3: ("125k", "CL"),
                       6: ("125k", "CL"), 7: ("125k", "CL"), 8: ("125k", "CL"),
                       9: ("125k", "CL")},
    "t17.2024.05.28": {**{b: ("125k", "OL") for b in [1, 2, 3, 4, 6, 7]},
                       8: ("125k", "CL"), 9: ("125k", "CL")},
    "t17.2024.08.12": {**{b: ("50w", "OL") for b in [1, 3, 4, 5, 6, 7]},
                       **{b: ("50w", "CL") for b in [8, 9, 10]}},
    "t17.2024.08.13": {**{b: ("50w", "OL") for b in [1, 2, 3, 5]},
                       **{b: ("50w", "CL") for b in [6, 7, 8]}},
}

G2P = G2p()
_phoneme_cache = {}


def phoneme_ids(sentence):
    """Convert a sentence to a phoneme-id list (with a global cache).

    Parameters
    ----------
    sentence : str
        Cue sentence text.

    Returns
    -------
    tuple of (list of int, list)
        Phoneme ids (``"SIL"`` maps to 40, anything else to its index in
        LOGIT_TO_PHONEME) and the phoneme sequence produced by
        ``sentence_to_phonemes``.
    """
    s = sentence.strip()
    if s not in _phoneme_cache:
        ph, _ = sentence_to_phonemes(s, g2p_instance=G2P)
        _phoneme_cache[s] = ph
    ph = _phoneme_cache[s]
    return [40 if p == "SIL" else LOGIT_TO_PHONEME.index(p) for p in ph], ph


def convert_block(mat_path, session, blk_meta):
    """Return list of trial dicts (features z-scored within this block).

    Extracts each trial's window between go_cue and trial_end on the binned
    NSP timestamp grid, rejects paused/timed-out and too-short trials,
    compresses 10 ms bins to 20 ms (TX summed, SBP averaged), orders
    features as [SBP, TX], z-scores within the block, and zero-pads channels
    up to N_FEATS.

    Parameters
    ----------
    mat_path : str
        Path to one block's RedisMat .mat file.
    session : str
        Session name (e.g. ``t17.2024.04.02``); used for context in error
        messages.
    blk_meta : dict
        The session's ``{block: (vocab, loop)}`` entry from SENTENCE_BLOCKS.

    Returns
    -------
    list of dict
        Per-trial dicts with keys ``feat``, ``ids``, ``sent``, ``block``,
        ``vocab``, ``loop``, ``trial``; empty list if no trials survive the
        filters.

    Raises
    ------
    AssertionError
        If the block number is missing from ``blk_meta``, or a phoneme is
        not present in LOGIT_TO_PHONEME.
    """
    m = loadmat(mat_path)
    blk = int(np.ravel(m["block_number"])[0])
    assert blk in blk_meta, f"{mat_path}: block {blk} not in session table"
    vocab, loop = blk_meta[blk]

    ts = np.ravel(m["binned_neural_nsp_timestamp"]).astype(np.float64)
    tx = m["binned_neural_threshold_crossings"].astype(np.float32)   # (T, C)
    sbp = m["binned_neural_spike_band_power"].astype(np.float32)     # (T, C)
    n_ch = tx.shape[1]
    cues = [str(c).strip() for c in np.ravel(m["cue"])]
    go = np.ravel(m["go_cue_nsp_neural_time"]).astype(np.float64)
    end = np.ravel(m["trial_end_nsp_neural_time"]).astype(np.float64)
    paused = np.ravel(m["trial_paused_by_CNRA"]).astype(int)
    timed = np.ravel(m["trial_timed_out"]).astype(int)

    feats, phones, sents = [], [], []
    skipped = {"flag": 0, "len": 0}
    for i in range(len(cues)):
        if paused[i] or timed[i]:
            skipped["flag"] += 1
            continue
        s_idx = int(np.argmin(np.abs(ts - go[i])))
        e_idx = int(np.argmin(np.abs(ts - end[i])))
        if e_idx - s_idx < 2:
            skipped["len"] += 1
            continue
        t = ((e_idx - s_idx) // 2) * 2
        tx_w = tx[s_idx:s_idx + t].reshape(t // 2, 2, n_ch).sum(axis=1)
        sbp_w = sbp[s_idx:s_idx + t].reshape(t // 2, 2, n_ch).mean(axis=1)
        f = np.concatenate([sbp_w, tx_w], axis=1)          # [SBP, TX] @20ms
        if f.shape[0] < 2 * n_ch:                           # pad width later
            pass
        sent = cues[i]
        ids, ph = phoneme_ids(sent)
        assert all(v is not None for v in ids), f"unknown phoneme: {ph} ({sent})"
        # official filter: compressed_T must exceed 4*nphon + 10
        if f.shape[0] <= 4 * len(ids) + 10 or not np.isfinite(f).all():
            skipped["len"] += 1
            continue
        feats.append(f)
        phones.append(ids)
        sents.append(sent)

    n = len(feats)
    if n == 0:
        print(f"  block {blk}: EMPTY (skipped {skipped})", flush=True)
        return []
    # per-block z-scoring
    stack = np.concatenate(feats, axis=0)
    mu = stack.mean(axis=0, keepdims=True)
    sd = stack.std(axis=0, keepdims=True)
    sd[sd == 0] = 1.0

    trials = []
    for i in range(n):
        f = (feats[i] - mu) / sd
        if f.shape[1] < N_FEATS:
            f = np.pad(f, ((0, 0), (0, N_FEATS - f.shape[1])))
        assert f.shape[1] == N_FEATS
        trials.append({"feat": f.astype(np.float32), "ids": phones[i],
                       "sent": sents[i], "block": blk, "vocab": vocab,
                       "loop": loop, "trial": i})
    print(f"  block {blk:2d} [{vocab}/{loop}]: {n} trials kept "
          f"(skipped {skipped}), T20ms med={int(np.median([t['feat'].shape[0] for t in trials]))}",
          flush=True)
    return trials


def write_day(session, train_trials, val_trials):
    """Write one session's train/val trials to hdf5 and accumulate CSV rows.

    Parameters
    ----------
    session : str
        Session directory name under OUT (e.g. ``t17.2024.04.02``).
    train_trials : list of dict
        Trial dicts from :func:`convert_block` for the train split.
    val_trials : list of dict
        Trial dicts for the val split (the day's last sentence block).

    Returns
    -------
    None
        Writes ``data_train.hdf5`` / ``data_val.hdf5`` under
        ``OUT/<session>`` and appends description rows to the module-level
        ALL_ROWS list.
    """
    sdir = os.path.join(OUT, session)
    os.makedirs(sdir, exist_ok=True)
    for fname, split, trials in [("data_train.hdf5", "Train", train_trials),
                                 ("data_val.hdf5", "Val", val_trials)]:
        csv_rows = []
        with h5py.File(os.path.join(sdir, fname), "w") as f:
            for i, t in enumerate(trials):
                g = f.create_group(f"trial_{i:04d}")
                g.create_dataset("input_features", data=t["feat"])
                seq = np.zeros(MAX_SEQ, dtype=np.int32)
                seq[: len(t["ids"])] = t["ids"]
                g.create_dataset("seq_class_ids", data=seq)
                tr = np.zeros(MAX_SEQ, dtype=np.int32)
                codes = [ord(c) for c in t["sent"]]
                tr[: len(codes)] = codes
                g.create_dataset("transcription", data=tr)
                g.attrs["n_time_steps"] = t["feat"].shape[0]
                g.attrs["seq_len"] = len(t["ids"])
                g.attrs["sentence_label"] = t["sent"]
                g.attrs["session"] = session
                g.attrs["block_num"] = int(t["block"])
                g.attrs["trial_num"] = i
                g.attrs["vocab"] = t["vocab"]
                g.attrs["loop"] = t["loop"]
                date = "-".join(session.split(".")[1:])
                csv_rows.append([date, 0, t["block"], len(t["ids"]),
                                 f"Jude{t['vocab']}", split])
        print(f"{fname}: {len(trials)} trials", flush=True)
        ALL_ROWS.extend(csv_rows)


def main():
    """Convert all configured sentence blocks into per-day hdf5 directories.

    For every session in SENTENCE_BLOCKS, maps block numbers to .mat files
    (warning about missing ones), converts each available block through
    :func:`convert_block` (the day's last block becomes the val split),
    writes the hdf5 files via :func:`write_day`, and finally writes the
    dataset description CSV (CSV_OUT).

    Returns
    -------
    None
    """
    os.makedirs(OUT, exist_ok=True)
    for session, blk_meta in sorted(SENTENCE_BLOCKS.items()):
        mats = {}
        for mp in glob.glob(os.path.join(SRC, session, "brainToText", "RedisMat", "*.mat")):
            try:
                m = loadmat(mp, variable_names=["block_number"])
                mats[int(np.ravel(m["block_number"])[0])] = mp
            except Exception:
                print(f"  unreadable (skipped): {mp}")
        missing = [b for b in blk_meta if b not in mats]
        if missing:
            print(f"{session}: WARNING missing blocks {missing}")
        last_blk = max(b for b in blk_meta if b in mats)
        train_trials, val_trials = [], []
        for blk in sorted(b for b in blk_meta if b in mats):
            trials = convert_block(mats[blk], session, blk_meta)
            (val_trials if blk == last_blk else train_trials).extend(trials)
        print(f"{session}: train={len(train_trials)} val={len(val_trials)} "
              f"(last block {last_blk} -> val)", flush=True)
        write_day(session, train_trials, val_trials)

    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Date", "Post-implant day", "Block number",
                    "Number of sentences", "Corpus", "Split"])
        w.writerows(ALL_ROWS)
    print(f"\ncsv: {CSV_OUT} | rows={len(ALL_ROWS)} | "
          f"unique sentences={len(_phoneme_cache)}")


ALL_ROWS = []

if __name__ == "__main__":
    main()
