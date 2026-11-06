# -*- coding: utf-8 -*-
"""Convert BtoT24 competitionData (.mat per day) into the nejm-brain-to-text hdf5 layout.

- features: np.concatenate([spikePow, tx1], axis=1) -> (T, 512) float32
- per-block z-scoring (blockIdx) as advised by the dataset readme
- phoneme targets via the repo's own sentence_to_phonemes() (g2p_en),
  class ids in LOGIT_TO_PHONEME order (SIL/' | ' -> 40)
- train/  -> <out>/<session>/data_train.hdf5
  test/   -> <out>/<session>/data_val.hdf5   (labeled; serves as our validation)
- emits btot24_description.csv for corpus lookup in evaluate scripts
competitionHoldOut/ is intentionally skipped (no labels).
"""
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

SRC = r".\20230831_willett\data\20230616_willett_BtoT24\competitionData"
OUT = r".\20230831_willett\data\20230616_willett_BtoT24\hdf5"
CSV_OUT = r".\20230831_willett\data\20230616_willett_BtoT24\btot24_description.csv"
MAX_SEQ = 500

G2P = G2p()
_phoneme_cache = {}


def phoneme_ids(sentence):
    s = sentence.strip()
    if s not in _phoneme_cache:
        ph, _ = sentence_to_phonemes(s, g2p_instance=G2P)
        _phoneme_cache[s] = ph
    ph = _phoneme_cache[s]
    # class ids in LOGIT_TO_PHONEME order; SIL maps to the ' | ' class at index 40
    return [40 if p == "SIL" else LOGIT_TO_PHONEME.index(p) for p in ph], ph


def convert_mat(mat_path, out_h5, session, split):
    m = loadmat(mat_path)
    n = m["sentenceText"].shape[0]
    blocks = m["blockIdx"].ravel().astype(int)

    feats, phones_per_trial = [], []
    for i in range(n):
        sp = m["spikePow"][0, i].astype(np.float32)
        tx = m["tx1"][0, i].astype(np.float32)
        feats.append(np.concatenate([sp, tx], axis=1))
        ids, ph = phoneme_ids(str(m["sentenceText"][i]))
        assert all(v is not None for v in ids), f"unknown phoneme in trial {i}: {ph}"
        phones_per_trial.append(ids)

    # blockwise z-scoring (readme hint): stats over all trials of the same block
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
            ids = phones_per_trial[i]
            assert len(ids) <= MAX_SEQ
            seq[: len(ids)] = ids
            g.create_dataset("seq_class_ids", data=seq)
            sent = str(m["sentenceText"][i]).strip()
            tr = np.zeros(MAX_SEQ, dtype=np.int32)
            codes = [ord(c) for c in sent]
            tr[: len(codes)] = codes
            g.create_dataset("transcription", data=tr)
            g.attrs["n_time_steps"] = feats[i].shape[0]
            g.attrs["seq_len"] = len(ids)
            g.attrs["sentence_label"] = sent
            g.attrs["session"] = session
            g.attrs["block_num"] = int(blocks[i])
            g.attrs["trial_num"] = i
            csv_rows.append((session, int(blocks[i]), len(ids), split))
    print(f"{out_h5}: {n} trials, {len(uniq)} blocks, "
          f"{sum(len(p) for p in phones_per_trial)} phonemes", flush=True)
    return csv_rows, n


def main():
    os.makedirs(OUT, exist_ok=True)
    all_rows = []
    totals = {}
    for part, split in [("train", "Train"), ("test", "Val")]:
        mats = sorted(glob.glob(os.path.join(SRC, part, "*.mat")))
        fname = "data_train.hdf5" if part == "train" else "data_val.hdf5"
        tot = 0
        for mp in mats:
            session = os.path.basename(mp).replace(".mat", "")
            sdir = os.path.join(OUT, session)
            os.makedirs(sdir, exist_ok=True)
            rows, n = convert_mat(mp, os.path.join(sdir, fname), session, split)
            all_rows.extend(rows)
            tot += n
        totals[part] = tot

    # description csv: Date,Post-implant day,Block number,Number of sentences,Corpus,Split
    import csv as _csv
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["Date", "Post-implant day", "Block number",
                    "Number of sentences", "Corpus", "Split"])
        for session, blk, nph, split in all_rows:
            date = "-".join(session.split(".")[1:])
            w.writerow([date, 0, blk, nph, "CompetitionData", split])
    print(f"\ntotals: {totals} | sentences phonemized: {len(_phoneme_cache)}")
    print(f"csv: {CSV_OUT}")


if __name__ == "__main__":
    main()
