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

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "shared"))
from study_paths import study_path

import argparse
import glob
import os

import h5py
import numpy as np
from scipy.io import loadmat

sys.path.insert(0, study_path("20240815_card", 'open_source_code_for_reference/nejm-brain-to-text'))                # nejm_b2txt_utils
sys.path.insert(0, study_path("20240815_card", 'open_source_code_for_reference/nejm-brain-to-text/model_training')) # evaluate_model_helpers
from nejm_b2txt_utils.general_utils import sentence_to_phonemes  # noqa: E402
from evaluate_model_helpers import LOGIT_TO_PHONEME                # noqa: E402
from g2p_en import G2p                                             # noqa: E402

SRC = study_path("20230831_willett", 'data/extracted/competitionData')
OUT = study_path("20230831_willett", 'data/processed/hdf5')
CSV_OUT = study_path("20230831_willett", 'data/processed/session_description.csv')
MAX_SEQ = 500

G2P = G2p()
_phoneme_cache = {}


def phoneme_ids(sentence):
    """Convert a sentence string into CTC phoneme class ids.

    Uses the repo's own sentence_to_phonemes() (g2p_en) with a module-level
    cache so repeated sentences are grapheme-to-phoneme converted only once.
    Class ids follow the LOGIT_TO_PHONEME order used by the GRU decoder;
    the SIL token maps to the ' | ' word-separator class at index 40.

    Parameters
    ----------
    sentence : str
        Raw sentence text (surrounding whitespace is stripped).

    Returns
    -------
    tuple(list of int, list of str)
        (ids, phonemes): class ids in LOGIT_TO_PHONEME order (SIL -> 40)
        and the phoneme sequence returned by sentence_to_phonemes.
    """
    s = sentence.strip()
    if s not in _phoneme_cache:
        ph, _ = sentence_to_phonemes(s, g2p_instance=G2P)
        _phoneme_cache[s] = ph
    ph = _phoneme_cache[s]
    # class ids in LOGIT_TO_PHONEME order; SIL maps to the ' | ' class at index 40
    return [40 if p == "SIL" else LOGIT_TO_PHONEME.index(p) for p in ph], ph


def convert_mat(mat_path, out_h5, session, split):
    """Convert one day's competitionData .mat file into the hdf5 trial layout.

    Concatenates spikePow and tx1 into (T, 512) float32 features, z-scores
    features per blockIdx across all trials of the block (readme advice),
    phonemizes each sentence via phoneme_ids(), and writes one hdf5 group
    per trial with input_features / seq_class_ids / transcription datasets
    plus n_time_steps, seq_len, sentence_label, session, block_num and
    trial_num attributes.

    Parameters
    ----------
    mat_path : str
        Path to the source .mat file (one recording day/session).
    out_h5 : str
        Output hdf5 path, e.g. <out>/<session>/data_train.hdf5.
    session : str
        Session name derived from the .mat filename; stored as the
        ``session`` attribute of every trial group.
    split : str
        Split label recorded for the description csv ("Train" or "Val").

    Returns
    -------
    tuple(list of tuple, int)
        (csv_rows, n): one (session, block_num, n_phonemes, split) tuple
        per trial and the number of trials converted.
    """
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
    """Convert all train/test competitionData days and write the corpus csv.

    Iterates the train/ and test/ .mat files under SRC, writes per-session
    data_train.hdf5 / data_val.hdf5 files under OUT via convert_mat(), and
    emits btot24_description.csv (Date, Post-implant day, Block number,
    Number of sentences, Corpus, Split) summarizing every trial.
    competitionHoldOut/ is intentionally skipped (no labels).
    """
    parser = argparse.ArgumentParser(description="Convert the Willett 2023 competition MAT files.")
    parser.add_argument("--session", help="Only convert this session, for example t12.2022.05.19")
    parser.add_argument("--split", choices=("train", "test"), help="Only convert this source split")
    parser.add_argument("--output-dir", type=Path, help="Write into a new directory")
    args = parser.parse_args()
    if (args.session or args.split) and args.output_dir is None:
        parser.error("--session and --split require a new --output-dir")

    partitions = [("train", "Train"), ("test", "Val")]
    inputs = {
        part: [
            mat for mat in sorted(glob.glob(os.path.join(SRC, part, "*.mat")))
            if args.session is None or Path(mat).stem == args.session
        ]
        for part, _ in partitions
        if args.split is None or part == args.split
    }
    if not any(inputs.values()):
        raise FileNotFoundError(
            f"No competition MAT files match the requested session and split under {SRC}. "
            "See studies/20230831_willett/tutorial.ipynb for data setup."
        )
    out = args.output_dir or Path(OUT)
    description = out / "session_description.csv" if args.output_dir else Path(CSV_OUT)
    out.mkdir(parents=True, exist_ok=args.output_dir is None)
    all_rows = []
    totals = {}
    for part, split in partitions:
        if part not in inputs:
            continue
        fname = "data_train.hdf5" if part == "train" else "data_val.hdf5"
        tot = 0
        for mp in inputs[part]:
            session = os.path.basename(mp).replace(".mat", "")
            sdir = os.path.join(out, session)
            os.makedirs(sdir, exist_ok=True)
            rows, n = convert_mat(mp, os.path.join(sdir, fname), session, split)
            all_rows.extend(rows)
            tot += n
        totals[part] = tot

    # description csv: Date,Post-implant day,Block number,Number of sentences,Corpus,Split
    import csv as _csv
    with description.open("w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["Date", "Post-implant day", "Block number",
                    "Number of sentences", "Corpus", "Split"])
        for session, blk, nph, split in all_rows:
            date = "-".join(session.split(".")[1:])
            w.writerow([date, 0, blk, nph, "CompetitionData", split])
    print(f"\ntotals: {totals} | sentences phonemized: {len(_phoneme_cache)}")
    print(f"csv: {description}")


if __name__ == "__main__":
    main()
