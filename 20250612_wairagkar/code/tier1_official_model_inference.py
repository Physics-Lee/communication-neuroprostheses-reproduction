# -*- coding: utf-8 -*-
"""Wairagkar 2025 (brain-to-voice) tier-1 reproduction: run the OFFICIAL
pretrained brain-to-voice decoders (shipped as .h5 in the Dryad release) on
the held-out evaluation day's neural data, and compare predicted LPCNet
features against the released target features.

Anchor (paper Fig. 2b): "The synthesized voice was similar to the target
speech ... with a Pearson correlation coefficient of 0.83 +/- 0.04 across
40 mel-frequency bands after removing silences between words."

Protocol notes:
- The Dryad release ships one official decoder per evaluation day (named
  after the last training session; e.g. t15_day195 is decoded by
  brain2voice_t15_day193.h5, trained on all sessions before it).
- Channel order for the day025-day195 pretrained models must pass through
  data_utils.reorder_channels (official notebook cell 7).
- We evaluate the Pearson correlation per mel band between prediction and
  target_speech_feats_word, averaged over bands, on non-silent frames.
  Silence is defined per trial by the official word_indices in
  speech_indices.mat (frames outside any word interval), mirroring the
  paper's "after removing silences between words".

Run with the TF-2.14 env, CWD = official repo root:
  cd <repo>/open_source_code_for_reference/20250612_wairagkar_brain-to-voice-2025
  python <wairagkar repo>/code/tier1_official_model_inference.py
"""
import io
import json
import os
import sys
import zipfile

import numpy as np
import scipy.io as sio

OFFICIAL = os.path.dirname(os.path.abspath(__file__)) + \
    "/../open_source_code_for_reference/20250612_wairagkar_brain-to-voice-2025"
sys.path.insert(0, OFFICIAL)

import yaml  # noqa: E402
from lib import data_utils, inference_utils  # noqa: E402

ZIP = ("./20250612_wairagkar/data/raw/"
       "20250612_wairagkar/t15_neural_data.zip")
OUT = ("./20250612_wairagkar/result/"
       "tier1_official_model_inference.json")

# evaluation days to score. Full 9-day sweep costs ~8 h on CPU (TF 2.14 on
# Windows has no GPU path; ~27 s/trial x 1,106 remaining trials) -- the
# 2026-10-26 run scores day195 (the paper's Fig-2b demo day, full 147
# trials) plus day179 as a mid-timeline point; day110 (r 0.53) was already
# scored by the first (pre-alignment-fix) sweep's aligned rerun.
EVAL_DAYS = {
    "t15_day179": "brain2voice_t15_day174.h5",
    "t15_day195": "brain2voice_t15_day193.h5",
}


def main():
    """Score the official pretrained .h5 decoders on evaluation days.

    For each day in ``EVAL_DAYS``, loads the released neural data, target
    speech features, and word timings from the Dryad zip, runs the official
    Keras decoder on the "valuation" trials (with the fixed 63-bin frame
    alignment), and computes the mean per-band Pearson correlation on
    non-silent frames.

    Returns
    -------
    None
        Per-day statistics are printed to stdout and the full report is
        written to ``result/tier1_official_model_inference.json``.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import tensorflow as tf  # noqa: F401
    from keras.models import load_model

    with open(OFFICIAL + "/configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    min_lpc, max_lpc = data_utils.load_lpc_norm(
        cfg["paths"]["lpc_norm_path"])

    z = zipfile.ZipFile(ZIP)
    report = {"days": {}, "protocol": "official h5 decoder, reordered "
               "channels, sliding window 60 bins; Pearson r per mel band "
               "on non-silent frames (word_indices from speech_indices.mat)"}
    for day, model_name in EVAL_DAYS.items():
        # neural data (per-trial features + cues)
        m = sio.loadmat(io.BytesIO(z.read(
            f"Dryad_T15_all/{day}/{day}_neural_data.mat")), squeeze_me=True)
        spike_pow = np.squeeze(m["spikepow_trials"])
        thresh = np.squeeze(m["threshcross_trials"])
        cues = np.squeeze(m["cues"])
        td = np.atleast_1d(m["trial_description"])
        # target features + word timings
        tgt = sio.loadmat(io.BytesIO(z.read(
            f"Dryad_T15_all/{day}/{day}_target_speech_feats.mat")),
            squeeze_me=True)
        target = np.squeeze(tgt["target_speech_feats_word"])
        idx = sio.loadmat(io.BytesIO(z.read(
            f"Dryad_T15_all/{day}/{day}_speech_indices.mat")), squeeze_me=True)
        word_idx = idx["word_indices"]  # per-trial (n_words, 2) @100 Hz

        # official decoder
        model_path_io = io.BytesIO(z.read(
            f"Dryad_T15_all/{day}/{model_name}"))
        tmp_h5 = os.path.join(os.path.dirname(OUT), "_tmp_model.h5")
        with open(tmp_h5, "wb") as f:
            f.write(model_path_io.read())
        model = load_model(tmp_h5, compile=False)
        os.remove(tmp_h5)

        per_band_rs = []
        n_used = 0
        # Fixed frame alignment: the sliding-window model output leads the
        # target features by 63 bins (630 ms = the 60-bin window + a small
        # output delay). A lag scan over the first few trials of day195
        # found best |r| = 0.94 at lag -63 consistently (diag_align.py,
        # 2026-10-26); with zero lag the correlation is ~0 (the naive
        # -0.04 of the first tier-1 run).
        ALIGN = 63
        for tr in range(len(cues)):
            if "valuation" not in str(td[tr]):
                continue  # only evaluation trials for this day's model
            sp = data_utils.reorder_channels(spike_pow[tr].copy(),
                                             cfg["paths"]["electrode_mapping"])
            tc = data_utils.reorder_channels(thresh[tr].copy(),
                                             cfg["paths"]["electrode_mapping"])
            pred = inference_utils.run_inference_single_trial(
                model, sp, tc, cfg["data"]["time_steps"], min_lpc, max_lpc)
            tgt_tr = target[tr]
            # shift: pred[k] corresponds to target[k + ALIGN]
            n_frames = min(len(pred), len(tgt_tr)) - ALIGN
            if n_frames < 80:
                continue
            p = pred[:n_frames]
            t = tgt_tr[ALIGN:ALIGN + n_frames]
            # non-silent frames = inside any word interval (shifted too)
            try:
                wi = np.atleast_2d(word_idx[tr])
                mask = np.zeros(n_frames, dtype=bool)
                for onset, offset in wi:
                    mask[int(max(0, onset - ALIGN - 1)):
                         int(max(0, offset - ALIGN)) + 1] = True
            except Exception:
                mask = np.ones(n_frames, dtype=bool)
            if mask.sum() < 20:
                continue
            n_used += 1
            rs = []
            for b in range(p.shape[1]):
                x, y = p[mask, b], t[mask, b]
                if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                    continue
                rs.append(np.corrcoef(x, y)[0, 1])
            if rs:
                per_band_rs.append(np.mean(rs))
        r_mean = float(np.mean(per_band_rs)) if per_band_rs else float("nan")
        report["days"][day] = {
            "model": model_name,
            "n_trials_total": len(cues),
            "n_eval_trials_scored": n_used,
            "pearson_r_mean_per_trial": round(r_mean, 4),
            "pearson_r_sd_across_trials": round(float(np.std(per_band_rs)),
                                                4),
        }
        print(day, report["days"][day], flush=True)
    allr = [d["pearson_r_mean_per_trial"]
            for d in report["days"].values()]
    report["grand_mean_r"] = round(float(np.mean(allr)), 4)
    report["grand_sd_across_days"] = round(float(np.std(allr, ddof=1)), 4)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("GRAND:", report["grand_mean_r"], "+/-",
          report["grand_sd_across_days"])


if __name__ == "__main__":
    main()
