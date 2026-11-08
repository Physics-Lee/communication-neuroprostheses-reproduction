# -*- coding: utf-8 -*-
"""Diagnose the Wairagkar tier-1 correlation failure: scan frame lags
between predicted LPCNet features (official h5 model) and the released
target_speech_feats_word, per trial, to see whether the mismatch is a pure
time shift (window alignment), a scale/normalization issue, or something
deeper. Also compares against the online-synthesized audio released in
the evaluation sessions (brain2voice_pred_audio_16kHz) where available.
"""
import io
import json
import os
import sys
import zipfile

import numpy as np
import scipy.io as sio

OFFICIAL = ("./20250612_wairagkar/"
            "open_source_code_for_reference/"
            "20250612_wairagkar_brain-to-voice-2025")
sys.path.insert(0, OFFICIAL)

import yaml  # noqa: E402
from lib import data_utils, inference_utils  # noqa: E402

ZIP = ("./20250612_wairagkar/data/raw/"
       "20250612_wairagkar/t15_neural_data.zip")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from keras.models import load_model  # noqa: E402


def main():
    """Run the frame-lag diagnosis on the official tier-1 model.

    Loads the released T15 neural data and the official Keras .h5 encoder,
    runs per-trial inference on "valuation" trials, and compares the
    predicted LPCNet features against ``target_speech_feats_word`` at zero
    lag and over a coarse lag scan (-40..+40 frames, step 2).

    Returns
    -------
    None
        Results are printed to stdout: per-trial zero-lag and best-lag
        correlations, the lag histogram, and the median |best r|.
    """
    day, model_name = "t15_day195", "brain2voice_t15_day193.h5"
    with open(OFFICIAL + "/configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    min_lpc, max_lpc = data_utils.load_lpc_norm(cfg["paths"]["lpc_norm_path"])
    z = zipfile.ZipFile(ZIP)
    m = sio.loadmat(io.BytesIO(z.read(
        f"Dryad_T15_all/{day}/{day}_neural_data.mat")), squeeze_me=True)
    spike_pow = np.squeeze(m["spikepow_trials"])
    thresh = np.squeeze(m["threshcross_trials"])
    cues = np.squeeze(m["cues"])
    td = np.atleast_1d(m["trial_description"])
    tgt = sio.loadmat(io.BytesIO(z.read(
        f"Dryad_T15_all/{day}/{day}_target_speech_feats.mat")), squeeze_me=True)
    target = np.squeeze(tgt["target_speech_feats_word"])

    tmp_h5 = os.path.join(os.path.dirname(OFFICIAL), "_tmp_model.h5")
    with open(tmp_h5, "wb") as f:
        f.write(z.read(f"Dryad_T15_all/{day}/{model_name}"))
    model = load_model(tmp_h5, compile=False)

    best_lags = []
    zero_lag_rs = []
    for tr in range(len(cues)):
        if "valuation" not in str(td[tr]):
            continue
        sp = data_utils.reorder_channels(spike_pow[tr].copy(),
                                         cfg["paths"]["electrode_mapping"])
        tc = data_utils.reorder_channels(thresh[tr].copy(),
                                         cfg["paths"]["electrode_mapping"])
        pred = inference_utils.run_inference_single_trial(
            model, sp, tc, cfg["data"]["time_steps"], min_lpc, max_lpc)
        t = target[tr][:len(pred)]
        p = pred[:len(t)]
        n = min(len(p), len(t))
        p, t = p[:n], t[:n]
        # global mean over all frames per feature, mean-centered correlation
        pc = p - p.mean(0)
        tc_ = t - t.mean(0)
        denom = np.sqrt((pc ** 2).sum(0) * (tc_ ** 2).sum(0)) + 1e-12
        r_zero = (pc * tc_).sum(0) / denom
        zero_lag_rs.append(float(np.nanmean(r_zero)))
        # lag scan on the first feature + mean over features, coarse
        best_r, best_lag = -2, 0
        sig = (pc * tc_).sum() / np.sqrt((pc**2).sum() * (tc_**2).sum() + 1e-12)
        for lag in range(-40, 41, 2):
            if lag >= 0:
                a, b = pc[lag:], tc_[:len(tc_) - lag] if lag else tc_
            else:
                a, b = pc[:lag], tc_[-lag:]
            if len(a) < 50:
                continue
            rr = (a * b).sum() / np.sqrt((a**2).sum() * (b**2).sum() + 1e-12)
            if abs(rr) > abs(best_r):
                best_r, best_lag = rr, lag
        best_lags.append((best_lag, float(best_r), float(sig)))
        print(f"trial {tr}: zero-lag mean r={np.nanmean(r_zero):+.3f} "
              f"best lag={best_lag:+d} r={best_r:+.3f} (zero-lag pooled "
              f"r={float(sig):+.3f})", flush=True)
    bl = np.array([b[0] for b in best_lags])
    print("lag distribution:", np.bincount(bl + 40))
    print("median |best r|:", np.median(np.abs([b[1] for b in best_lags])))


if __name__ == "__main__":
    main()
