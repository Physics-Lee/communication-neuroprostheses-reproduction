# -*- coding: utf-8 -*-
"""Wairagkar 2025 mel-layer alignment, paper-faithful silence handling:
synthesize the WHOLE trial (silences included), compute mel spectrograms
of both waveforms, then drop the silent FRAMES in the mel domain. The
tier-2 version concatenated voiced feature chunks before vocoding,
creating splice artifacts at every silence boundary -- the prime suspect
for the 0.65-vs-0.83 residual.
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
from tier2_mel_alignment import synth, mel_bands  # noqa: E402

ZIP = ("./20250612_wairagkar/data/raw/"
       "20250612_wairagkar/t15_neural_data.zip")
OUT = ("./20250612_wairagkar/result/"
       "tier2_mel_full_trial.json")
WORK = "./20250612_wairagkar/tmp_mel2"
ALIGN = 63
N_TRIALS = 20
DAYS = {"t15_day179": "brain2voice_t15_day174.h5",
        "t15_day195": "brain2voice_t15_day193.h5"}


def main():
    os.makedirs(WORK, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    from keras.models import load_model

    with open(OFFICIAL + "/configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    min_lpc, max_lpc = data_utils.load_lpc_norm(cfg["paths"]["lpc_norm_path"])
    z = zipfile.ZipFile(ZIP)
    report = {"days": {}, "protocol": {
        "v2": "whole-trial synthesis, silence removed at the mel-frame "
              "layer (paper-faithful), vs tier-2's feature-chunk "
              "concatenation",
        "n_trials_per_day": N_TRIALS}}
    for day, model_name in DAYS.items():
        m = sio.loadmat(io.BytesIO(z.read(
            f"Dryad_T15_all/{day}/{day}_neural_data.mat")), squeeze_me=True)
        spike_pow = np.squeeze(m["spikepow_trials"])
        thresh = np.squeeze(m["threshcross_trials"])
        td = np.atleast_1d(m["trial_description"])
        tgt = sio.loadmat(io.BytesIO(z.read(
            f"Dryad_T15_all/{day}/{day}_target_speech_feats.mat")),
            squeeze_me=True)
        target = np.squeeze(tgt["target_speech_feats_word"])
        idx = sio.loadmat(io.BytesIO(z.read(
            f"Dryad_T15_all/{day}/{day}_speech_indices.mat")), squeeze_me=True)
        word_idx = idx["word_indices"]

        tmp_h5 = os.path.join(WORK, "_model.h5")
        with open(tmp_h5, "wb") as f:
            f.write(z.read(f"Dryad_T15_all/{day}/{model_name}"))
        model = load_model(tmp_h5, compile=False)

        rs = []
        done = 0
        for tr in range(len(td)):
            if "valuation" not in str(td[tr]) or done >= N_TRIALS:
                continue
            sp = data_utils.reorder_channels(spike_pow[tr].copy(),
                                             cfg["paths"]["electrode_mapping"])
            tc = data_utils.reorder_channels(thresh[tr].copy(),
                                             cfg["paths"]["electrode_mapping"])
            pred = inference_utils.run_inference_single_trial(
                model, sp, tc, cfg["data"]["time_steps"], min_lpc, max_lpc)
            t = target[tr]
            n = min(len(pred), len(t)) - ALIGN
            if n < 80:
                continue
            p = pred[:n]
            tt = t[ALIGN:ALIGN + n]
            # voiced mask on the 10-ms grid (for MEL-frame dropping)
            try:
                wi = np.atleast_2d(word_idx[tr])
                mask = np.zeros(n, dtype=bool)
                for onset, offset in wi:
                    a = int(max(0, onset - ALIGN))
                    b = int(min(n, offset - ALIGN))
                    if b > a:
                        mask[a:b] = True
            except Exception:
                mask = np.ones(n, dtype=bool)
            if mask.sum() < 30:
                continue
            try:
                wav_p = synth(p, f"{day}_full_pred_{tr}")
                wav_t = synth(tt, f"{day}_full_tgt_{tr}")
            except Exception as e:  # noqa: BLE001
                print(tr, "synth fail", e, flush=True)
                continue
            mp = mel_bands(wav_p)   # hop 160 = 10 ms -> frame k ~ mask[k]
            mt = mel_bands(wav_t)
            nb = min(mp.shape[1], mt.shape[1], len(mask))
            sel = mask[:nb]
            band_rs = []
            for b in range(mp.shape[0]):
                x, y = mp[b, :nb][sel], mt[b, :nb][sel]
                if len(x) < 10 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
                    continue
                band_rs.append(np.corrcoef(x, y)[0, 1])
            if band_rs:
                rs.append(float(np.mean(band_rs)))
            done += 1
            print(f"{day} trial {tr}: r={rs[-1] if rs else float('nan'):.3f}"
                  f" ({done}/{N_TRIALS})", flush=True)
        report["days"][day] = {
            "model": model_name, "n_scored": len(rs),
            "mel_band_r_mean": round(float(np.mean(rs)), 3),
            "mel_band_r_sd_trials": round(float(np.std(rs)), 3)}
        print(day, report["days"][day], flush=True)
    allr = [d["mel_band_r_mean"] for d in report["days"].values()]
    report["grand_mean"] = round(float(np.mean(allr)), 3)
    report["tier2_concat_ref"] = {"day179": 0.532, "day195": 0.646}
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("GRAND:", report["grand_mean"])


if __name__ == "__main__":
    main()
