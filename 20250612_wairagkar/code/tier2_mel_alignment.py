# -*- coding: utf-8 -*-
"""Wairagkar 2025 tier-2: mel-spectrogram-layer alignment of the paper's
0.83 +/- 0.04 anchor.

Pipeline (paper Fig 2b caliber): synthesize the decoded voice from the
predicted LPCNet features with the OFFICIAL LPCNet vocoder (xiph lpcnet_demo,
built 2026-10-26 in WSL), synthesize the target voice from the released
target features the same way, compute 40-band mel spectrograms of both
(16 kHz), remove inter-word silence frames via the released word_indices,
and correlate band-by-band (Pearson per band, mean over bands, mean over
trials).

Samples N_TRIALS evaluation trials of day195 (the paper's demo day) plus
day179 as a second point. TF inference on CPU (27 s/trial) dominates the
runtime; vocoding and mel are seconds.

Writes result/tier2_mel_alignment.json.
"""
import io
import json
import os
import subprocess
import sys
import zipfile

import numpy as np
import scipy.io as sio

OFFICIAL = ("./20250612_wairagkar/"
            "open_source_code_for_reference/20250612_wairagkar_brain-to-voice-2025")
sys.path.insert(0, OFFICIAL)

import yaml  # noqa: E402
from lib import data_utils, inference_utils  # noqa: E402

ZIP = ("./20250612_wairagkar/data/raw/"
       "20250612_wairagkar/t15_neural_data.zip")
OUT = ("./20250612_wairagkar/result/"
       "tier2_mel_alignment.json")
WORK = "./20250612_wairagkar/tmp_mel"
WSL_SYNTH = "cd ~/lpcnet && ./lpcnet_demo -synthesis"

ALIGN = 63  # frame alignment found by the tier-1 diagnosis
N_TRIALS = 20
DAYS = {"t15_day179": "brain2voice_t15_day174.h5",
        "t15_day195": "brain2voice_t15_day193.h5"}


def mel_bands(pcm, sr=16000, n_mels=40):
    """Convert int16 PCM audio to a log-power Mel spectrogram.

    Parameters
    ----------
    pcm : numpy.ndarray
        Raw int16 waveform samples.
    sr : int, optional
        Sampling rate in Hz. Default 16000.
    n_mels : int, optional
        Number of Mel frequency bands. Default 40.

    Returns
    -------
    numpy.ndarray
        Shape (n_mels, n_frames) dB-scale Mel spectrogram
        (``librosa.power_to_db`` with ``ref=1.0``).
    """
    import librosa
    m = librosa.feature.melspectrogram(y=pcm.astype(np.float32) / 32768.0,
                                       sr=sr, n_fft=512, hop_length=160,
                                       n_mels=n_mels)
    return librosa.power_to_db(m, ref=1.0)


def synth(feat, tag):
    """feat (T,20) float32 -> waveform via WSL lpcnet_demo.

    Writes the frame sequence to ``<WORK>/<tag>.f32``, invokes the official
    xiph ``lpcnet_demo -synthesis`` binary inside WSL, and reads back the
    synthesized int16 PCM.

    Parameters
    ----------
    feat : numpy.ndarray
        Shape (T, 20) float32 LPCNet feature frames to vocode.
    tag : str
        Base filename (without extension) for the intermediate ``.f32``
        and output ``.pcm`` files inside ``WORK``.

    Returns
    -------
    numpy.ndarray
        Int16 synthesized waveform samples.

    Raises
    ------
    subprocess.CalledProcessError
        If the WSL ``lpcnet_demo`` subprocess exits non-zero.
    """
    f32 = os.path.join(WORK, f"{tag}.f32")
    pcm = os.path.join(WORK, f"{tag}.pcm")
    feat = np.asarray(feat, dtype="<f4")
    with open(f32, "wb") as f:
        f.write(feat.tobytes())
    wsl_f32 = subprocess.run(["wslpath", "-w", f32], capture_output=True,
                             text=True).stdout if False else None
    # path translation: D:/... -> /mnt/d/...
    p = f32.replace("D:/", "/mnt/d/").replace("\\", "/")
    po = pcm.replace("D:/", "/mnt/d/").replace("\\", "/")
    subprocess.run(["wsl", "-e", "bash", "-c",
                    f"{WSL_SYNTH} {p} {po}"], check=True,
                   capture_output=True)
    return np.fromfile(pcm, dtype=np.int16)


def main():
    """Run the tier-2 audio-level Mel alignment evaluation.

    For each day in ``DAYS``, samples up to ``N_TRIALS`` evaluation trials,
    runs the official Keras decoder, vocodes the predicted and target
    features with LPCNet (after silence removal on the feature grid), and
    correlates the resulting Mel spectrograms band-by-band.

    Returns
    -------
    None
        Per-trial and per-day correlations are printed to stdout and the
        report is written to ``result/tier2_mel_alignment.json``.
    """
    os.makedirs(WORK, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import tensorflow as tf  # noqa: F401
    from keras.models import load_model

    with open(OFFICIAL + "/configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    min_lpc, max_lpc = data_utils.load_lpc_norm(cfg["paths"]["lpc_norm_path"])
    z = zipfile.ZipFile(ZIP)
    report = {"days": {}, "protocol": {
        "vocoder": "xiph LPCNet lpcnet_demo (official pre-trained weights)",
        "mel": "librosa 40 mel bands, n_fft 512, hop 160 (10 ms), power->dB",
        "silence": "frames outside word intervals (word_indices, shifted "
                   f"-{ALIGN} to match model output timing)",
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
            # silence mask on the 10-ms feature grid
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
            # vocode only the voiced span (silence removal happens at the
            # feature layer, mirroring "after removing silences")
            pf = p[mask]
            tf_ = tt[mask]
            try:
                wav_p = synth(pf, f"{day}_pred_{tr}")
                wav_t = synth(tf_, f"{day}_tgt_{tr}")
            except Exception as e:  # noqa: BLE001
                print(tr, "synth fail", e, flush=True)
                continue
            mp = mel_bands(wav_p)
            mt = mel_bands(wav_t)
            nb = min(mp.shape[1], mt.shape[1])
            band_rs = []
            for b in range(mp.shape[0]):
                x, y = mp[b, :nb], mt[b, :nb]
                if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                    continue
                band_rs.append(np.corrcoef(x, y)[0, 1])
            if band_rs:
                rs.append(float(np.mean(band_rs)))
            done += 1
            print(f"{day} trial {tr}: r={rs[-1] if rs else float('nan'):.3f} "
                  f"({done}/{N_TRIALS})", flush=True)
        report["days"][day] = {
            "model": model_name, "n_scored": len(rs),
            "mel_band_r_mean": round(float(np.mean(rs)), 3),
            "mel_band_r_sd_trials": round(float(np.std(rs)), 3)}
        print(day, report["days"][day], flush=True)
    allr = [d["mel_band_r_mean"] for d in report["days"].values()]
    report["grand_mean"] = round(float(np.mean(allr)), 3)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)
    print("GRAND:", report["grand_mean"])


if __name__ == "__main__":
    main()
