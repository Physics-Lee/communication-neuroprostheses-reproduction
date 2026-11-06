# -*- coding: utf-8 -*-
"""Scan librosa mel configurations over the ALREADY synthesized day195
pcm pairs in tmp_mel/ (no TF re-inference needed) to see how much of the
0.646-vs-0.83 gap is mel-parameter choice. Paper says only "40
mel-frequency bands"; configurations tried: htk vs slaney norm, three
n_fft values, plus a wider fmin. Reports per-config mean band Pearson."""
import glob
import os
import re

import numpy as np


def mel_bands(pcm, sr=16000, n_mels=40, n_fft=512, htk=False, fmin=0.0):
    import librosa
    m = librosa.feature.melspectrogram(
        y=pcm.astype(np.float32) / 32768.0, sr=sr, n_fft=n_fft,
        hop_length=160, n_mels=n_mels, htk=htk, fmin=fmin)
    return librosa.power_to_db(m, ref=1.0)


def main():
    work = ("./20250612_wairagkar/"
            "tmp_mel")
    days = {}
    for f in sorted(glob.glob(os.path.join(work, "*_pred_*.pcm"))):
        m = re.match(r"(t15_day\d+)_pred_(\d+)\.pcm", os.path.basename(f))
        if not m:
            continue
        day, tr = m.group(1), m.group(2)
        days.setdefault(day, []).append(int(tr))
    out = {}
    for day, trs in days.items():
        for cfg_name, kw in [
                ("slaney_fft512 (tier-2 default)",
                 dict(n_fft=512, htk=False)),
                ("htk_fft512", dict(n_fft=512, htk=True)),
                ("htk_fft400", dict(n_fft=400, htk=True)),
                ("htk_fft1024", dict(n_fft=1024, htk=True)),
                ("slaney_fft1024", dict(n_fft=1024, htk=False)),
                ("htk_fft512_fmin40", dict(n_fft=512, htk=True, fmin=40.0))]:
            rs = []
            for tr in trs:
                try:
                    wp = np.fromfile(
                        os.path.join(work, f"{day}_pred_{tr}.pcm"),
                        dtype=np.int16)
                    wt = np.fromfile(
                        os.path.join(work, f"{day}_tgt_{tr}.pcm"),
                        dtype=np.int16)
                except OSError:
                    continue
                mp = mel_bands(wp, **kw)
                mt = mel_bands(wt, **kw)
                nb = min(mp.shape[1], mt.shape[1])
                br = []
                for b in range(mp.shape[0]):
                    x, y = mp[b, :nb], mt[b, :nb]
                    if np.std(x) < 1e-9 or np.std(y) < 1e-9:
                        continue
                    br.append(np.corrcoef(x, y)[0, 1])
                if br:
                    rs.append(float(np.mean(br)))
            key = (day, cfg_name)
            out[f"{day} {cfg_name}"] = round(float(np.mean(rs)), 3)
            print(f"{day} {cfg_name}: {out[f'{day} {cfg_name}']:.3f} "
                  f"(n={len(rs)})", flush=True)
    with open("./20250612_wairagkar/"
              "result/mel_param_scan.json", "w", encoding="utf-8") as f:
        import json
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
