# `20250612_wairagkar/code/` — Code Guide

Reproduces the acoustic stage of the **Wairagkar et al., *Nature* 2025** instantaneous
voice-synthesis neuroprosthesis (participant T15): a causal encoder maps neural
features to 40-bin LPCNet/Mel acoustic features, which a vocoder turns into speech.
Study-level context: [`../README.md`](../README.md).

## Pipeline order

```text
Tier 1 (official weights):
   tier1_official_model_inference.py
      └─ run released .h5 encoders on the held-out evaluation day (1.5 GB VRAM)
           └─ diag_align.py ── diagnose the correlation failure (lag scan)
Tier 2 (from scratch, paper-faithful):
   tier2_mel_alignment.py ── DTW-aligned mel training (splice-artifact caveat)
   tier2_mel_full_trial.py ── whole-trial synthesis + silent-frame drop (fix)
mel_param_scan.py ── how much of the r-gap is librosa mel configuration?
```

## File-by-file

| File | Role |
|---|---|
| `tier1_official_model_inference.py` | Runs the authors' released Keras `.h5` decoders on the held-out day and correlates predicted vs released target features. |
| `diag_align.py` | Failure diagnosis: scans frame lags between predicted and released features per trial (pure time shift vs scale vs deeper mismatch); also compares against the released online audio. |
| `tier2_mel_alignment.py` | Trains a causal 40-bin Mel encoder from scratch with DTW target alignment; synthesizes voiced chunks with the official LPCNet vocoder. |
| `tier2_mel_full_trial.py` | Paper-faithful variant: synthesizes the whole trial (silences included) and drops silent *frames* in the mel domain — removes the splice artifacts of the tier-2 chunking. |
| `mel_param_scan.py` | Sweeps librosa mel configurations (htk/slaney, n_fft, fmin) over already-synthesized audio to bound how much of the 0.65-vs-0.83 correlation gap is parameter choice. |
