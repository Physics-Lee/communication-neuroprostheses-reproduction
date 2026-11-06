# `20250612_wairagkar` — Wairagkar et al., *Nature* 2025 (Instantaneous Voice Synthesis, T15)

- **Paper**: Wairagkar, M. et al. An instantaneous voice-synthesis neuroprosthesis. *Nature* **642**, 417–424 (2025-06-12). [doi:10.1038/s41586-025-09127-3](https://doi.org/10.1038/s41586-025-09127-3)
- **Dataset & Released Weights**: Dryad [`doi:10.5061/dryad.dncjsxm85`](https://doi.org/10.5061/dryad.dncjsxm85)
- **Upstream Code**: [`Neuroprosthetics-Lab/brain-to-voice-2025`](https://github.com/Neuroprosthetics-Lab/brain-to-voice-2025)

## Consumer-PC Reproduction Pipeline
- **`code/tier1_official_model_inference.py`**: Runs the authors' released Keras `.h5` causal neural-to-vocoder encoder weights in 1.5 GB VRAM.
- **`code/tier2_mel_alignment.py` & `code/tier2_mel_full_trial.py`**: Trains a causal 40-bin Mel-spectrogram encoder from scratch on T15 neural features with dynamic time-warping (DTW) target alignment.

## Reproduction Summary (`results/`)
- **Mel-Spectrogram Pearson $r$**: Day-195 $r = 0.686$ (spectral distortion $1 - r = 31.4\%$, multi-day mean $r = 0.649$) vs. paper 3× RTX 3090 + RTX A6000 anchor $r = 0.83$ ($1 - r = 17.0\%$).
