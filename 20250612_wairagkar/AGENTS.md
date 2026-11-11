# AGENTS.md — `20250612_wairagkar/` (Wairagkar et al., *Nature* 2025 Instantaneous Voice Synthesis Reproduction)

This directory reproduces the acoustic synthesis stage of **Wairagkar et al., *Nature* 642, 417–424 (2025)** (instantaneous brain-to-voice neuroprosthesis in participant T15; Dryad DOI: `10.5061/dryad.2280gb64f`). Unlike discrete text-decoding pipelines, this study is a **continuous neural-to-audio synthesis pipeline (open-vocabulary acoustic synthesis without an n-gram/LLM text decoder)**: a causal neural encoder maps motor-cortical features at 10 ms frame steps into 40-dimensional LPCNet / Mel acoustic features, which drive a neural vocoder.

We compress the upstream **3× RTX 3090 (24 GB) + RTX A6000 (48 GB)** multi-GPU workflow into **1.5 GB GPU VRAM and 1.8 GB host RAM** (streaming window inference + single-GPU causal 40-bin Mel encoder training), achieving a held-out Day-195 Mel-spectrogram Pearson correlation of **$r = 0.686$** (spectral distortion $1 - r = 31.4\%$, multi-day mean $r = 0.649$; upstream multi-GPU anchor $r = 0.83, 1 - r = 17.0\%$).

---

## What Not to Do Here (Conventions & Guardrails)

1. **Do NOT fabricate or expect a discrete language-model WER metric for this directory**:
   - Our reproduction of `20250612_wairagkar` evaluates continuous acoustic synthesis (**Mel-spectrogram Pearson $r$** and spectral distortion **$1 - r$** in Panel e of `reproduction_benchmark.yaml`). The paper's 43.8% WER comes from human listener transcription ratings; therefore, Panel (f) (`panel_f_repro_full_wer`) intentionally sets `ours: null, paper: null` with note `WER n/r (voice; see e)`.
2. **Prefer `tier2_mel_full_trial.py` over `tier2_mel_alignment.py` for from-scratch training**:
   - `tier2_mel_alignment.py` synthesizes voiced chunks separately and concatenates them, introducing boundary splice artifacts. `tier2_mel_full_trial.py` synthesizes the entire trial waveform (including silences) and masks out silent frames in the Mel domain afterwards, matching the paper's evaluation protocol cleanly.
3. **Keras `.h5` weight loading**:
   - `tier1_official_model_inference.py` loads the authors' released Keras `.h5` causal encoders using `tensorflow==2.14.0` / `keras==2.14.0` inside `bci-repro`.

---

## Directory Contents & File Roles

```text
20250612_wairagkar/
├── README.md                                  # Study summary (1.5 GB VRAM causal Mel synthesis & Pearson r)
├── code/
│   ├── README.md                              # Tier 1 / Tier 2 pipeline order & script guide
│   ├── tier1_official_model_inference.py      # Tier 1: Runs released Keras .h5 causal encoders on the held-out day (1.5 GB VRAM)
│   ├── diag_align.py                          # Diagnostic: scans frame lags between predicted and target features per trial
│   ├── tier2_mel_alignment.py                 # Tier 2 (chunked): DTW target alignment + causal 40-bin Mel encoder training
│   ├── tier2_mel_full_trial.py                # Tier 2 (canonical full-trial): whole-trial synthesis + Mel-domain silent-frame drop
│   └── mel_param_scan.py                      # Sweeps librosa Mel filterbank parameters (htk/slaney, n_fft, fmin) to bound r sensitivity
└── results/
    ├── tier1_official_model_inference.json    # Tier 1 official checkpoint correlation results
    ├── tier2_mel_alignment.json               # Tier 2 chunked synthesis correlation results
    ├── tier2_mel_full_trial.json              # Tier 2 canonical full-trial synthesis results (Day-195 r = 0.686, mean r = 0.649)
    └── mel_param_scan.json                    # Mel filterbank parameter sweep results
```

---

## Quick Inspection & Execution

* **Inspect archived correlation metrics (zero external download)**:
  ```bash
  python -c "import json; print(json.dumps(json.load(open('20250612_wairagkar/results/tier2_mel_full_trial.json')), indent=2))"
  ```
* **Run inference or training (requires Dryad dataset `doi:10.5061/dryad.2280gb64f`)**:
  ```bash
  conda activate bci-repro
  python 20250612_wairagkar/code/tier1_official_model_inference.py
  python 20250612_wairagkar/code/tier2_mel_full_trial.py
  ```
