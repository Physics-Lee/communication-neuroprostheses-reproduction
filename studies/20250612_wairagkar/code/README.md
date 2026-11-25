# `studies/20250612_wairagkar/code/` — Code Guide

Reproduces the acoustic stage of the **Wairagkar et al., *Nature* 2025** instantaneous
voice-synthesis neuroprosthesis (participant T15): a causal encoder maps neural
features to 40-bin LPCNet/Mel acoustic features, which a vocoder turns into speech.
Study-level context: [`../README.md`](../README.md).

## Public commands

Run these commands from the repository root in the environment required by the study. Use `--help` on an entry point for its arguments.

| Public command | Role |
| --- | --- |
| [`export_sessions.py`](export_sessions.py) | Export one recording day from the release ZIP to the common session format. |
| [`infer.py`](infer.py) | Run the released neural-to-voice encoder. |
| [`evaluate.py`](evaluate.py) | Evaluate Mel correlations over full synthesized trials. |

Use [`record_study_run.py`](../../../shared/tools/record_study_run.py) to name scored outputs from a completed command. See the [eight-study naming map](../../../docs/20261124_repository_migration/naming_map.md) for the result and data paths.

The export writes `data/processed/t15/dryad_2280gb64f_<day>.h5` for one day. `python shared/tools/validate_session.py <file> --source trials` re-reads every trial from the ZIP, and `python studies/20250612_wairagkar/code/check_session_timing.py <file>` checks that the neural features and the acoustic targets share one time axis. The format is defined in [`shared/session_format.py`](../../../shared/session_format.py).

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
| `export_sessions.py` | Writes one recording day of the release ZIP to the common session format: neural features, acoustic targets, word and syllable timings as released, and the ZIP members each trial came from. |
| `check_session_timing.py` | Checks an exported day without the ZIP: equal frame counts on the neural and acoustic axes, intervals inside their trial, and agreement between word-interval silence and the silent frames of the targets. |
| `build_split_manifest.py` | Writes the evaluation-trial and listener-benchmark lists under `data/splits/` from the release ZIP. |
| `tier1_official_model_inference.py` | Runs the authors' released Keras `.h5` decoders on the held-out day and correlates predicted vs released target features. |
| `diag_align.py` | Failure diagnosis: scans frame lags between predicted and released features per trial (pure time shift vs scale vs deeper mismatch); also compares against the released online audio. |
| `tier2_mel_alignment.py` | Trains a causal 40-bin Mel encoder from scratch with DTW target alignment; synthesizes voiced chunks with the official LPCNet vocoder. |
| `tier2_mel_full_trial.py` | Paper-faithful variant: synthesizes the whole trial (silences included) and drops silent *frames* in the mel domain — removes the splice artifacts of the tier-2 chunking. |
| `mel_param_scan.py` | Sweeps librosa mel configurations (htk/slaney, n_fft, fmin) over already-synthesized audio to bound how much of the 0.65-vs-0.83 correlation gap is parameter choice. |
