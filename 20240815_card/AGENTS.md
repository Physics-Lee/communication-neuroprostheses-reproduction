# AGENTS.md — `20240815_card/` (Card et al., *NEJM* 2024 Speech Neuroprosthesis & Rapid Calibration Reproduction)

This directory reproduces **Card et al., *N. Engl. J. Med.* 391, 609–618 (2024)** (high-accuracy speech neuroprosthesis in participant T15 with ALS across 45 sessions; Dryad DOI: `10.5061/dryad.dncjsxm85`). It covers two main tracks:
1. **Full 45-Session Training, BiGRU Upgrade & Day-Index Routing Fix**: Trained and decoded on a single 16 GB GPU (13.0 GB peak VRAM) + 19.6 GB `mmap` 5-gram WFST, achieving **10.19% greedy PER**, **7.02% single-seed WER** (`p4_fixed_opt_fx.tsv`), and **6.25% 41-session pooled BiGRU WER** (paper: **6.7%**).
2. **Rapid Day-1 Calibration & Within-Session Online Adaptation (`calib_*` suite)**: Reproduces the sentence-by-sentence Day-1 calibration curve and Table S5 online fine-tuning dynamics (`copy` / `keep` / `eye` arms).

---

## What Not to Do Here (Conventions & Guardrails)

1. **Always use `dump_logits_fixed.py` when exporting validation logits (Day-Index routing bugfix)**:
   - Earlier scripts (`dump_t15_logits.py` and `dump_p4_logits.py`) used the loop enumeration index instead of the session's true `dayIdx` when iterating across evaluation splits with skipped days, misrouting the day-specific input layers. `dump_logits_fixed.py` fixes this routing and produces the canonical `results/decode_outputs/p4_fixed_opt_fx.tsv` (the earlier scripts are kept only to document the bugfix provenance).
2. **Do NOT duplicate `opt_rescore.py`, `wer_from_tsv.py`, or C++ WFST decoders inside `code/tools/`**:
   - Those byte-identical copies were consolidated into [`../shared/tools/`](../shared/tools/) (see `../docs/CLEANUP_AUDIT.md` §4). Keep only T15-specific helpers in `code/tools/`.
3. **Never alter row order in `results/decode_outputs/val_greedy_per_trial.csv` or `p4_fixed_opt_fx.tsv`**:
   - Root script `build_session_per_vs_wer.py` joins these two files positionally to compute the 41 session points for `Card 2024 (T15)` in Panel (i) of the benchmark figure.

---

## Directory Contents & File Roles

```text
20240815_card/
├── README.md                                          # Study summary (memory mitigations, Day-1 calibration, PER/WER)
├── code/
│   ├── README.md                                      # Pipeline topology & complete 30+ script reference table
│   ├── train_btot25.py                                # Full 45-session GRUDecoder training from scratch (run_120k)
│   ├── train_btot25_step2.py                          # Rapid-calibration Step 2 baseline: 40 training days (last 5 val days excluded)
│   ├── train_bigrun.py                                # Session-subset ablations (T2a/T2b) & bidirectional GRU (T4 BiGRUDecoder) trainer
│   ├── calib_models.py                                # BiGRUDecoder definition (drop-in bidirectional upgrade of official GRUDecoder)
│   ├── eval_val_per.py / eval_btot25_fromscratch.py / eval_p4_retrain.py # Greedy CTC PER evaluators across checkpoints
│   ├── prep_rapid_calibration.py / run_zeroshot_all.py / smoke_init_checkpoint.py # Rapid-calibration dataset slicers & smoke tests
│   ├── plot_rapid_calibration_curve.py                # Plots the Day-1 trial-budget rapid calibration curve
│   ├── calib_online_sim.py                            # Offline simulation of Table S5 sentence-by-sentence online fine-tuning
│   ├── calib_p0_*.py / calib_perday_ft_probe.py / calib_tta_probe.py / calib_t2a_eval.py # Calibration probes & test-time augmentation
│   ├── calib_assemble_20261014.py / calib_compare_first32_20261015.py # P1/P2 comparison tables & Table S3 32-session alignment
│   ├── dump_logits_fixed.py                           # [CANONICAL] Corrected day-index logit dumper feeding p4_fixed_* products
│   ├── dump_logits_ensemble.py                        # P4 x T2a log-probability ensemble dumper
│   ├── dump_t15_logits.py / dump_p4_logits.py         # Historical dumpers preserved for pre-fix provenance
│   ├── wer_eval_card.py                               # Paper-style WER evaluator (bootstrap CI + per-session/corpus breakdown)
│   ├── plot_btot25.py / plot_card24_*.py / plot_paper_fig2_panels.py # T15 timeline, corpus, and scope comparison figures
│   └── tools/
│       ├── export_npy_t15.py                          # Converts HDF5 logits to per-trial .npy + meta.tsv for C++ WFST decoding
│       ├── prep_rescore_for_wereval.py                # Uppercases opt_rescore hypothesis column for wer_eval_card.py
│       ├── p4_track.py                                # Live training log parser comparing P4 vs. run_120k trajectories
│       └── *.sh / p4_runbook_20261013.md              # WSL C++ WFST decode drivers and P4 runbook
└── results/
    ├── decode_outputs/
    │   ├── val_greedy_per_trial.csv                   # Per-trial greedy PER across validation sessions (pooled PER = 10.19%)
    │   ├── p4_fixed_final_full.tsv                    # Day-index-fixed 125k 5-gram WFST 1-best hypotheses
    │   ├── p4_fixed_opt_fx.tsv                        # Day-index-fixed 5-gram + OPT-6.7B rescored hypotheses (WER = 7.02%)
    │   └── p4_opt_fx.tsv                              # Pre-fix rescoring baseline
    ├── calib_*.csv / p4_vs_run120k_trajectory.csv     # Rapid-calibration and first-32-session comparison tables
    └── logs/                                          # Training, calibration simulation, and rescoring logs
```

---

## Tensor Shapes & Quick Verification

* **Input neural features**: Dryad `hdf5_data_final` is natively in `nejm-brain-to-text` HDF5 format (no conversion needed), with **512 features per 20 ms frame** (256 channels from four 64-electrode arrays × [threshold crossings + spike-band power]).
* **Zero-download verification commands (run from repository root)**:
  ```bash
  # 1) Verify canonical day-index-fixed 5-gram + OPT-6.7B WER
  python shared/tools/wer_from_tsv.py 20240815_card/results/decode_outputs/p4_fixed_opt_fx.tsv

  # 2) Run paper-style corpus/session WER breakdown with bootstrap CI
  python 20240815_card/code/wer_eval_card.py 20240815_card/results/opt_fx_card.upperhyp.tsv
  ```
