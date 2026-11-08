# `20240815_card/code/` — Code Guide

Reproduces the **Card et al., *NEJM* 2024** speech neuroprosthesis study (participant
T15, ALS) on the official `nejm-brain-to-text` GRU stack, with two focuses: (1) the
full decode/rescore pipeline over the released dataset, and (2) a systematic
investigation of the paper's **rapid day-1 calibration** mechanism.
Study-level context: [`../README.md`](../README.md).

## Pipeline order

```text
(Dryad hdf5_data_final — native trainer layout, no conversion needed)
   └─ train_btot25.py ──→ run_120k from-scratch checkpoint
        └─ eval_val_per.py ──→ per-trial greedy PER CSV (official-pretrained eval too)
             └─ dump_t15_logits.py ──→ run_120k per-trial logits hdf5
   └─ train_btot25_step2.py ──→ 40-day baseline (5 val days excluded)
        └─ smoke_init_checkpoint.py ──→ init-from-checkpoint mechanism smoke test
        └─ prep_rapid_calibration.py ──→ ft_<day>_N trial-budget datasets
             └─ ../shared/finetune_chain/train_finetune.py ──→ rapid-calibration curve
                  └─ plot_rapid_calibration_curve.py ──→ the curve figure
   └─ train_bigrun.py ──→ T2a/T2b/T4 subset & BiGRU runs (uses calib_models.py)
Calibration investigation (calib_*):
   calib_online_sim.py ──→ offline simulation of the paper's sentence-by-sentence
        online fine-tuning (arms: copy/keep/eye; runners in tools/*.sh)
   calib_p0_wer_baseline.py / calib_p0_loss_scale.py ──→ P0 baselines
   calib_perday_ft_probe.py / calib_tta_probe.py ──→ T1/T7 probes
   calib_t2a_eval.py ──→ zero-shot + cross-fit per-day eval
   calib_assemble_20261014.py ──→ P1/P2-lite comparison table
   calib_compare_first32_20261015.py ──→ paper-online vs our-offline pairing
   calib_plot_trajectories_20261014.py / calib_plot_session_release_20261015.py ──→ figures
Decode chain (WSL side):
   dump_p4_logits.py / dump_logits_fixed.py / dump_logits_ensemble.py
        └─ tools/export_npy_t15.py ──→ trials/*.npy + meta.tsv
             └─ shared/tools/wenet_lattice.cc (+ tools/*.sh runners) ──→ final/nbest TSVs
                  └─ shared/tools/opt_rescore.py ──→ OPT/Qwen rescoring
                       └─ tools/prep_rescore_for_wereval.py → wer_eval_card.py
Figures: plot_btot25.py, plot_card24_figs.py, plot_card24_scopes.py,
         plot_card24_session_timeline_paperbench.py, plot_paper_fig2_panels.py
```

## File-by-file

### Training / evaluation

| File | Role |
|---|---|
| `train_btot25.py` | Trains the GRUDecoder from scratch on the full 45-session dataset with the official recipe/optimizer. |
| `train_btot25_step2.py` | Step-2 baseline of the rapid-calibration plan: 40 training days, last 5 val days excluded. |
| `train_bigrun.py` | Scratch training with session subsets and/or the BiGRU architecture (T2a 44-sess, T2b 39-sess, T4 biGRU); monkeypatches `calib_models.BiGRUDecoder` into the trainer. |
| `calib_models.py` | `BiGRUDecoder`: Yoon-2026-style bidirectional upgrade of the official GRUDecoder (same CTC interface). |
| `eval_val_per.py` | Phase-1 evaluation: pretrained or self-trained RNN inference + greedy CTC PER (mirrors official `evaluate_model.py` semantics). Produces `val_greedy_per_trial.csv`. |
| `eval_btot25_fromscratch.py` | Final eval of the from-scratch `run_120k` checkpoint (same semantics as `eval_val_per.py`). |
| `eval_p4_retrain.py` | Final eval of the P4 retrain checkpoint (`run_p4_20261013`). |
| `run_zeroshot_all.py` | Driver: zero-shot `eval_val_per.py` for all 5 excluded days. |
| `smoke_init_checkpoint.py` | Step-3 mechanism check: `init_from_checkpoint` loads and continues training sensibly. |
| `prep_rapid_calibration.py` | Builds per-day, per-budget fine-tune datasets (`ft_<day>_N`) for the rapid-calibration Step 3. |
| `plot_rapid_calibration_curve.py` | Rebuilt rapid-calibration figure (zero-shot + fine-tune trajectories + paper reference). |

### Calibration investigation (`calib_*`)

| File | Role |
|---|---|
| `calib_online_sim.py` | Core simulation of the paper's Table-S5 sentence-by-sentence online fine-tuning, run offline in file order (arms: `copy`/`keep`/`eye`). |
| `calib_p0_wer_baseline.py` | P0.1: per-day 5-gram and +OPT WER for the six 2025 follow-up sessions from the P4 endpoint products. |
| `calib_p0_loss_scale.py` | P0.2/P0.3: CTC loss-scale calibration on good vs bad days (predicts fine-tune neutrality). |
| `calib_perday_ft_probe.py` | T1 probe: dedicated same-day fine-tune from the P4 soup (does the val split move?). |
| `calib_tta_probe.py` | T7 probe: noise-perturbation test-time augmentation (pseudo-ensembling) on P4. |
| `calib_t2a_eval.py` | T2a evaluation: zero-shot + cross-fit per-day fine-tune on the fully-excluded 2025.01.10 (day-layer remapping). |
| `calib_assemble_20261014.py` | Assembles the P1/P2-lite comparison table (static vs simulated arms; honest per-trial online PER). |
| `calib_compare_first32_20261015.py` | Pairs our offline per-session results with the paper's Table-S3 online numbers over the first 32 sessions. |
| `calib_plot_trajectories_20261014.py` | Fig-S14-style within-session PER trajectories for all calibration arms. |
| `calib_plot_session_release_20261015.py` | Paper-84 vs released-45 session timeline figure. |

### Logits dumping & decode chain

| File | Role |
|---|---|
| `dump_t15_logits.py` | Dumps per-trial log-probs for the archive `run_120k` val set. |
| `dump_p4_logits.py` | Same for the P4 retrain run (never touches the archive file). |
| `dump_logits_fixed.py` | Corrected day-index dumper (fixes the day-layer routing bug in the two above; feeds the `p4_fixed_*` decode products). |
| `dump_logits_ensemble.py` | Averages P4 × T2a log-probs into a single logits hdf5 for the decode chain. |
| `tools/export_npy_t15.py` | hdf5 → per-trial `.npy` + `meta.tsv` for the WSL C++ `wenet_lattice`/`tlg_viterbi` decoders. |
| `tools/*.sh` | One-shot WSL runners for each decode (P2 reproduction, P4 endpoint, fixed-index, ensemble, biGRU). Byte-verified against archived outputs. |
| `tools/prep_rescore_for_wereval.py` | Uppercases the hyp column of `opt_rescore` TSVs so `wer_eval_card.py` scores them correctly. |
| `tools/p4_track.py` | Live tracker comparing the P4 from-scratch run against `run_120k` (parses trainer logs; safe to re-run). |
| `wer_eval_card.py` | Paper-style WER evaluation of word-decode TSVs (pooled + bootstrap CI, session/corpus breakdowns). |
| `tools/p4_runbook_20261013.md` | Runbook for the P4 endpoint decode. |

### Figures

| File | Role |
|---|---|
| `plot_btot25.py` | Phase-1 figures: PER by corpus + 20-month per-session timeline. |
| `plot_card24_figs.py` | Card-only adaptation of `shared/plots/plot_figs_separate.py` panels C/D/E from repo-pinned inputs. |
| `plot_card24_scopes.py` | WER "口径" (reporting-scope) comparison: ours vs paper anchors. |
| `plot_card24_session_timeline_paperbench.py` | Per-session PER timeline overlaid with the paper's reported numbers. |
| `plot_paper_fig2_panels.py` | Paper-Fig2-style WER panels rebuilt from the byte-verified P2 decode outputs. |

Note: the duplicate local copies of `opt_rescore.py`, `wer_from_tsv.py`, `tlg_viterbi.cc`,
`wenet_lattice.cc`, `glog_shim.h` were removed in favor of `shared/tools/` (byte-identical);
see `docs/CLEANUP_AUDIT.md`.
