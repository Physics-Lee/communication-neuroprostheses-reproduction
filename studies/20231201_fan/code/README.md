# `studies/20231201_fan/code/` — Code Guide

Reproduces the **Fan et al., *NeurIPS* 2023** CORP study (403-day plug-and-play
handwriting BCI, participant T5): the paper's key claim is that *online unsupervised
pseudo-label recalibration* keeps a decoder usable for a year. The reproduction is
organized in tiers.
Study-level context: [`../README.md`](../README.md).

## Public commands

Run these commands from the repository root in the environment required by the study. Use `--help` on an entry point for its arguments.

| Public command | Role |
| --- | --- |
| [`export_sessions.py`](export_sessions.py) | Export one session to the common session format. |
| [`infer.py`](infer.py) | Evaluate inference from the frozen CORP seed model. |
| [`evaluate.py`](evaluate.py) | Replay CORP recalibration and score matched held-out trials. |

Use [`record_study_run.py`](../../../shared/tools/record_study_run.py) to name scored outputs from a completed command. See the [eight-study naming map](../../../docs/20261124_repository_migration/naming_map.md) for the result and data paths.

The export command writes `data/processed/t5/<release>_<date>.h5` for one session. `python shared/tools/validate_session.py <file> --source trials` re-reads every trial from the publisher's files. The format is defined in [`shared/session_format.py`](../../../shared/session_format.py).

## Pipeline order

```text
Tier 1 (no model):  rescore_official_outputs.py
      └─ rescore the OFFICIAL released online decode outputs (recal vs no-recal arms)
Tier 2 (frozen):    tier2_seed_model_inference.py
      └─ run the released pretrained seed RNN on all 10 follow-up sessions
Tier 3 (the CORP dynamic, an intentional version chain):
      tier3_corp_sim_official_labels.py (v4: official lm_decoded pseudo-labels;
              also the shared library imported by v5–v8)
        └─ tier3_corp_sim_v5.py  (500-step CORP-scale budget, input layer only)
             └─ tier3_corp_sim_v6.py  (full decoder trainable — README headline:
                    mean CER 6.20% vs paper 6.16%)
                  ├─ tier3_corp_sim_v7.py  (leakage-free final-6-trials holdout)
                  └─ tier3_corp_sim_v8.py  (worst-case self-greedy labels test)
      diag_train_step.py / diag_train_budget.py
      └─ one-step / budget diagnostics that justified v5's 500-step choice
```

The v-numbering is an **experiment log, not duplication**: each version states in its
module docstring what the previous one got wrong and what changed. v6 is the
README-documented result; v7/v8 are the methodological follow-ups (their JSONs are in
`../results/`). The retired v1 (greedy self-labels, retracted claim) was removed —
see `docs/20261108_cleanup_audit/CLEANUP_AUDIT.md`.

## File-by-file

| File | Role |
|---|---|
| `export_sessions.py` | Writes one session to the common session format (`data/processed/t5/dryad_hqbzkh1p6_<date>.h5`): the released features of every trial of the seed partitions or of both online arms, with the TFRecord shard, the example position, and the row of the session's sentence table. It reads the sentence tables from the ZIP. |
| `build_split_manifest.py` | Writes `data/splits/dryad_hqbzkh1p6__paired_v7_holdout.csv` from the exported session files and checks the holdout trials against the archived evaluation list. |
| `rescore_official_outputs.py` | Tier-1: rescores the official online decoded outputs shipped in the Dryad release (independent of running any model). |
| `tier2_seed_model_inference.py` | Tier-2: evaluates the frozen pretrained seed RNN across the 10 follow-up sessions (shows the drift the paper's CORP fixes). |
| `tier3_corp_sim_official_labels.py` | Tier-3 v4 + library: session-by-session online recalibration simulation using the official `lm_decoded` pseudo-labels; provides `make_args`, `gather_trials`, `load_official_labels`, `forward_logits`, `greedy_decode`, `ctc_loss`, … to v5–v8. |
| `tier3_corp_sim_v5.py` | v5: adds the CORP-scale update budget (500 steps/session, 70/30 pseudo/replay); provides `cer` used by v6–v8. |
| `tier3_corp_sim_v6.py` | v6: unfreezes the full decoder (matching the paper's update scope) — the headline reproduction run. |
| `tier3_corp_sim_v7.py` | v7: same protocol with the final 6 trials of every session held out of the training pool (leakage-free post measure). |
| `tier3_corp_sim_v8.py` | v8: re-tests the label axis with self-greedy pseudo-labels (robustness worst case). |
| `diag_train_step.py` | One-step gradient-flow diagnostic on the worst session. |
| `diag_train_budget.py` | Session-3-only update-budget experiment that motivated v5's 500 steps. |

## Matched-trial WER

`reproduce_paired_wer.py` replays the v7 adaptation and exports the online, greedy, and character-LM transcripts for the final six held-out trials per session. Adaptation uses the released seed checkpoint and recorded online pseudo-labels from the training portion of each session. Evaluation uses the same released reference for both outputs. [`corp_char_lm.py`](corp_char_lm.py) contains the v9 character 5-gram decoder, which fits only seed-session text.

The [archived transcripts](../results/20261124_import_paired_v7/paired_v7_trial_transcripts.csv) and [run summary](../results/20261124_import_paired_v7/paired_v7_summary.json) cover 60 trials across ten sessions. The [paired benchmark tables](../../../figures/docs/20261124_evaluation_protocol/figure2cd.md#fan-corp-held-out-replay) report the beam-search result alongside the released online output on those trials.

Run from the repository root with the original Fan reproduction checkout available. The CUDA option converts the restored Keras GRU to PyTorch and checks its logits against Keras before training.

```bash
python studies/20231201_fan/code/reproduce_paired_wer.py --study /path/to/fan-checkout --output /path/to/new-run --backend torch-cuda
```
