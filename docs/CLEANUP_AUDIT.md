# Repository Cleanup Audit — 2026-11-08

Goal: every student can read this repo and understand what each file does.
Method: audited **all 147 tracked `.py` files** one by one (headers, cross-references,
import graphs, README documentation, results artifacts), then deleted only files that
are *strictly superseded* or *one-shot patch scripts whose side effects are already
archived*. Nothing was deleted on guesswork; each decision below cites its evidence.

**Outcome: 22 redundant `.py` files + 3 duplicated non-`.py` tool files deleted
(147 → 125 `.py` files). All deletions are recoverable via `git checkout`.**

Deletion policy applied (conservative — when in doubt, keep):

1. **Version-superseded**: a later file in the same folder replaces it AND the later
   file's own docstring declares the replacement, or the pipeline driver invokes only
   the later one.
2. **Dead experiment line**: the script feeds a binary/tool that is not in the repo and
   is not referenced by any README.
3. **Applied one-shot patches**: scripts that edit a results table/markdown in place,
   whose output is already committed; re-running them would fail their `assert` anyway.
4. **Byte-identical duplicates**: copies of files that exist canonically in `shared/`.

---

## 1. `20210512_willett/` (Willett 2021, handwriting) — 3 deletions

| Deleted | Evidence |
|---|---|
| `code/pipeline_watchdog.py` | Watchdog **v2**: guarded only the HeldOut*Trials* training phase of one night (2026-10-20). Its own header hands over to v3 ("前置状态…由 v2 交代"). Superseded by v4, which covers the full chain (train → infer → step5 → kaldimat → decode → figure → report). |
| `code/pipeline_watchdog_v3.py` | Watchdog v3 = v4 minus the final `m4_report.py` step (verified line-by-line: identical flow, v4 docstring says "相对 v3：末尾新增 m4_report.py"). Only v4 is referenced (by `m4_report.py`). |
| `code/fig_perday_cer.py` | Generic single-model per-day CER bar plot (`--which official\|ours`). Superseded by `fig_perday_cer_ours.py`, which draws official vs. self-trained side by side for *both* partitions and is the only variant wired into `pipeline_watchdog_v4.py`. No other file imports it. |

**Kept** (all distinct pipeline stages): `step2_relabel` (HMM relabel audit), `step3_make_synthetic` (data augmentation port), `step5_eval_from_errcounts` (metric recomputation + library), `step6_make_kaldimat`, `step6_select_from_kaldioutput` (official n-best selection), `step6_decode_batch` (official-output decode **and** library imported by `step6_decode_ours`), `step6_decode_ours` (self-trained decode), `step6_decode_bigram` (pure-Python lazy FST decoder — the README-documented memory-wall mitigation), `step7_rescore_gpt2`, `forcedAlignmentHMM` (numpy-2 shim imported by spawned workers), `m4_report`, `pipeline_watchdog_v4` (final orchestrator — kept as executable documentation of pipeline order), `fig_perday_cer_ours`, `torch_port/` (model/train/infer/augmentation/sanity_check — distinct roles).

## 2. `20230831_willett/` (Willett 2023, speech) — 3 deletions

| Deleted | Evidence |
|---|---|
| `code/decode_words_btot24.py` | Two-stage decoder v1. `decode_words_btot24_v2.py`'s docstring: "Replaces the earlier two-stage pipeline (decode_words_btot24.py) whose lexicon lookup read lexicon_numbers.txt with tokens.txt ids … all prior pronunciations (and the 31.5% WER) were wrong." README documents only v2. |
| `code/export_npy_trials.py` | Exports trials to `/root/lm/trials` for the `lg_viterbi` WSL decoder. That binary is not in this repo (grep over `.cc/.sh/.md`), and no README mentions it — dead experiment line superseded by the documented `tlg_viterbi.cc` path. |
| `code/wer_sweep.py` | WER diagnostic for the same absent `lg_viterbi` csv outputs. |

**Kept**: `convert_btot24`, `train_btot24`, `eval_btot24`, `dump_btot24_logits`, `decode_words_btot24_v2` (canonical KenLM decoder), `export_trials_npy` (feeds the documented `tlg_viterbi.cc` via `~/tlv`), `plot_willett2023_wer`.

## 3. `20231201_fan/` (Fan 2023, CORP) — 1 deletion

| Deleted | Evidence |
|---|---|
| `code/tier3_corp_sim.py` | CORP simulation **v1** (greedy self-labels). `tier3_corp_sim_v8.py`'s docstring calls v1–v3 "the retracted claim"; no results JSON exists for it (`results/` has JSONs for v4/v5/v6/v7/v8 only); nothing imports it; README documents v6. |

**Kept**: the whole documented ablation chain `tier3_corp_sim_official_labels.py` (v4 — *library* imported by v5+), `_v5` (library: v6/v7/v8 import `cer` from it), `_v6` (README-canonical result, CER 6.20%), `_v7` (leakage-free holdout), `_v8` (self-label robustness), plus `tier2_seed_model_inference`, `rescore_official_outputs`, `diag_train_budget`/`diag_train_step` (diagnostics cited in v5's docstring). The chain is an intentional experiment log, not duplication.

## 4. `20240815_card/` (Card 2024, speech) — 2 `.py` + 3 non-`.py` deletions

| Deleted | Evidence |
|---|---|
| `code/t15_wer_sweep.py` | Same dead `lg_viterbi` line as §2 (docstring: "WER for T15 subset lg_viterbi csv"). |
| `code/tools/opt_rescore.py` | `diff` = byte-identical to `shared/tools/opt_rescore.py`; shared/ is the canonical copy documented in `shared/README.md` and the root README. No card `.sh` invokes the local path (verified). |
| `code/tools/wer_from_tsv.py` | Byte-identical to `shared/tools/wer_from_tsv.py`. |
| `code/tools/tlg_viterbi.cc`, `code/tools/wenet_lattice.cc`, `code/tools/glog_shim.h` | Byte-identical to the copies in `shared/tools/` (README documents the shared ones; decode `.sh` scripts call externally built `~/src_lattice` / `~/bin` binaries, not these local copies). |

**Kept**: every `calib_*` (distinct P0/P1/P2 calibration-plan stages, probes, plots), `calib_models.py` (BiGRU architecture), `train_btot25`/`train_btot25_step2`/`train_bigrun` (distinct training configs), `eval_val_per`/`eval_btot25_fromscratch`/`eval_p4_retrain` (distinct checkpoints; v-files explicitly state "identical semantics, different run"), `dump_t15_logits`/`dump_p4_logits`/`dump_logits_fixed`/`dump_logits_ensemble` (each bound to a different run directory/artifact; `dump_logits_fixed` documents the day-index bugfix lineage), all `plot_*`, `prep_rapid_calibration`, `run_zeroshot_all`, `smoke_init_checkpoint`, `wer_eval_card`, and card-specific tools (`export_npy_t15.py`, `p4_track.py`, `prep_rescore_for_wereval.py` + runners).

## 5. `20250612_wairagkar/` (Wairagkar 2025, voice) — 0 deletions

All six files are distinct: `tier1_official_model_inference` (released-weights inference), `tier2_mel_alignment` (first Mel alignment), `tier2_mel_full_trial` (paper-faithful whole-trial synthesis — fixes tier-2 splice artifacts, not a duplicate), `diag_align` (lag diagnosis), `mel_param_scan` (Mel-parameter ablation).

## 6. `20250821_kunz/` (Kunz 2025, inner speech) — 1 deletion

| Deleted | Evidence |
|---|---|
| `code/inspect_kunz2.py` | Un-documented follow-up scratch inspector for 3 specific T16 files. Its one working feature (correct `decodedSentenceText` extraction via `sent_str`) was **merged into `inspect_kunz.py`** before deletion (the original `inspect_kunz.py` decoded-text branch always printed `None`). |

**Kept**: `convert_kunz`, `train_kunz` (3 training arms), `eval_kunz`, `dump_kunz_logits`, `20261028_gap_decomposition` (README-documented), `make_t16_holdout`/`train_t16_holdout`/`eval_t16_holdout` (holdout experiment family), `plot_ablation_holdout`, `inspect_kunz` (now merged).

## 7. `20260316_jude/` (Jude 2026a, touch typing) — 12 deletions

| Deleted | Evidence |
|---|---|
| `code/nightly_S7_convert_t18_20241204.py` | Hardcoded one-night converter for t18.2024.12.04 blocks 5,6,7. Superseded by the parameterized `window_W4_convert.py` ("任意 T18 会话"), which both `.sh` chain drivers use. |
| `code/nightly_S8_train_m1.py` | One-shot official-TF2.14 train+infer driver (the path the `pytorch_port/README.md` documents as infeasible: 13+ days on CPU). Final archived runs (`results/runs/*_torch`) all come from `pytorch_port/`. Script also imports `open_source_code_for_reference/`, which is not part of this repo. |
| `code/nightly_S8a_fix_report.py`, `code/nightly_S8a_fix_report_v2.py` | One-shot repairs of the transient `result/runs/m1_t18_20241204_smoke` report (that run dir no longer exists in `results/runs/`); v2 explicitly supersedes v1. |
| `code/patch_table_{s9g,g1,s9h,r16,r14,t17,t17final,1205lm}.py` (8 files) | One-off string-replacement patches appending rows to `result/复现对照表.md` (each begins `assert old in s`). All already applied and committed; re-running fails the assert. The table itself remains in `results/`. |
| `code/verify_and_patch_1205.py` | Same category: one-shot verify + patch for the 12.05 session (applied 2026-10-11). |
| `code/pytorch_port/s9a_lm_decode.py` | Single-run 5-gram decode. Superseded by `s9a_lm_decode_multi.py` (same decode loop driven by `s9_runs.json`, loads the TLG once — the only variant the `.sh` chain drivers invoke). |

**Kept**: `pytorch_port/` minus s9a-single (dataset/model/train/dump_logits/export_tfrecord_to_npz/s9a_lm_decode_multi + fidelity-table README), `window_W4_convert` (generic converter), `g1_recompute_cpm` (README-documented free-typing CPM/WER), `g1_scan_blocks` (sentence-copy-block scan — *complementary* block type, produced the 110-cpm finding recorded in the comparison table), all `nightly_*.sh` build scripts (environment/toolchain records), `fix_t17_chain.sh`, `run_t17_remaining.sh`, args yamls.

## 8. `20260428_jude/` (Jude 2026b, intended speech) — 0 deletions

All six files distinct: `inspect_jude` (single .mat survey), `scan_jude_blocks` (all-blocks scan — different granularity), `convert_jude`, `train_jude`, `eval_jude`, `dump_jude_logits`.

## 9. `shared/` — 1 deletion

| Deleted | Evidence |
|---|---|
| `vacation_tlg/vac_opt_rescore_old.py` | Self-declared "old" pre-Qwen snapshot of `opt_rescore.py`: missing the Qwen path, length-aware token-budget batching, and the swapped-variant augmentation fix (verified by diff). Superseded by `shared/tools/opt_rescore.py`. |

**Kept**: `tools/*` (all referenced by READMEs/CI: `wer_from_tsv`, `opt_rescore`, `qwen_rank_probe`, `compare_rescorers`, `parse_train_log`, `pkl_online_metrics`, `extract_four_papers` + C++/sh toolchain), `plots/*` (each produces a *different* supplementary figure — training trajectories, WER spectrum, recording-day timelines, vacation-TLG results — none duplicate the root `render_figure2.py` benchmark figure; inputs are repo-relative), `vacation_tlg/*` scripts (T6/T7 experiment series, each with archived outputs), `finetune_chain/`.

## 10. Root — 0 deletions

`build_session_per_vs_wer.py` and `render_figure2.py` are exercised by the CI workflow (`.github/workflows/verify_benchmark.yml`) — untouched functionally.

---

## Other cleanup performed

- Removed all untracked `__pycache__/` directories (already gitignored; no history impact).
- Merged the working `decodedSentenceText` extractor from `inspect_kunz2.py` into `inspect_kunz.py` (the original branch printed `None` unconditionally).
- Every remaining file now carries a numpy-style module and function docstring (see `docs/` guides); per-study `code/README.md` guides list each script, its role, and pipeline order.
