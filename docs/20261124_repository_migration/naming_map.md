# Study naming map

Paths in the tables are relative to the named study directory. The public commands call the existing implementations and retain their study-specific arguments. Historical paths still resolve through the shared path helper when using a data tree in the previous layout.

## Public command names

| Study | Prepare data | Session export | Train | Infer | Decode | Evaluate |
| --- | --- | --- | --- | --- | --- | --- |
| `20210512_willett` | `step3_make_synthetic.py` → `prepare_data.py` | `export_sessions.py` | `torch_port/train.py` → `train.py` | `torch_port/infer.py` → `infer.py` | `step6_decode_bigram.py` → `decode.py` | `step5_eval_from_errcounts.py` → `evaluate.py` |
| `20230831_willett` | `convert_btot24.py` → `prepare_data.py` | `export_sessions.py` | `train_btot24.py` → `train.py` | `dump_btot24_logits.py` → `infer.py` | `decode_words_btot24_v2.py` → `decode.py` | `eval_btot24.py` → `evaluate.py` |
| `20231201_fan` | — | `export_sessions.py` | — | `tier2_seed_model_inference.py` → `infer.py` | — | `reproduce_paired_wer.py` → `evaluate.py` |
| `20240815_card` | — | `export_sessions.py` | `train_btot25.py` → `train.py` | `dump_logits_fixed.py` → `infer.py` | `../../../shared/tools/decode_wfst.py` → `decode.py` | `wer_eval_card.py` → `evaluate.py` |
| `20250612_wairagkar` | — | `export_sessions.py` | — | `tier1_official_model_inference.py` → `infer.py` | — | `tier2_mel_full_trial.py` → `evaluate.py` |
| `20250821_kunz` | `convert_kunz.py` → `prepare_data.py` | `export_sessions.py` | `train_kunz.py` → `train.py` | `dump_kunz_logits.py` → `infer.py` | `../../../shared/tools/decode_wfst.py` → `decode.py` | `eval_kunz.py` → `evaluate.py` |
| `20260316_jude` | `window_W4_convert.py` → `prepare_data.py` | `export_sessions.py` | `pytorch_port/train_multisession.py` → `train.py` | `pytorch_port/dump_logits.py` → `infer.py` | `pytorch_port/decode_multisession_lm.py` → `decode.py` | `g1_recompute_cpm.py` → `evaluate.py` |
| `20260428_jude` | `convert_jude.py` → `prepare_data.py` | `export_sessions.py` | `train_jude.py` → `train.py` | `dump_jude_logits.py` → `infer.py` | `../../../shared/tools/decode_wfst.py` → `decode.py` | `eval_jude.py` → `evaluate.py` |

The command paths are under `code/`. Fan’s evaluation command includes the paired replay adaptation. Wairagkar reads the released ZIP directly, and Card reads the released HDF5 directly. Every study writes common session files with `export_sessions.py`; see [Common session files](README.md#common-session-files).
## Data paths

Paths are relative to each study. The common rules are in the [repository layout](README.md#repository-layout). Every row lists a current input path. Publisher archive members keep their names after extraction.

| Study | Dataset key used in paths | Current primary input |
| --- | --- | --- |
| `20210512_willett` | `dryad_wh70rxwmv` | `data/extracted/dryad_wh70rxwmv/handwritingBCIData/` |
| `20230831_willett` | `dryad_x69p8czpq` | `data/extracted/dryad_x69p8czpq/competitionData/` |
| `20231201_fan` | `dryad_hqbzkh1p6` | `data/extracted/dryad_hqbzkh1p6/CORP_data_release/` |
| `20240815_card` | `dryad_dncjsxm85` | `data/extracted/dryad_dncjsxm85/hdf5_data_final/` |
| `20250612_wairagkar` | `dryad_2280gb64f` | `data/raw/dryad_2280gb64f/dryad_2280gb64f__t15_neural_data.zip` |
| `20250821_kunz` | `dryad_gf1vhhn1j` | `data/extracted/dryad_gf1vhhn1j/sentenceData/` |
| `20260316_jude` | `dryad_cz8w9gjjk` | `data/extracted/dryad_cz8w9gjjk/Data/` |
| `20260428_jude` | `dryad_vq83bk481` | `data/extracted/dryad_vq83bk481/Data/` |

| Study | Other current paths using the shared names |
| --- | --- |
| Willett 2021 | `data/derived/dryad_wh70rxwmv/npy/synthetic_sentences/<partition>/`, `data/derived/dryad_wh70rxwmv/pkl/synthetic_snippets/<partition>/` |
| Willett 2023 | `data/derived/dryad_x69p8czpq/hdf5/trainer_input/`, `data/derived/dryad_x69p8czpq/hdf5/dryad_x69p8czpq__session_description.csv`, `data/language_models/kenlm_3gram/`, `data/language_models/wfst_5gram/` |
| Fan 2023 | `data/raw/dryad_hqbzkh1p6/dryad_hqbzkh1p6__CORP_data_release.zip` |
| Card 2024 | `data/raw/dryad_dncjsxm85/dryad_dncjsxm85__t15_copyTask.pkl`, `data/extracted/dryad_dncjsxm85/t15_copyTaskData_description.csv`, `data/checkpoints/dryad_dncjsxm85/t15_pretrained_rnn_baseline/` |
| Kunz 2025 | `data/derived/dryad_gf1vhhn1j/hdf5/trainer_input/`, `data/derived/dryad_gf1vhhn1j/hdf5/t16_holdout/`, `data/derived/dryad_gf1vhhn1j/hdf5/dryad_gf1vhhn1j__session_description.csv` |
| Jude 2026a | `data/derived/dryad_cz8w9gjjk/tfrecord/typing_20ms/<participant>/<session>/`, `data/derived/dryad_cz8w9gjjk/npz/typing_20ms/<participant>/<session>/`, `data/language_models/wfst_5gram/` |
| Jude 2026b | `data/derived/dryad_vq83bk481/hdf5/trainer_input/`, `data/derived/dryad_vq83bk481/hdf5/dryad_vq83bk481__session_description.csv` |

Seven studies have a trial list named `data/splits/<release>__archived_evaluation.csv`. Wairagkar's listener test is recovered in `data/splits/dryad_2280gb64f__listener_benchmark.csv`. Each study also has split lists keyed by `trial_id` for the trials of its session files, described in its `data/splits/README.md`. Common session files go to `data/processed/<participant>/<release>_<session>.h5`; the [export inventory](session_export_inventory.md) lists which ones exist. Historical dataset locations still resolve through the path helper. Historical model workspaces and diagnostic paths are listed in the study code guides when used.

## Result names

The dated runtime packages replace local `data/<old-study-name>/run_*` workspaces. The `runtime/` directory holds trainer-native files, and `log/<run_id>/training_log` holds the training log. Existing code can resolve both through `runtime_path` or the original alias passed to `study_path`.

| Study | Migrated runs |
| --- | --- |
| Willett 2023 | `results/20260831_dryad_x69p8czpq__hdf5__ctc_greedy__run_60k/` |
| Card 2024 | `results/20260902_dryad_dncjsxm85__hdf5__ctc_greedy__run_120k/`, `results/20260903_dryad_dncjsxm85__hdf5__ctc_greedy__excl5_40k/`, and `results/20261013_dryad_dncjsxm85__hdf5__ctc_greedy__p4/` |
| Kunz 2025 | Twelve HDF5 training runs for the participant arms and holdout folds, plus `results/20261001_dryad_gf1vhhn1j__npy__wfst_5gram__decoder_sweep/` |
| Jude 2026b | `results/20260901_dryad_vq83bk481__hdf5__ctc_greedy__run_25k/` and `results/20261002_dryad_vq83bk481__npy__wfst_5gram__decoder_sweep/` |

For a new run, use `results/<yyyymmdd>_<release>__<format>__<decoder>__<experiment>/` with `run.yaml` for provenance, `trials.csv` for scored trials, and `summary.json` for aggregate results. Where a historical run exposes an aggregate only, use `summary.json` when repackaging that run. The published figures and tutorials continue to cite their exact archived filenames.

The [result recording command](../../shared/tools/record_study_run.py) can build this path from the source manifest, `--input-format`, `--decoder`, and `--label`. For example, from the repository root:

```text
python shared/tools/record_study_run.py 20230831_willett --input-format hdf5 --decoder ctc_greedy --label checkpoint_eval --split test --trials studies/20230831_willett/results/20261124_import_btot24/test_partition_greedy_per_trial.csv
```

| Study | Representative published run | Source for `trials.csv` | Source for `summary.json` |
| --- | --- | --- | --- |
| `20210512_willett` | `results/20261124_import_handwriting_baselines/` | `m1b_own_decode_HeldOutTrials.csv` | — |
| `20230831_willett` | `results/20261124_import_btot24/` | `test_partition_greedy_per_trial.csv` | — |
| `20231201_fan` | `results/20261124_import_paired_v7/` | `paired_v7_trial_transcripts.csv` | `paired_v7_summary.json` |
| `20240815_card` | `results/20261124_import_p4_archive/` | `p4_fixed_opt_fx.tsv` | — |
| `20250612_wairagkar` | `results/20261124_import_voice_synthesis/` | — | `tier2_mel_full_trial.json` |
| `20250821_kunz` | `results/20261124_import_t12_mixed/` | `val_greedy_per_trial.csv` | — |
| `20260316_jude` | `results/20261124_import_multisession/` | — | `lm_decode_summary.json` |
| `20260428_jude` | `results/20261124_import_lis_decode/` | `val_greedy_per_trial.csv` | — |
