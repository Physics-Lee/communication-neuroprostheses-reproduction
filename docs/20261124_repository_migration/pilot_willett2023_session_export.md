# Pilot report: Willett 2023 session export

2026-11-24

The first common session file was exported for Willett et al. 2023 (speech, T12)
and used to repeat the archived greedy evaluation of one recording day.

## What was exported

| Item | Value |
| --- | --- |
| Session | `t12.2022.05.19` |
| Session file | `studies/20230831_willett/data/processed/t12/dryad_x69p8czpq_2022-05-19.h5` (75.7 MiB) |
| Trials | 200: 180 from the `train` partition (blocks 5–11, 14, 15) and 20 from the `test` partition (block 16). The release has no `competitionHoldOut` file for this day. |
| Neural features | `spike_pow`, `tx`, `tx2`, `tx3`, `tx4`, each `(frames, 256)` with the released dtype, 20 ms bins |
| Targets | `sentence`, `phonemes` (41 CTC class ids), `transcription` (character codes) |
| Provenance | every trial names its publisher `.mat` file and its position in that file |

The exporter is `studies/20230831_willett/code/export_sessions.py`. It stores
the features as released and leaves out the block-wise z-scoring that
`convert_btot24.py` applies when it builds the trainer input.

## Commands

Run from the repository root with `bci-repro` active.

```text
python studies/20230831_willett/code/export_sessions.py --session t12.2022.05.19
python shared/tools/validate_session.py studies/20230831_willett/data/processed/t12/dryad_x69p8czpq_2022-05-19.h5 --source trials
python studies/20230831_willett/code/eval_session_file.py studies/20230831_willett/data/processed/t12/dryad_x69p8czpq_2022-05-19.h5 studies/20230831_willett/data/20230616_willett_BtoT24/run_60k --partition test --compare studies/20230831_willett/results/20261124_import_btot24/test_partition_greedy_per_trial.csv --trainer-input studies/20230831_willett/data/derived/dryad_x69p8czpq/hdf5/trainer_input/t12.2022.05.19 --output-dir <new-results-directory>
python studies/20230831_willett/code/build_split_manifest.py
```

The evaluation output of the pilot run is in
`studies/20230831_willett/results/20261124_dryad_x69p8czpq__hdf5__ctc_greedy__session_file_eval/`
(`trials.csv`, `summary.json`, `run.yaml`), with its console log under
`studies/20230831_willett/log/20261124_dryad_x69p8czpq__hdf5__ctc_greedy__session_file_eval/`.

## Results

| Check | Result |
| --- | --- |
| Source trace | 200 of 200 trials were re-read from the two publisher `.mat` files and are identical to the stored data. |
| Trainer input | 200 of 200 trials: the z-scored features rebuilt from the session file are bit-identical to `data_train.hdf5` / `data_val.hdf5`, and sentence, phoneme ids, and transcription are identical. |
| Inference | Recording-day index 3 (fourth day in sorted order, the index `eval_btot24.py` passes). 20 test trials, 245 errors over 637 reference phonemes, 38.46 % PER. |
| Archive comparison | 20 of 20 trials identical row by row to `test_partition_greedy_per_trial.csv`: block, sentence, reference and predicted phoneme strings, edit distance, and reference length. Archived total for the day: 245 / 637. |
| Split manifest | `data/splits/dryad_x69p8czpq__source_partitions.csv` lists the partition and role of every exported trial. The 20 validation trials of this day equal the archived evaluation rows in content and order. |

The acceptance target of the overnight plan was the total of 245 errors over 637
reference phonemes with trial-by-trial agreement. Both hold.

## Finding: the trainer input depends on memory layout

`convert_btot24.py` z-scores each block over a stack of MATLAB arrays, which
are column-major. NumPy sums a column-major `(frames, 512)` stack in a
different order than a row-major one, so the float32 means and deviations
differ in their last bits. Channels with a small standard deviation then move
by up to 6.1e-4 after the division.

The first version of `eval_session_file.py` stacked row-major arrays read from
the session file. Its predictions were still identical row by row, but its
features were not those of the trainer input. The script now converts the
arrays to column-major layout before stacking, and the features match bit for
bit. A public-command conversion run into a fresh directory was used to
confirm that the trainer input on disk equals what `convert_btot24.py`
produces today.

Anyone who rebuilds the trainer input from a session file needs to keep that
layout.

## Limits of this pilot

- Word-level decoding was not repeated from the session file. The archived
  WFST and language-model decoders read the logits dump of
  `dump_btot24_logits.py`, which still reads the trainer input.
- `tx2`, `tx3`, and `tx4` are stored under their publisher names. The release
  readme in this checkout does not describe how they differ from `tx1`.
- How many sessions are exported is recorded in the
  [export inventory](session_export_inventory.md), not here.
