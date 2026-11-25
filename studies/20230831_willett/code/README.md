# `studies/20230831_willett/code/` — Code Guide

Reproduces the **Willett et al., *Nature* 2023** speech neuroprosthesis (participant
T12): a 5-layer GRU emits CTC phoneme logits from neural features, then word-level
decoding runs either through a 1.46 GB KenLM 3-gram beam search (pure Python, this
folder) or through the C++ 5-gram TLG decoder in `shared/tools/`.
Study-level context: [`../README.md`](../README.md).

## Public commands

Run these commands from the repository root in the environment required by the study. Use `--help` on an entry point for its arguments.

| Public command | Role |
| --- | --- |
| [`prepare_data.py`](prepare_data.py) | Prepare Willett 2023 competition data for training and evaluation. |
| [`export_sessions.py`](export_sessions.py) | Export one recording day to the common session format. |
| [`train.py`](train.py) | Train the Willett speech model. |
| [`infer.py`](infer.py) | Export speech model logits. |
| [`decode.py`](decode.py) | Decode speech logits with the KenLM 3-gram model. |
| [`evaluate.py`](evaluate.py) | Compute greedy phoneme error rates for the test split. |

For a single-session check, run `python studies/20230831_willett/code/prepare_data.py --session t12.2022.05.19 --split test --output-dir <new-directory>`. The evaluation command can write `trials.csv`, `summary.json`, and `run.yaml` with `--output-dir`.

The export command writes `data/processed/t12/dryad_x69p8czpq_<date>.h5` for one day. `python shared/tools/validate_session.py <file> --source trials` re-reads every trial from the publisher's `.mat` files. The format is defined in [`shared/session_format.py`](../../../shared/session_format.py).

The release input is `data/extracted/dryad_x69p8czpq/competitionData/`. A full conversion writes native trainer inputs to `data/derived/dryad_x69p8czpq/hdf5/trainer_input/` and the index to `data/derived/dryad_x69p8czpq/hdf5/dryad_x69p8czpq__session_description.csv`. Paths are relative to the study directory.

Use [`record_study_run.py`](../../../shared/tools/record_study_run.py) to name scored outputs from a completed command. See the [eight-study naming map](../../../docs/20261124_repository_migration/naming_map.md) for the result and data paths.

## Pipeline order

```text
(Dryad competitionData .mat)
   └─ convert_btot24.py ──→ hdf5 in the nejm-brain-to-text layout
        └─ train_btot24.py ──→ GRU checkpoint (FP16, ~3.5 GB VRAM)
             ├─ eval_btot24.py ──→ greedy PER (test partition)
             └─ dump_btot24_logits.py ──→ per-trial CTC log-probs (hdf5)
                  ├─ decode_words_btot24_v2.py ──→ KenLM 3-gram word decode (WSL)
                  │       (supplies word_decode_joint3gram.csv)
                  └─ export_trials_npy.py ──→ per-trial .npy + meta.tsv
                        └─ shared/tools/tlg_viterbi.cc ──→ 5-gram WFST decode (WSL)
                             └─ shared/tools/opt_rescore.py ──→ OPT/Qwen rescoring
plot_willett2023_wer.py ──→ PER-vs-WER reproduction figure
```

## File-by-file

| File | Role |
|---|---|
| `convert_btot24.py` | Converts the released per-day `.mat` files into the nejm-brain-to-text hdf5 layout (spikePow+tx1 features, per-block z-scoring, g2p phoneme targets). |
| `export_sessions.py` | Writes one recording day to the common session format (`data/processed/t12/dryad_x69p8czpq_<date>.h5`): released features, targets, and the publisher file and position of each trial. |
| `eval_session_file.py` | Greedy CTC evaluation of a checkpoint on trials read from a common session file. `--trainer-input` checks the rebuilt features and labels against the trainer input, `--compare` checks the predictions against an archived per-trial CSV. |
| `build_split_manifest.py` | Writes `data/splits/dryad_x69p8czpq__source_partitions.csv` from the exported session files and checks their validation trials against the archived evaluation list. |
| `train_btot24.py` | Trains the GRUDecoder on the converted data by programmatically adapting the official `rnn_args.yaml` (Windows-safe settings, scaled LR schedule). |
| `eval_btot24.py` | Greedy CTC evaluation of a checkpoint on the test partition (PER = Σeditdist / Σlen, official semantics). |
| `dump_btot24_logits.py` | Same model/iteration path as `eval_btot24.py`, but saves per-trial `log_softmax` outputs for word-level decoding. |
| `decode_words_btot24_v2.py` | **Canonical word decoder** (replaces the retired buggy v1, see `docs/20261108_cleanup_audit/CLEANUP_AUDIT.md`): joint lexicon-constrained CTC beam search over a pronunciation trie with KenLM 3-gram scoring; the 41× RAM-reduction path documented in the README. |
| `export_trials_npy.py` | Exports per-trial log-prob `.npy` + `meta.tsv` into `~/tlv` for the C++ `tlg_viterbi` 5-gram decoder (`shared/tools/tlg_viterbi.cc`). |
| `plot_willett2023_wer.py` | Draws the paper-vs-ours PER/WER figure from the CSVs produced above. |
