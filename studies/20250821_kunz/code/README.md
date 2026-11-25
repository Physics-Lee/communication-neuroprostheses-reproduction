# `studies/20250821_kunz/code/` — Code Guide

Reproduces the **Kunz et al., *Cell* 2025** inner-speech study (participants T12, T15,
T16) on the `nejm-brain-to-text` GRU stack, plus a gap decomposition explaining why
training from scratch on the small released inner-speech subsets cannot reach the
paper's closed-loop online numbers.
Study-level context: [`../README.md`](../README.md).

## Public commands

Run these commands from the repository root in the environment required by the study. Use `--help` on an entry point for its arguments.

| Public command | Role |
| --- | --- |
| [`prepare_data.py`](prepare_data.py) | Prepare the released sentence data for the Kunz models. |
| [`export_sessions.py`](export_sessions.py) | Export one session, condition, and vocabulary file to the common session format. |
| [`train.py`](train.py) | Train the selected Kunz speech condition. |
| [`infer.py`](infer.py) | Export logits for the selected Kunz model. |
| [`evaluate.py`](evaluate.py) | Evaluate the Kunz model's greedy phoneme errors. |
| [`decode.py`](decode.py) | Run the indexed WFST decoder in WSL. |

Run the indexed WFST command in WSL with the compiled `shared/tools/wenet_lattice.cc` decoder. Supply the graph, words, exported trials, metadata, and the run-specific decoding options.

Use [`record_study_run.py`](../../../shared/tools/record_study_run.py) to name scored outputs from a completed command. See the [eight-study naming map](../../../docs/20261124_repository_migration/naming_map.md) for the result and data paths.

The export command writes `data/processed/<participant>/<release>_<session>.h5` for one session. `python shared/tools/validate_session.py <file> --source trials` re-reads every trial from the publisher's files. The format is defined in [`shared/session_format.py`](../../../shared/session_format.py).

## Pipeline order

```text
(Dryad sentenceData .mat)
   └─ inspect_kunz.py ── survey .mat structure (trial counts, blocks, decoded text)
   └─ convert_kunz.py ──→ hdf5 in the nejm-brain-to-text layout
        └─ train_kunz.py ──→ one model per participant × arm
           (mixed / imagined-only / attempted-only; runners: run_ablation_arms.sh)
             ├─ eval_kunz.py ──→ greedy PER on the closed-loop *_eval blocks
             └─ dump_kunz_logits.py ──→ per-trial logits → shared decode/rescore chain
T16 attempted-speech holdout experiment (review task A.4):
   make_t16_holdout.py ──→ 3-fold block-level CV splits
        └─ train_t16_holdout.py (runner: run_t16_holdout.sh)
             └─ eval_t16_holdout.py
Analysis & figures:
   20261028_gap_decomposition.py ──→ why offline WER ≫ paper online WER
   plot_ablation_holdout.py ──→ arm-comparison + holdout figures
```

## File-by-file

| File | Role |
|---|---|
| `export_sessions.py` | Writes one session, condition, and vocabulary file to the common session format (`data/processed/<participant>/dryad_gf1vhhn1j_<date>_<condition>_<vocabulary>.h5`): the released go-period features at the bin size the file records, the sentences, and the block of each trial. For T15 and T16 these features equal the per-trial `sentenceFeatures` the converter reads. For T12 they do not: the session file keeps the continuous 128-electrode arrays, and the converter reads the publisher's 8-bit per-trial copy with another electrode order (see the exporter docstring). |
| `build_split_manifest.py` | Writes `data/splits/dryad_gf1vhhn1j__conditions.csv` and `data/splits/dryad_gf1vhhn1j__t16_folds.csv` from the exported session files. |
| `inspect_kunz.py` | Prints the structure of every released `sentenceData` `.mat`: variables, trial counts, blocks, channel sets, sentences, and the recorded `decodedSentenceText` (works for both string/char-code storage conventions). |
| `convert_kunz.py` | Converts sentenceData `.mat` files into the nejm-brain-to-text hdf5 layout (first-half binnedTX + second-half spikePow features, per-vocab phoneme targets). |
| `train_kunz.py` | Trains the GRUDecoder per participant and arm (mixed / imagined / attempted). |
| `eval_kunz.py` | Greedy CTC PER evaluation on the closed-loop imagined `*_eval` sessions (+ per-session and per-vocab breakdowns). |
| `dump_kunz_logits.py` | Dumps per-trial CTC log-probs for one ablation arm's eval trials → hdf5 for the decode chain. |
| `make_t16_holdout.py` | Builds the 3-fold block-level holdout splits from the T16 attempted 50-word session. |
| `train_t16_holdout.py` | Trains one holdout fold (train = 2 blocks, val = held-out block). |
| `eval_t16_holdout.py` | Evaluates a holdout fold checkpoint on its held-out block. |
| `20261028_gap_decomposition.py` | Decomposes the paper-vs-ours WER gap along three axes (vocab/condition, LM tier, training data) from archived decode products. |
| `plot_ablation_holdout.py` | Review-response figures: training-arm comparison on identical eval blocks + T16 holdout bars. |
