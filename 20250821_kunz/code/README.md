# `20250821_kunz/code/` — Code Guide

Reproduces the **Kunz et al., *Cell* 2025** inner-speech study (participants T12, T15,
T16) on the `nejm-brain-to-text` GRU stack, plus a gap decomposition explaining why
training from scratch on the small released inner-speech subsets cannot reach the
paper's closed-loop online numbers.
Study-level context: [`../README.md`](../README.md).

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
