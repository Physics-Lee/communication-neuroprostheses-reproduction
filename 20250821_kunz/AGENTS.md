# AGENTS.md — `20250821_kunz/` (Kunz et al., *Cell* 2025 Inner Speech in Motor Cortex Reproduction)

This directory reproduces **Kunz et al., *Cell* 188, 4658–4673 (2025)** (inner speech vs. attempted speech in motor cortex across participants T12, T15, and T16; Dryad DOI: `10.5061/dryad.gf1vhhn1j`).

In addition to training and decoding `mixed`, `imagined`, and `attempted` speech models on a single 16 GB GPU (13.0 GB VRAM + 19.6 GB `mmap` 5-gram WFST), this directory includes a quantitative **gap decomposition (`code/20261028_gap_decomposition.py`)**. It demonstrates why training from scratch on the small publicly released inner-speech evaluation subsets (only 1–3 sessions / 450–720 sentences per participant) yields ~60% greedy PER compared to the authors' unreleased longitudinal pre-trained online checkpoints, and why feeding >26% PER phoneme posteriors into a 125k 5-gram WFST + LLM rescorer crosses the language-model phase boundary into hallucinated word insertions (Panel i of the benchmark overview figure).

---

## What Not to Do Here (Conventions & Guardrails)

1. **Do NOT misdiagnose the high offline WER on Kunz 2025 as a decoder bug**:
   - `results/20261028_gap_decomposition.tsv` proves that this folder runs the **exact same GRU trainer and C++ WFST decoder** that achieve 23.22% WER on Willett 2023 and 7.02% WER on Card 2024. The gap is entirely due to **training from scratch on 1–3 sparse released sessions** vs. the authors' multi-week pre-trained online checkpoints. If a student wants to lower PER/WER on Kunz 2025, the proper research direction is **cross-study transfer learning** (initializing from `20230831_willett` T12 or `20240815_card` T15 checkpoints before fine-tuning on `20250821_kunz`).
2. **Use `inspect_kunz.py` for `.mat` inspection**:
   - Released `sentenceData` `.mat` files store `decodedSentenceText` as either character arrays or string objects depending on the session; `inspect_kunz.py` handles both formats cleanly (see `../docs/CLEANUP_AUDIT.md` §6).
3. **Never reorder or modify `val_greedy_per_trial.csv` or `off_final_full.tsv` under `results/decode_outputs/run_t{12,15,16}_mixed/`**:
   - Root script `build_session_per_vs_wer.py` reads these three pairs to populate `Kunz 2025 (inner, T12/T15/T16)` in `session_per_vs_wer.json`.

---

## Directory Contents & File Roles

```text
20250821_kunz/
├── README.md                                          # Study summary & phase-transition gap attribution
├── code/
│   ├── README.md                                      # Pipeline order & script guide
│   ├── inspect_kunz.py                                # Surveys Dryad sentenceData .mat variables, trial counts, blocks, and online text
│   ├── convert_kunz.py                                # Converts .mat files into nejm-brain-to-text HDF5 layout (binnedTX + spikePow)
│   ├── train_kunz.py                                  # Trains GRUDecoder per participant (t12/t15/t16) x arm (mixed/imagined/attempted)
│   ├── eval_kunz.py                                   # Evaluates greedy CTC PER on closed-loop *_eval blocks (by session & vocab)
│   ├── dump_kunz_logits.py                            # Exports per-trial CTC log-probs to HDF5 for the shared decode/rescore chain
│   ├── make_t16_holdout.py                            # Builds 3-fold block-level cross-validation splits for T16 attempted 50-word session
│   ├── train_t16_holdout.py / eval_t16_holdout.py     # Trains and evaluates the 3-fold T16 holdout models
│   ├── 20261028_gap_decomposition.py                  # Decomposes paper-vs-ours WER gap across vocab size, LM tier, and training volume
│   ├── plot_ablation_holdout.py                       # Plots training-arm ablations and T16 3-fold holdout figures
│   └── run_ablation_arms.sh / run_t16_holdout.sh      # Shell drivers for multi-arm training and holdout CV
└── results/
    ├── 20261028_gap_decomposition.tsv                 # Complete decomposition table across participants, vocabs, and LM stages
    ├── decode_outputs/
    │   ├── run_t12_mixed/                             # Archived T12 decode outputs (val_greedy_per_trial.csv, off_final_full.tsv, etc.)
    │   ├── run_t15_mixed/                             # Archived T15 decode outputs
    │   └── run_t16_mixed/                             # Archived T16 decode outputs
    └── logs/                                          # Training, fine-tune chain, and T16 holdout logs
```

---

## Quick Verification Commands (run from repository root)

```bash
conda activate bci-repro
# Recompute the 3-axis gap decomposition table and generate ablation figures (zero external data needed)
python 20250821_kunz/code/20261028_gap_decomposition.py
python 20250821_kunz/code/plot_ablation_holdout.py
```
