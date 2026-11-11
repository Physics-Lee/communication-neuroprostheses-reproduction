# AGENTS.md — `20260428_jude/` (Jude et al., *Cell Reports* 2026b Intended Speech in LIS Reproduction)

This directory reproduces **Jude et al., *Cell Rep.* 45, 117162 (2026-04-28)** (decoding intended speech in participant T17 with long-standing anarthria and locked-in syndrome; Dryad DOI: `10.5061/dryad.vq83bk481`).

We extract and train a 5-layer GRU phoneme decoder across T17's 12 conversational 125,000-word vocabulary sessions on a single 16 GB GPU (13.0 GB peak VRAM) + 19.6 GB `mmap` 5-gram WFST:
- **Acoustic Stage (No LM)**: Pooled greedy PER **39.97%** across 12 conversational sessions vs. paper **34.0%** (session mean ± SD: $39.97 \pm 9.15\%$);
- **Full Pipeline (125k 5-gram WFST + OPT-6.7B / Qwen3-8B)**: Pooled WER **46.47%** (with Qwen3-8B rescoring: **45.2%**, Day-68 conversational benchmark **44.37 ± 15.96%**) vs. paper conversational benchmark **~50%**.

---

## What Not to Do Here (Conventions & Guardrails)

1. **Preserve the 256 → 384 channel zero-padding convention (768 total features)**:
   - `convert_jude.py` follows upstream `mat_to_tfrecord.py` by windowing trials on 10 ms neural timestamps and zero-padding the 256 recorded channels to 384 channels (× 2 feature streams = **768 features**) so tensors match the unified `nejm-brain-to-text` HDF5 trainer layout. Do not alter channel ordering or padding width.
2. **Distinguish 50-word vs. 125k-word conversational blocks and open- vs. closed-loop blocks**:
   - The released RedisMat `.mat` files contain mixed block types; `scan_jude_blocks.py` and `eval_jude.py` report breakdowns by `session`, `vocab` (50w vs. 125k), and `loop` (open vs. closed). Always verify evaluation scope when comparing numbers.
3. **Never alter row ordering in `results/decode_outputs/val_greedy_per_trial.csv` or `off_5g_opt_official.tsv`**:
   - Root script `build_session_per_vs_wer.py` joins these two files positionally to compute the 12 session points for `Jude 2026 (LIS, T17)` in `session_per_vs_wer.json`.

---

## Directory Contents & File Roles

```text
20260428_jude/
├── README.md                                       # Study summary (16 GB VRAM GRU + 125k 5-gram WFST + OPT/Qwen results)
├── code/
│   ├── README.md                                   # Pipeline order & script guide
│   ├── inspect_jude.py                             # Inspects variables, shapes, and cue structure of a single RedisMat .mat block
│   ├── scan_jude_blocks.py                         # Scans all .mat blocks to inventory block numbers, trial counts, cues, and shapes
│   ├── convert_jude.py                             # Extracts trials from RedisMat .mat blocks into 768-dim nejm-brain-to-text HDF5 format
│   ├── train_jude.py                               # Trains the 5-layer GRU phoneme decoder in 16 GB VRAM
│   ├── eval_jude.py                                # Evaluates greedy CTC PER with breakdowns by session, vocabulary, and loop type
│   └── dump_jude_logits.py                         # Exports per-trial CTC log-probs to HDF5 for the shared decode/rescore chain
└── results/
    ├── decode_outputs/
    │   ├── val_greedy_per_trial.csv                # Per-trial greedy PER across 12 conversational sessions (pooled PER = 39.97%)
    │   ├── word_decode_3gram.csv                   # KenLM 3-gram beam-search decode output
    │   ├── off_final_full.tsv                      # 125k 5-gram WFST 1-best decode output
    │   ├── off_5g_opt_official.tsv                 # 125k 5-gram + OPT-6.7B rescored output (pooled WER = 46.47%)
    │   ├── off_5g_opt_qwen3base_fx.tsv             # 125k 5-gram + Qwen3-8B Base rescored output
    │   └── off_5g_qwen.tsv                         # Early Qwen rescoring baseline
    └── logs/
        └── jude_train.log                          # Full 5-layer GRU training log
```

---

## Quick Verification Command (run from repository root)

```bash
conda activate bci-repro
# Recompute pooled WER on the archived 125k 5-gram + OPT-6.7B decode output
python shared/tools/wer_from_tsv.py 20260428_jude/results/decode_outputs/off_5g_opt_official.tsv
```
