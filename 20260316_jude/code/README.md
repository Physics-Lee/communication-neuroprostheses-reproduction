# `20260316_jude/code/` — Code Guide

Reproduces the **Jude et al., *Nat. Neurosci.* 2026** touch-typing neuroprosthesis
(participants T17 intact speech / T18 SCI). The official trainer is TF 2.14 (no
Windows GPU support for this card), so the canonical training line is the
**PyTorch port** in `pytorch_port/` (11.5 GB peak VRAM).
Study-level context: [`../README.md`](../README.md).

## Pipeline order

```text
Per session (generic chain — see fix_t17_chain.sh / run_t17_remaining.sh):
   window_W4_convert.py <session> <blocks> <num_test> <rm_spec>
      └─ official mat_to_tfrecord → tfdata_20ms (TF env)
           └─ pytorch_port/export_tfrecord_to_npz.py <session> (TF env)
                └─ pytorch_port/train.py --npz ... --out <run_dir> (torch env)
                     └─ pytorch_port/dump_logits.py <run_dir> <npz>
                          └─ pytorch_port/s9a_lm_decode_multi.py (+ s9_runs.json,
                             WSL py310lm env) ── 5-gram TLG decode → CER/WER
Analysis:
   g1_recompute_cpm.py ── free-typing blocks: CPM/WPM/WER vs paper (110 cpm anchor)
   g1_scan_blocks.py ── sentence-copy blocks: per-block cpm table
Build/infra: nightly_S1..S6*.sh (env, unzip, LM decoder build), window_W2*.sh
```

Archived per-session runs live in `../results/runs/*_torch`; night logs in
`../results/logs/`. The comparison table `../results/复现对照表.md` was assembled
incrementally by one-off `patch_table_*.py` scripts (removed after being applied —
see `docs/CLEANUP_AUDIT.md`).

## File-by-file

| File | Role |
|---|---|
| `window_W4_convert.py` | Generic per-session converter: official `mat_to_tfrecord.main` with CLI-selected blocks / test-trial count / trial removals (both T17 & T18). |
| `g1_recompute_cpm.py` | G1 analysis: recomputes online typing speed (CPM, multiple conventions incl. peak) and WER of the LM output for free-typing blocks, from trial timestamps in the released `.mat` files. |
| `g1_scan_blocks.py` | Complementary G1 scan over sentence-copy blocks (typing content = cue sentence): per-block mean/median/peak CPM across all T18 sessions — locates the paper's 110-cpm figure. |
| `pytorch_port/` | PyTorch port of the official trainer: `dataset.py` (npz + official augmentation/smoothing), `model.py` (GRU + session input net), `train.py`, `dump_logits.py`, `export_tfrecord_to_npz.py`, `s9a_lm_decode_multi.py` (batched official 5-gram decode). See [`pytorch_port/README.md`](pytorch_port/README.md) for the full fidelity table. |
| `m1_t18_20241204_args*.yaml` | Official-derived training args (full / smoke variants) used by the drivers. |
| `nightly_S*.sh` | Environment setup, data/LM unzip, and LM-decoder build scripts (record of the toolchain). |
| `fix_t17_chain.sh`, `run_t17_remaining.sh` | End-to-end chain drivers for the T17 sessions (convert → npz → train → dump → decode). |
