# `20260428_jude/code/` — Code Guide

Reproduces the **Jude et al., *Cell Reports* 2026** intended-speech study (participant
T17, long-standing anarthria/LIS) on the `nejm-brain-to-text` GRU stack: 12
conversational sessions with a 125,000-word vocabulary.
Study-level context: [`../README.md`](../README.md).

## Pipeline order

```text
(Dryad RedisMat .mat blocks)
   └─ inspect_jude.py / scan_jude_blocks.py ── survey structure & block inventory
   └─ convert_jude.py ──→ hdf5 in the nejm-brain-to-text layout (768-dim features)
        └─ train_jude.py ──→ GRUDecoder (16 GB VRAM recipe)
             └─ eval_jude.py ──→ greedy PER by session/vocab/loop
             └─ dump_jude_logits.py ──→ per-trial logits
                  └─ shared decode chain (KenLM 3-gram / TLG 5-gram / OPT / Qwen)
```

## File-by-file

| File | Role |
|---|---|
| `inspect_jude.py` | Prints the variables/shapes/cue structure of one RedisMat `.mat` block. |
| `scan_jude_blocks.py` | Inventory scan over all `.mat` blocks: block numbers, trial counts, cue samples, feature shapes. |
| `convert_jude.py` | Per-block extraction following the official `mat_to_tfrecord.py` (trial windowing on 10 ms neural timestamps, 768-dim features with 256→384 channel padding). |
| `train_jude.py` | Trains the 5-layer GRU phoneme decoder on the converted conversational data. |
| `eval_jude.py` | Greedy CTC PER evaluation with breakdowns by session, vocabulary (50w/125k), and loop (open/closed). |
| `dump_jude_logits.py` | Dumps per-trial CTC log-probs for the run's val partition → hdf5 for the decode/rescore chain. |
