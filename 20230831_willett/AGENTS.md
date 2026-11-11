# AGENTS.md — `20230831_willett/` (Willett et al., *Nature* 2023 Speech Neuroprosthesis Reproduction)

This directory reproduces **Willett et al., *Nature* 620, 1031–1036 (2023)** (high-performance speech neuroprosthesis in participant T12 across 24 sessions; Dryad DOI: `10.5061/dryad.x69p8czpq`). To overcome the official **300 GB 5-gram WFST host RAM wall** and **80 GB A100 VRAM wall**, this directory provides two consumer-PC decoding paths:
1. **1.46 GB RAM Lightweight Path**: Pure-Python pronunciation-trie + KenLM 3-gram joint CTC beam search (`code/decode_words_btot24_v2.py`, 41× RAM reduction);
2. **19.6 GB RAM Full-Vocabulary Path**: Memory-mapped (`mmap + .idx`) 125,000-word 5-gram Kaldi WFST decoding (`../shared/tools/tlg_viterbi.cc`) + single-GPU OPT-6.7B (FP16, 13.0 GB VRAM) / Qwen3-8B (4-bit) second-pass rescoring (**23.22% WER** vs. paper **23.8%**).

---

## What Not to Do Here (Conventions & Guardrails)

1. **Always use `decode_words_btot24_v2.py`; never resurrect the retired v1 two-stage lexicon lookup**:
   - The deleted v1 script indexed `lexicon_numbers.txt` using `tokens.txt` IDs, corrupting phoneme-to-word pronunciations (see `../docs/CLEANUP_AUDIT.md` §2). `decode_words_btot24_v2.py` is the canonical pronunciation-trie decoder.
2. **Never reorder or filter rows in `results/decode_outputs/` CSV/TSV files**:
   - Root script `build_session_per_vs_wer.py` performs a **strict positional join** between `test_partition_greedy_per_trial.csv` (880 test trials) and `wl_5g_opt_official.tsv` (880 rows), asserting that `sentence_label` matches row-by-row.
3. **Keep Windows-safe DataLoader settings in `train_btot24.py`**:
   - Avoid spawning high-concurrency multiprocessing HDF5 readers on Windows, which duplicate file handles and inflate host RAM.

---

## Directory Contents & File Roles

```text
20230831_willett/
├── README.md                                          # Study summary (RAM/VRAM mitigations & PER/WER results)
├── code/
│   ├── README.md                                      # Pipeline order & file-by-file guide
│   ├── convert_btot24.py                              # Converts Dryad competitionData .mat files to nejm-brain-to-text HDF5 layout
│   ├── train_btot24.py                                # Trains the 5-layer GRUDecoder in FP16 (~3.5 GB peak VRAM)
│   ├── eval_btot24.py                                 # Greedy CTC phoneme error rate (PER) evaluation on the test partition
│   ├── dump_btot24_logits.py                          # Exports per-trial frame-level log_softmax phoneme posteriors to HDF5
│   ├── decode_words_btot24_v2.py                      # 1.46 GB RAM path: joint lexicon-trie CTC beam search + KenLM 3-gram scoring
│   ├── export_trials_npy.py                           # Exports per-trial log-prob .npy + meta.tsv for shared/tools/tlg_viterbi.cc
│   └── plot_willett2023_wer.py                        # Plots paper-vs-ours PER and WER comparison figures
└── results/
    ├── decode_outputs/
    │   ├── test_partition_greedy_per_trial.csv        # Per-trial greedy PER across all 880 test sentences (pooled PER = 23.42%)
    │   ├── word_decode_joint3gram.csv                 # KenLM 3-gram beam-search hypotheses (WER = 31.76%)
    │   ├── wl_final_full.tsv                          # 125k 5-gram WFST 1-best hypotheses (WER = 24.85%)
    │   ├── wl_5g_opt_official.tsv                     # 5-gram + OPT-6.7B rescored hypotheses (canonical WER = 23.22%)
    │   ├── wl_5g_opt_qwen3base_fx.tsv                 # 5-gram + Qwen3-8B Base rescored hypotheses
    │   ├── wl_5g_qwen.tsv                             # Early uncompensated Qwen rescoring baseline
    │   └── curves.csv                                 # Validation PER learning curve trajectory
    └── logs/
        ├── train_60k.log                              # First 60k training iterations log
        └── train_btot24_resume120k.log                # Resumed training log to 120k iterations
```

---

## Tensor Shapes & Quick Verification

* **Input neural features**: Variable-length `(T_i, 256)` `float32` per trial (128 `spikePow` + 128 `tx1` channels, block-wise z-scored).
* **CTC output classes**: 41 classes (39 ARPAbet phonemes + word boundary `SIL` + CTC `<blank>`).

### Zero-Download Verification Commands (run from repository root)

```bash
# 1) Verify canonical 5-gram + OPT-6.7B WER (outputs 1277/5500 = 23.22% over 880 sentences)
python shared/tools/wer_from_tsv.py 20230831_willett/results/decode_outputs/wl_5g_opt_official.tsv

# 2) Regenerate the Willett 2023 PER/WER comparison figure
python 20230831_willett/code/plot_willett2023_wer.py
```
