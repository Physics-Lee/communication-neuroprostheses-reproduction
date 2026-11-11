# AGENTS.md — `20260316_jude/` (Jude et al., *Nature Neuroscience* 2026a Bimanual Touch-Typing BCI Reproduction)

This directory reproduces **Jude et al., *Nat. Neurosci.* 29, 2613–2622 (2026-03-16)** (bimanual touch-typing neuroprosthesis in participants T17 with ALS/dysarthria and T18 with spinal cord injury; Dryad DOI: `10.5061/dryad.cz8w9gjjk`).

Because the upstream trainer relies on TensorFlow 2.14 (which lacks native Windows GPU support and whose bundled CUDA 12.2 does not support the RTX 5060 Ti Blackwell architecture, taking 13+ days on CPU per session), we built a **faithful PyTorch port in `code/pytorch_port/` (`InputNet + TypingGRU`, 11.5 GB peak VRAM)** and bypassed eager preloading of the unpruned 5-gram FST during beam-search initialization (**64+ GB → 22.5 GB host RAM**).
- **T18 (8 sessions)**: Raw CER **9.34 ± 2.84%** (best **5.3%**, paper $<12\%$); post-LM WER **7.74 ± 4.33%** across sessions (best **0.0%**, 4-session copy-typing mean **6.9%**) vs. paper $<7\%$.
- **T17 (8 sessions)**: Multi-session joint training + per-session warm-start fine-tuning (`train_multisession.py`) achieves raw CER **19.42 ± 5.80%** (best **8.2%**) vs. paper Source Data **18.6 ± 5.75%**, and post-LM WER **11.41 ± 6.85%** across the 6 Fig. 3b online days (best **4.6%**, 5/8 sessions $<15\%$) vs. paper **12.5 ± 2.49%**.

---

## What Not to Do Here (Conventions & Guardrails)

1. **Never replace the 5 stacked single-layer `nn.GRU` modules in `TypingGRU` with a monolithic `nn.GRU(num_layers=5, dropout=0.6)`**:
   - On Windows CUDA (`cu128`), `nn.GRU(num_layers=5, dropout=0.6)` falls off the cuDNN fast path (measured forward pass **2.04 s vs. 0.23 s — 9× slower**) and does not implement Keras variational dropout (which shares a single `(B, 1, F)` dropout mask across all time steps and applies dropout before layer 0 as well). `code/pytorch_port/model.py` stacks 5 single-layer `nn.GRU` modules in an `nn.ModuleList` with explicit `(B, 1, F)` variational masks, preserving both cuDNN speed and exact Keras semantics.
2. **Respect participant-specific hyperparameters for T17 vs. T18 (see `code/pytorch_port/train_multisession.py`)**:
   - **T17**: excludes Area 55b channels `range(128, 256)` (`n_features = 512`), `patch_size = 16`, `patch_stride = 9`, `white_noise_sd = 1.4`, `const_offset_sd = 0.4`.
   - **T18**: uses all 384 channels (`n_features = 768`), `patch_size = 15`, `patch_stride = 6`, `white_noise_sd = 1.5`, `const_offset_sd = 0.22`.
   - Keep **`Adam(eps=0.1)`** for full 10,000-batch runs (upstream's large epsilon stabilizes early CTC gradients).
3. **Do NOT overwrite archived run reports in `results/runs/` or `results/runs_multisess/`**:
   - `results/runs_multisess/lm_decode_summary.json` is read directly by [`../tutorials/03_single_session_end_to_end_demo.ipynb`](../tutorials/03_single_session_end_to_end_demo.ipynb).

---

## Directory Contents & File Roles

```text
20260316_jude/
├── README.md                                       # Study summary (T17/T18 single- and multi-session CER/WER)
├── code/
│   ├── README.md                                   # Pipeline order & script inventory
│   ├── window_W4_convert.py                        # Generic per-session converter wrapping official mat_to_tfrecord
│   ├── g1_recompute_cpm.py                         # Recomputes free-typing speed (CPM/WPM) and online WER from trial timestamps
│   ├── g1_scan_blocks.py                           # Scans sentence-copy blocks across T18 sessions (locates the 110 CPM peak block)
│   ├── m1_t18_20241204_args*.yaml                  # Official-derived training hyperparameter YAML configs (full & smoke)
│   ├── nightly_S*.sh / fix_t17_chain.sh / run_t17_remaining.sh # Toolchain build, LM setup, and batch training shell drivers
│   └── pytorch_port/                               # [CANONICAL] PyTorch port of the TF 2.14 trainer & 5-gram decode drivers
│       ├── README.md                               # Component-by-component TF-vs-PyTorch fidelity table
│       ├── model.py                                # InputNet (identity-init softsign) & TypingGRU (unfold + 5x single-layer cuDNN GRU)
│       ├── dataset.py                              # TypingDataset, 13-tap gauss_smooth, greedy_ctc_decode, edit_distance
│       ├── export_tfrecord_to_npz.py               # Lossless one-time converter from official tfrecord to m1_torch_data.npz
│       ├── train.py                                # Single-session PyTorch trainer (supports --smoke 30)
│       ├── train_multisession.py                   # Multi-session joint trainer (shared TypingGRU + per-day InputNet) + warm-start fine-tuning
│       ├── dump_logits.py                          # Exports test-partition logits from single-session checkpoints
│       ├── s9a_lm_decode_multi.py                  # Loads Typing5gramPunc FST once in WSL (py310lm) to decode multiple single-session runs
│       └── decode_multisession_lm.py               # Batch 5-gram WFST decoder across all 16 T17/T18 multi-session runs
└── results/
    ├── runs/m*_t{17,18}*_torch/final_report.json   # Archived single-session training & raw CER reports (16 sessions)
    ├── runs_multisess/
    │   ├── lm_decode_summary.json                  # Complete per-session and per-trial (ref, hyp, ce, we) 5-gram decode summary
    │   ├── t17/multisession_summary.json           # T17 8-session joint + warm-start raw CER summary
    │   └── t18/multisession_summary.json           # T18 8-session joint + warm-start raw CER summary
    └── logs/                                       # Environment build, conversion, training, and LM decode logs
```

---

## Tensor Shapes & Quick Start

* **Input neural tensor**: Variable-length `(T_i, 768)` `float32` at 20 ms steps (`(T_i, 512)` for T17 after Area 55b exclusion). Temporal patch unfolding (`patch_size`, `patch_stride`) produces $T' = \lfloor (T_i - \text{patch\_size}) / \text{patch\_stride} \rfloor + 1$ frames.
* **CTC output classes**: 31 classes (0..29 = characters/space/punctuation, **30 = CTC `<blank>`**).
* **5-Second Zero-Download GPU Demo**:
  Open [`../tutorials/03_single_session_end_to_end_demo.ipynb`](../tutorials/03_single_session_end_to_end_demo.ipynb) to train `InputNet + TypingGRU` from scratch on GPU in ~4.3 seconds and inspect the archived T17/T18 results.
