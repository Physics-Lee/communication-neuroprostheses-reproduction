# AGENTS.md — `20210512_willett/` (Willett et al., *Nature* 2021 Handwriting BCI Reproduction)

This directory reproduces **Willett et al., *Nature* 593, 249–254 (2021)** (intracortical handwriting BCI in participant T5; Dryad DOI: `10.5061/dryad.wh70rxwmv`). Our pipeline compresses the official **56 GB host RAM** and legacy **TensorFlow 1.15 `CudnnGRU`** requirements into **26.9 GB host RAM** (`numpy.memmap` synthetic fold streaming + lazy Kaldi/Python bigram FST decoding) and **4.0 GB GPU VRAM** (native PyTorch `CharSeqRNN` port + HuggingFace GPT-2 rescoring).

---

## What Not to Do Here (Conventions & Guardrails)

1. **Do NOT delete or rename `code/forcedAlignmentHMM.py`**:
   - This file is a NumPy 1.26+/2.x compatibility shim. When `step2_relabel.py` spawns multiprocessing workers, `code/` precedes the upstream repository on `sys.path` so workers import this patched `forcedAlignmentHMM.py` instead of the unpatched upstream module. Removing it breaks Step 2 HMM forced alignment.
2. **Do NOT revert `step3_make_synthetic.py` to in-memory array concatenation**:
   - The upstream implementation concatenates synthetic handwriting snippets in heap RAM before writing `.tfrecord` files (peaking at 56 GB RAM). Our `step3_make_synthetic.py` streams folds directly to `.npy` via `numpy.memmap` (26.9 GB RAM peak).
3. **Respect two upstream `CharSeqRNN` quirks documented in `code/torch_port/`**:
   - **Random-walk noise axis**: Upstream `charSeqRNN.py` applies `cumsum(axis=1)` along the **feature dimension (192 channels)** rather than the time dimension. `code/torch_port/augmentation.py` defaults to `axis_mode='official'` for exact metric parity and exposes `axis_mode='time'` for student ablations.
   - **`.contiguous()` before GRU layer 2**: Layer 2 subsamples every `skipLen=5` frames; calling `.contiguous()` on the strided slice is mandatory on Windows CUDA to stay on the cuDNN fast path (omitting it slows training by ~50%).
4. **Do NOT overwrite archived milestone outputs in `results/`** (`m1a_*`, `m1b_*`, `m1c_*`, `m2_*`, `m3_*`, `eval_from_errcounts*.csv`).

---

## Directory Contents & File Roles

```text
20210512_willett/
├── README.md                              # Study summary (memory-wall adaptations & CER/WER table)
├── code/
│   ├── README.md                          # Pipeline order diagram & script inventory
│   ├── step2_relabel.py                   # Re-runs Step 2 HMM forced alignment and audits agreement with official labels
│   ├── forcedAlignmentHMM.py              # NumPy compatibility shim imported by spawned workers (do not delete)
│   ├── step3_make_synthetic.py            # Synthetic data augmentation writing mmap-friendly .npy folds (56 GB -> 26.9 GB RAM)
│   ├── step5_eval_from_errcounts.py       # Recomputes raw CER/WER + 10,000-sample bootstrap 95% CI from errCounts.mat
│   ├── step6_make_kaldimat.py             # Converts self-trained RNN outputs into Kaldi probability matrices
│   ├── step6_decode_bigram.py             # Pure-Python lazy HCLxG bigram FST decoder (no WSL/Kaldi required)
│   ├── step6_decode_batch.py              # Batch bigram decode of official RNN matrices via WSL latgen_lazy2 (also imported library)
│   ├── step6_decode_ours.py               # Runs the same bigram decoder on self-trained PyTorch RNN matrices
│   ├── step6_select_from_kaldioutput.py   # Reproduces official n-best selection: argmin(ac + 2.0 * lm)
│   ├── step7_rescore_gpt2.py              # HuggingFace GPT-2 second-pass rescorer over candidate lists
│   ├── fig_perday_cer_ours.py             # Plots Fig. 2c-style per-session raw CER bars (official vs. self-trained, both partitions)
│   ├── m4_report.py                       # Idempotent aggregator writing the milestone reproduction report
│   ├── pipeline_watchdog_v4.py            # End-to-end orchestrator (executable documentation of pipeline stage order)
│   ├── latgen_lazy.cc.txt / latgen_nbest.cc.txt # Reference C++ sources for WSL lazy Kaldi lattice decoders
│   └── torch_port/                        # PyTorch port of legacy TF 1.15 charSeqRNN
│       ├── README.md                      # Component-by-component fidelity table
│       ├── model.py                       # CharSeqRNN (Gaussian smoothing + 10 per-day input layers + 2-layer 512-unit GRU + 32-dim head)
│       ├── augmentation.py                # Snippet extraction, mean-drift noise, and white-noise augmentation
│       ├── train.py                       # PyTorch training loop (Adam eps=0.1, grad clip 10, single-day batches)
│       ├── infer.py                       # Step 5 inference exporting logits and errCounts.mat
│       └── sanity_check.py                # 9-point standalone operator & loss equivalence test (zero external data needed)
└── results/                               # Archived milestone CSVs, decoded transcripts, and training logs
```

---

## Tensor Shapes & Common Commands

* **Input neural tensor**: `(B, T=1200, F=192)` `float32` (192 intracortical channels from participant T5).
* **Output tensor**: `(B, T=1200, C=32)` `float32` (31 character probability logits + 1 character-start transition signal `charStartTarget`).

### Quick Commands (run from repository root in `bci-repro`)

1. **Zero-data PyTorch model & loss sanity check (~2 seconds)**:
   ```bash
   python 20210512_willett/code/torch_port/sanity_check.py
   ```
2. **Recompute CER/WER and bootstrap confidence intervals from archived `errCounts`**:
   ```bash
   python 20210512_willett/code/step5_eval_from_errcounts.py
   ```
3. **Smoke-test PyTorch RNN training on real data (requires extracted Dryad dataset)**:
   ```bash
   python 20210512_willett/code/torch_port/train.py --partition HeldOutBlocks --nBatches 500
   ```
