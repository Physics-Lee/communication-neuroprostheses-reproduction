# AGENTS.md — `tutorials/` (Interactive Consumer-GPU Jupyter Notebooks)

This directory contains **3 zero-download interactive Jupyter Notebook tutorials** in both **English (`tutorials/*.ipynb`)** and **Chinese (`tutorials/zh/*_zh.ipynb`)**, allowing students and researchers to explore archived decode outputs, benchmark 16 GB GPU / 64 GB RAM memory-wall mitigations, and train a neural decoder end-to-end in ~5 seconds on a consumer GPU.

---

## Directory Contents

| Path | Role & Contents |
|---|---|
| [`README.md`](README.md) | Curriculum overview, local execution instructions (`bci-repro`), and AI coding agent tips. |
| [`01_zero_download_results_playground.ipynb`](01_zero_download_results_playground.ipynb) | **Tutorial 01 (English, zero-download)**: Loads archived per-trial decode outputs across all 8 studies, traces 3-gram KenLM → 125k 5-gram WFST → OPT-6.7B → Qwen3-8B word corrections, plots the 26% PER language-model phase transition, and visualizes 403-day CORP online recalibration. |
| [`02_memory_wall_autopsy_16gb_gpu.ipynb`](02_memory_wall_autopsy_16gb_gpu.ipynb) | **Tutorial 02 (English, zero-download)**: Benchmarks `np.memmap` on-demand slicing (4% resident RAM), profiles the 25.4M-parameter `InputNet + TypingGRU` in FP32 vs. BF16 on GPU (8.1× faster step time, 36% less VRAM), and demonstrates decoupled LLM rescoring ($\alpha, \beta$ word-length bonus). |
| [`03_single_session_end_to_end_demo.ipynb`](03_single_session_end_to_end_demo.ipynb) | **Tutorial 03 (English, 41-class ARPAbet Speech BCI end-to-end)**: Trains a 41-class ARPAbet phoneme (`39 phonemes + SIL + <blank>`) stacked cuDNN GRU decoder on 15 training sentences, proves compositional generalization on **6 strictly unseen test sentences** (`PER: 95.6% -> 0.0%`), plots frame-level phoneme posteriors, and runs full-sequence Viterbi pronunciation-lexicon decoding (`PER -> 0.0% WER`). |
| [`zh/`](zh/) | **Chinese Companion Notebooks (`*_zh.ipynb`)**: Full Chinese translations of Tutorials 01–03, also pre-executed with embedded tables and PNG plots. |

---

## Conventions for AI Coding Agents

1. **Do NOT manually edit multi-hundred-kilobyte base64 PNG strings inside `.ipynb` files**:
   - Each `.ipynb` file embeds pre-rendered `stdout` streams and `image/png` outputs so GitHub readers see all tables and figures immediately. If you update code cells in a notebook, re-execute the notebook using the `bci-repro` Python kernel so all cell outputs and plots remain synchronized.
2. **Preserve the zero-download guarantee**:
   - Every default code cell in `tutorials/*.ipynb` and `tutorials/zh/*_zh.ipynb` must run cleanly on a fresh `git clone` without requiring external 100+ GB Dryad `.mat`/`.hdf5` downloads (relying only on repo-tracked files in `results/`, `reproduction_benchmark.yaml`, `session_per_vs_wer.json`, or in-memory synthetic tensors).
3. **Keep English (`tutorials/*.ipynb`) and Chinese (`tutorials/zh/*_zh.ipynb`) notebooks in sync**:
   - Both sets use `REPO_ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "reproduction_benchmark.yaml").exists())` so relative path resolution works identically regardless of working directory depth.
