# `tutorials/` — Interactive Consumer-GPU (16 GB VRAM / 64 GB RAM) Notebooks

This directory provides **3 progressive, zero-download Jupyter Notebooks** designed for undergraduate/graduate students and neuroengineering researchers. Every notebook runs out-of-the-box in the unified `bci-repro` Conda environment (**Python 3.11.18 + PyTorch 2.11.0+cu128, verified on 1× NVIDIA GeForce RTX 5060 Ti 16 GB**) without waiting for 100+ GB of external Dryad downloads. All code cells are pre-executed with embedded tables and PNG figures so they can be read directly on GitHub or in VS Code.

* **Primary English Notebooks**: `tutorials/01_*.ipynb` – `tutorials/03_*.ipynb`
* **Chinese Companion Notebooks (中文版)**: [`tutorials/zh/`](zh/)

---

## Tutorial Curriculum & Recommended Order

| # | English Notebook | Chinese Version (中文版) | Runtime | External Data Needed? | What You Will Learn & Run |
|---|---|---|---|---|---|
| **01** | [`01_zero_download_results_playground.ipynb`](01_zero_download_results_playground.ipynb) | [`zh/01_..._zh.ipynb`](zh/01_zero_download_results_playground_zh.ipynb) | ~3 min | **No (Zero-Download)** | Load archived per-trial decode outputs across all 8 studies; trace identical neural trials across **KenLM 3-gram (31.76% WER) → 125k 5-gram WFST (24.85%) → OPT-6.7B (23.22%) → Qwen3-8B (24.11%)**; re-plot the **26% PER language-model phase transition**; visualize **403-day neural drift vs. CORP online pseudo-label recalibration** (Fan et al., 2023). |
| **02** | [`02_memory_wall_autopsy_16gb_gpu.ipynb`](02_memory_wall_autopsy_16gb_gpu.ipynb) | [`zh/02_..._zh.ipynb`](zh/02_memory_wall_autopsy_16gb_gpu_zh.ipynb) | ~5 min | **No (Zero-Download)** | Dissect how **300 GB host RAM & 80 GB A100 VRAM** requirements are compressed into **≤26.9 GB RAM & ≤13.0 GB VRAM**: benchmark `np.memmap` on-demand paging (4% resident RAM), profile the 25.4M-parameter `InputNet + TypingGRU` in **FP32 vs. BF16 (8.1× faster step time, 36% less VRAM)** on your own GPU, and test decoupled LLM rescoring ($\alpha, \beta$ word-length compensation). |
| **03** | [`03_single_session_end_to_end_demo.ipynb`](03_single_session_end_to_end_demo.ipynb) | [`zh/03_..._zh.ipynb`](zh/03_single_session_end_to_end_demo_zh.ipynb) | **~18 sec (GPU)** | **No (41-class ARPAbet phoneme synthesis + strictly unseen test sentences)** | Train a **41-class ARPAbet phoneme** (`39 phonemes + SIL + CTC <blank>`) GRU decoder from scratch on 15 training sentences, evaluate **compositional generalization on 6 strictly unseen test sentences (0% sentence overlap, `PER: 95.6% → 0.0%`)**, plot frame-level ARPAbet posterior heatmaps, and decode noisy phoneme streams (`S P IY R` → `SPEECH`) into exact English words via Viterbi pronunciation-lexicon dynamic programming (`WER = 0.0%`). |

---

## How to Run Locally

1. **Activate the unified `bci-repro` environment**:
   ```bash
   conda activate bci-repro
   ```
2. **Open in VS Code, Cursor, or JupyterLab**:
   - In **VS Code / Cursor**: open any `tutorials/*.ipynb` file, select the **`bci-repro` (Python 3.11.18)** kernel in the top-right picker (`ipykernel==7.4.0` is pre-installed in `bci-repro`), and click **Run All**.
   - In **JupyterLab**: run `jupyter lab tutorials/` inside `bci-repro` (install `pip install jupyterlab` if launching the browser UI).

---

## Working with an AI Coding Agent (ZCode / Claude Code / Cursor)

If you are using an AI coding agent to run course projects or architecture ablations (e.g., *"Swap the 5-layer GRU for a causal Conformer"*, *"Evaluate channel-drop robustness with 64 electrodes"*, or *"Rescore N-best lattices with Llama-3"*):
- **Point your agent to the root [`AGENTS.md`](../AGENTS.md) and the target study's `AGENTS.md` first** rather than asking it to edit multi-hundred-kilobyte `.ipynb` JSON blobs directly.
- Every top-level folder in this repository contains a dedicated English `AGENTS.md` specifying tensor shapes, consumer-PC memory ceilings, read-only benchmark artifacts, and canonical CLI entrypoints.
