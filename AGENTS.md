# AGENTS.md — Repository Root (Communication Neuroprostheses Consumer-GPU Reproduction Suite)

This repository reproduces **8 open-data intracortical speech, typing, and handwriting brain–computer interface (BCI) studies** (*Nature*, *NEJM*, *Cell*, *Nature Neuroscience*, *Cell Reports*, *NeurIPS*) end-to-end on a single consumer desktop (**1× NVIDIA GeForce RTX 5060 Ti 16 GB VRAM, 64 GB DDR5 host RAM**). It holds per-study memory-wall mitigations and PyTorch/TF model ports (`studies/yyyymmdd_author/code/`), archived per-trial decode outputs (`studies/yyyymmdd_author/results/`), shared C++ `mmap` OpenFST decoders and LLM rescorers (`shared/`), zero-download interactive Jupyter Notebooks (`tutorials/`), and the benchmark overview figure renderer (`figures/code/render_figure2.py`).

---

## Hard Prohibitions (Do Not Break These)

1. **Never violate the 16 GB GPU VRAM & 64 GB host RAM hardware ceiling**:
   - **Do NOT preload unpruned Kaldi 5-gram `TLG.fst` graphs into heap RAM** (official pipelines consume 64–300 GB RAM and will trigger an immediate OS OOM crash on a consumer PC). Always use the `mmap + .idx` on-demand state-offset decoders in `shared/tools/tlg_viterbi.cc` and `shared/tools/wenet_lattice.cc` (19.6 GB RAM), or the 1.46 GB KenLM 3-gram pronunciation-trie decoder in `studies/20230831_willett/code/decode_words_btot24_v2.py`.
   - **Do NOT eagerly `np.concatenate` all training folds or `.mat` sessions in memory** (official Willett 2021 Step 3 consumes 56 GB RAM). Always stream via `numpy.memmap` or per-session iterators.
   - **Do NOT load the acoustic trainer and a 6.7B–8B causal LLM in the same process**. Acoustic training/logit dumping and second-pass LLM rescoring (`shared/tools/opt_rescore.py`) must run as decoupled sequential processes.
2. **Never hand-edit auto-generated benchmark files**:
   - `figures/results/20261124_repository_migration/reproduction_benchmark.json` is auto-synced from the canonical source of truth `figures/reproduction_benchmark.yaml` whenever `figures/code/render_figure2.py` runs. **Edit `figures/reproduction_benchmark.yaml` only.**
   - `figures/results/20261124_repository_migration/session_per_vs_wer.json` is generated programmatically by `figures/code/build_session_per_vs_wer.py` from the archived CSV/TSV decode outputs. **Do not hand-edit `figures/results/20261124_repository_migration/session_per_vs_wer.json`.**
3. **Never overwrite archived baseline artifacts in `studies/yyyymmdd_author/results/`**:
   - The `.tsv`, `.csv`, and `.json` files under each study's `results/` folder are verified by `.github/workflows/verify_benchmark.yml` and `figures/code/build_session_per_vs_wer.py`. When running new student ablations or model modifications, write outputs to new filenames or subdirectories (e.g., `results/yyyymmdd_ablation_*/`).
4. **Never force-add large binary weights or raw datasets to Git**:
   - `.gitignore` excludes `*.pt`, `*.pth`, `*.ckpt`, `*.h5`, `*.bin`, `*.npy`, `*.npz`, `*.mat`, `*.tfrecord`, `*.fst`, and `*.arpa`.

---

## Top-Level Directory & File Map

| Path | Role & Contents |
|---|---|
| [`studies/20210512_willett/`](studies/20210512_willett) | Willett et al., *Nature* 2021 (handwriting BCI, T5): `mmap` synthetic fold assembly + PyTorch `CharSeqRNN` port + lazy Kaldi/Python bigram FST decode + GPT-2 rescoring. |
| [`studies/20230831_willett/`](studies/20230831_willett) | Willett et al., *Nature* 2023 (speech BCI, T12): FP16 5-layer GRU + 1.46 GB KenLM 3-gram / 19.6 GB `mmap` 5-gram WFST + OPT-6.7B / Qwen3-8B rescoring. |
| [`studies/20231201_fan/`](studies/20231201_fan) | Fan et al., *NeurIPS* 2023 (403-day handwriting CORP, T5): frozen seed RNN drift evaluation (Tier 2) and 403-day online pseudo-label recalibration chain (Tier 3 `v4–v8`). |
| [`studies/20240815_card/`](studies/20240815_card) | Card et al., *NEJM* 2024 (ALS speech BCI & rapid calibration, T15): 45-session GRU/BiGRU training, day-index routing fix (`p4_fixed_*`), and Day-1 sentence-by-sentence calibration trajectories. |
| [`studies/20250612_wairagkar/`](studies/20250612_wairagkar) | Wairagkar et al., *Nature* 2025 (instantaneous voice synthesis, T15): 1.5 GB VRAM causal 40-bin Mel encoder inference (Tier 1) and DTW-aligned from-scratch training (Tier 2). |
| [`studies/20250821_kunz/`](studies/20250821_kunz) | Kunz et al., *Cell* 2025 (inner & attempted speech, T12/T15/T16): attempted/imagined/mixed training arms, T16 3-fold block holdout CV, and >26% PER phase-transition gap attribution. |
| [`studies/20260316_jude/`](studies/20260316_jude) | Jude et al., *Nat. Neurosci.* 2026a (bimanual touch typing, T17/T18): PyTorch `InputNet + TypingGRU` port (single-session & multi-session + warm-start), 5-gram WFST decode, and CPM/WPM audits. |
| [`studies/20260428_jude/`](studies/20260428_jude) | Jude et al., *Cell Rep.* 2026b (intended speech in LIS, T17): 768-dim neural feature extraction, 12 conversational 125k-word sessions GRU training, and 5-gram + OPT/Qwen rescoring. |
| `studies/*/code/export_sessions.py` | Common session export of each study: writes one `data/processed/<participant>/<release>_<session>.h5` per recording session in the layout defined in `shared/session_format.py` (neural features as released, zero-based label ids, the publisher file and position of every trial). `shared/tools/validate_session.py <file> --source trials` re-reads every trial from its publisher file; trial roles, folds, and calibration order stay in `data/splits/` lists. |
| [`figures/`](figures) | Benchmark configuration, figure notebook, renderers, and dated release bundles. |
| [`shared/`](shared) | Cross-study C++ `mmap` OpenFST decoders (`tlg_viterbi.cc`, `wenet_lattice.cc`), LLM rescorers (`opt_rescore.py`), WER evaluator (`wer_from_tsv.py`), `vacation_tlg/` ablations, `finetune_chain/`, and `plots/`. |
| [`tutorials/`](tutorials) | 3 zero-download interactive Jupyter Notebooks in English (`tutorials/*.ipynb`) and Chinese (`tutorials/zh/*_zh.ipynb`) for students. |
| [`docs/`](docs) | Repository governance and cleanup audit logs ([`docs/20261108_cleanup_audit/CLEANUP_AUDIT.md`](docs/20261108_cleanup_audit/CLEANUP_AUDIT.md)), and the fig. 2 / Table 2 cross-repository reconciliation workfiles ([`docs/20261126_fig2_reconciliation/README.md`](docs/20261126_fig2_reconciliation/README.md)): the three reproduced-WER protocol candidates synced against the paper manuscript (adopted: `fig2_matched_dates`), migrated from `tmp/` on 2026-11-28. |
| `figures/reproduction_benchmark.yaml` | Single source of truth for all benchmark panels: Figure 2 panels (a)–(f) and the 4-panel phase/memory companion figure. |
| `figures/results/20261124_repository_migration/reproduction_benchmark.json` | Auto-synced JSON mirror of `figures/reproduction_benchmark.yaml` written by `figures/code/render_figure2.py`. |
| `figures/code/build_session_per_vs_wer.py` | Recomputes session-level `(PER, post-LM WER)` pairs from archived per-trial CSVs and TSVs across the 4 speech studies. |
| `figures/results/20261124_repository_migration/session_per_vs_wer.json` | Output of `figures/code/build_session_per_vs_wer.py`, consumed by panel (a) of `figures/code/render_figure3.py`. |
| `figures/code/build_session_cer_vs_wer.py` | Character-level analog of the PER builder: recomputes session-level `(raw CER, post-LM WER)` pairs for the Willett 2021 / Fan 2023 / Jude 2026a arms (Willett raw-CER counts extracted from archived inference MAT files and stored in the study). |
| `figures/results/20261124_repository_migration/session_cer_vs_wer.json` | Output of `figures/code/build_session_cer_vs_wer.py`, consumed by panel (b) of `figures/code/render_figure3.py`. Do not hand-edit. |
| `figures/code/render_figure2.py` | Standalone renderer and bounding-box collision verifier (`check_figure_overlaps`) for `communication_neuroprostheses_reproduction.{png,pdf,html}` and `index.html` (panels a–f since 2026-11-11; the RAM/VRAM panels and the PER-vs-WER phase diagram moved to the companion figure). The interactive HTML shows the companion's phase-diagram panels (a, b) in its "Insights" section and the memory-wall panels (c, d) in a separate "Memory Savings" section (2026-11-13 split), each page-CSS-capped at the figure's natural 1210 px width (2026-11-12; the `.pdf`/`.png` stay full-size). Figure 6 country hover tooltips list each country's official languages from the curated `_OFFICIAL_LANGUAGES` table (Wikidata P37 / Wikipedia snapshot 2026-11, cross-checked with mledoze/countries; NE-only codes PSX/SOL/CYN/SDS/KOS/SAH handled by hand). On first load/refresh the HTML plays a staggered top-to-bottom entrance cascade (`html.js`-gated `.rv` markup tags + IntersectionObserver; scroll unfolds below-the-fold sections; reduced-motion and no-JS fall back to a static page) (2026-11-16). |
| `figures/code/render_figure3.py` | 4-panel companion figure — WER-vs-PER and WER-vs-CER phase diagrams plus the Host-RAM / GPU-VRAM memory-wall panels moved out of Figure 2 (2026-11-11) — rendering `communication_neuroprostheses_phase_memory.{png,pdf}`; `build_html_svgs()` additionally serialises the panels-(a)+(b) and panels-(c)+(d) halves as the two inline SVGs consumed by `render_figure2.export_interactive_html` (deferred import; builds in-process so CI render order does not matter). The serialised SVGs stamp `pmS--/pmP--/mmB--` gid hooks (see `_ARM_IDS` / `_CATEGORY_IDS`) that the interactive HTML binds to tooltips and `selectStudy()` cross-highlighting, matching the Figure 2 slots (2026-11-16); the `.png`/`.pdf` outputs stay pixel-identical. |
| `environment.yml` / `requirements.txt` | Verified Conda/pip environment specification (`bci-repro`: Python 3.11.18, CUDA 12.8, PyTorch 2.11.0+cu128, TensorFlow 2.14.0). |

---

## End-to-End Pipeline & Quick Verification

### 1. Environment Activation
All Python scripts and notebooks run inside the unified **`bci-repro`** Conda environment:
```bash
conda activate bci-repro
```

### 2. Data Flow Across Stages
```text
[Dryad Raw Recordings (.mat / .hdf5)]
   │  (per-study convert_*.py / window_W4_convert.py / export_tfrecord_to_npz.py)
   ▼
[Standardized Neural Tensors (.hdf5 / .npz / .npy memmap)]
   │  (per-study train_*.py / torch_port/train.py / pytorch_port/train_multisession.py)
   ▼
[Frame-Level Acoustic/Character Logits + Greedy Eval (val_greedy_per_trial.csv / test_logits.npz)]
   │  (shared/tools/tlg_viterbi.cc, wenet_lattice.cc + opt_rescore.py)
   ▼
[Archived Per-Trial Decode Outputs (studies/yyyymmdd_author/results/)]
   │  (figures/code/build_session_per_vs_wer.py + figures/code/render_figure2.py)
   ▼
[figures/results/20261124_repository_migration/session_per_vs_wer.json + communication_neuroprostheses_reproduction.{png,pdf}]
```

### 3. Fast Zero-Download Regression Check (< 10 seconds, matches GitHub Actions CI)
After modifying any aggregation script, evaluation utility, or `figures/reproduction_benchmark.yaml`, always run:
```bash
python figures/code/build_session_per_vs_wer.py
python figures/code/build_session_cer_vs_wer.py
python shared/tools/wer_from_tsv.py --no-sidecar studies/20230831_willett/results/20261124_import_btot24/wl_5g_opt_official.tsv
python figures/code/render_figure2.py
python figures/code/render_figure3.py
```

---

## Conventions for AI Coding Agents

1. **Read the subdirectory `AGENTS.md` and `code/README.md` before editing**: Every top-level subdirectory has its own `AGENTS.md` detailing study-specific tensor dimensions, pitfalls, and read-only files, plus a `code/README.md` listing every script's role.
2. **Windows + WSL2 Hybrid Toolchain**:
   - All GPU training, PyTorch/TF inference, KenLM 3-gram decoding, OPT/Qwen LLM rescoring, and Matplotlib figure generation run **natively on Windows inside `bci-repro`**.
   - Only the C++ Kaldi OpenFST binaries (`shared/tools/tlg_viterbi.cc`, `wenet_lattice.cc`, `studies/20210512_willett/code/latgen_lazy.cc.txt`, and `studies/20260316_jude`'s `py310lm` OpenFST wrapper) run inside **WSL2 (Ubuntu)**. Because all first-pass WFST lattice/1-best outputs are already archived in `results/`, students without WSL2 can still run acoustic model ablations, KenLM 3-gram decoding, or second-pass LLM rescoring natively.
3. **Docstring & Code Style**: Every `.py` module and function follows NumPy-style docstrings (`Parameters`, `Returns`, `Notes`). Preserve this convention when adding or updating code.
4. **Working Language: English**: The working language of this repository is English. Write all code, comments, docstrings, documentation, READMEs, and commit messages in English. **All Git commits must use English commit messages** (e.g., `render: update Figure 2 panels` instead of `更新渲染`). The only exceptions are the Chinese tutorial notebooks under `tutorials/zh/` and Chinese prose inside bilingual educational materials.
5. **Change the common session layout only with a schema version bump**: every `studies/*/code/export_sessions.py` and `validate_session` share `TrialSpec`, `Field`, and the attribute names of `shared/session_format.py`. Extend that module, bump `SCHEMA_VERSION`, and keep `read_source_trial` rebuilding exactly what `iter_trials` wrote; the rules are in [`shared/AGENTS.md`](shared/AGENTS.md).
