# AGENTS.md — `shared/` (Cross-Study C++ `mmap` WFST Decoders, LLM Rescorers & Utilities)

This directory houses shared decoding, rescoring, calibration, and plotting infrastructure used across the speech and handwriting reproduction pipelines (especially `20230831_willett`, `20240815_card`, `20250821_kunz`, and `20260428_jude`):
1. **300 GB → 19.6 GB RAM C++ OpenFST Decoders (`tools/tlg_viterbi.cc`, `tools/wenet_lattice.cc`)**: Memory-mapped (`mmap` + `.idx` state-offset index) C++ Kaldi OpenFST 5-gram WFST beam-search decoders;
2. **Single-GPU Second-Pass LLM Rescorers (`tools/opt_rescore.py`, `tools/qwen_rank_probe.py`)**: Score N-best lattices with Facebook OPT-6.7B (FP16, 13.0 GB VRAM) or 4-bit Qwen2.5-7B / Qwen3-8B on a single 16 GB GPU;
3. **Standardized Evaluation, Label-Free Day Adaptation & Supplementary Figures (`tools/wer_from_tsv.py`, `vacation_tlg/`, `finetune_chain/`, `plots/`)**.

---

## What Not to Do Here (Conventions & Guardrails)

1. **Do NOT copy scripts from `shared/tools/` back into individual study folders**:
   - Duplicate copies of `opt_rescore.py`, `wer_from_tsv.py`, `tlg_viterbi.cc`, `wenet_lattice.cc`, and `glog_shim.h` were removed during the 2026-11-08 cleanup audit (`../docs/CLEANUP_AUDIT.md` §4, §9). Always invoke the canonical versions in `shared/tools/`.
2. **Respect the tokenization convention in `tools/wer_from_tsv.py`**:
   - `norm_ref(sent)` applies `re.sub(r"[^A-Za-z'< ]", " ", sent).upper().split()`, preserving internal apostrophes (e.g., `LORI'S`, `WON'T`) and `<`/`>` tags in uppercase. Note that `wer_from_tsv.py` splits `parts[2]` (`hyp`) on whitespace without uppercasing it; if a rescorer outputs lowercase hypotheses (such as raw Qwen TSVs), uppercase the hypothesis column first (see `20240815_card/code/tools/prep_rescore_for_wereval.py`).
3. **Distinguish `shared/plots/` from root `render_figure2.py`**:
   - Root `render_figure2.py` renders the main 9-panel benchmark overview figure (`communication_neuroprostheses_reproduction.{png,pdf}`).
   - Scripts in `shared/plots/` generate **supplementary multi-study figures** (training trajectories, 5-participant PER difficulty spectrum, recording-day calendars, and `vacation_tlg` ablations).

---

## Directory Contents & File Roles

```text
shared/
├── README.md                            # Shared infrastructure overview & file tables
├── tools/
│   ├── tlg_viterbi.cc                   # C++ frame-synchronous CTC Viterbi decoder over mmap TLG.fst + .idx (19.6 GB RAM 1-best path)
│   ├── wenet_lattice.cc                 # C++ N-best lattice decoder over mmap TLG.fst + .idx (feeds second-pass LLM rescoring)
│   ├── glog_shim.h                      # Minimal header shim allowing .cc decoders to compile without full Google glog
│   ├── opt_rescore.py                   # Second-pass N-best rescorer for OPT-6.7B (FP16) & Qwen (4-bit) with alpha/beta interpolation
│   ├── qwen_rank_probe.py               # Diagnostic: evaluates how well raw LLM log-probabilities rank near-oracle candidates
│   ├── wer_from_tsv.py                  # Official sidecar WER calculator for decode TSVs (writes companion <path>.wer CSV)
│   ├── compare_rescorers.py             # Cross-study pooled WER comparison across 5-gram, OPT-6.7B, and Qwen rescorers
│   ├── parse_train_log.py               # Parses nejm-brain-to-text trainer logs into curves.csv and training_curves.png
│   ├── pkl_online_metrics.py            # Extracts online greedy PER and post-LM WER from official t15_copyTask.pkl records
│   ├── extract_four_papers.py           # Fault-tolerant PDF-to-text extractor for verifying reported paper anchor numbers
│   └── *.sh                             # WSL compilation and sweep driver scripts
├── finetune_chain/
│   └── train_finetune.py                # Rapid-calibration Step 3: fine-tunes a multi-day checkpoint on the first N trials of a target day
├── vacation_tlg/                        # 5-gram TLG decoder diagnostic & ablation suite
│   ├── vac_day_adapt.py                 # T6: label-free per-session logit mean-shift adaptation (lambda sweep)
│   ├── vac_error_ops.py                 # T7: word-level insertion (INS) / deletion (DEL) / substitution (SUB) breakdown
│   ├── vac_oracle_depth.py              # Computes oracle WER as a function of N-best depth k
│   ├── vac_vocab_audit.py               # T7: audits out-of-vocabulary (OOV) reference words against the 125k TLG lexicon
│   └── vac_patch_wip.py                 # Provenance script that patched word-insertion-penalty (WIP) support into wenet_lattice.cc
└── plots/
    ├── plot_overview.py / plot_figs_separate.py # 6-panel BtoT24/BtoT25 training & per-day PER dashboard
    ├── plot_wer_summary.py              # 4-paper PER/WER reproduction vs. paper anchor comparison
    ├── plot_wer_spectrum.py             # Unified-pipeline greedy PER difficulty spectrum across 5 participants
    ├── plot_kunz_jude.py                # Kunz 2025 + Jude 2026b extended PER/WER comparison
    ├── plot_recording_days.py           # Recording-day calendar timelines across the 4 BrainGate speech datasets
    ├── plot_two_experiments.py          # Official-decoder port WERs and LLM rescorer swap figures
    └── plot_vacation_tlg.py             # Summary figures for the vacation_tlg experiment series
```

---

## Common Commands (run from repository root)

```bash
conda activate bci-repro

# 1) Compute official-convention WER on any decode TSV
python shared/tools/wer_from_tsv.py 20230831_willett/results/decode_outputs/wl_5g_opt_official.tsv

# 2) Compare 5-gram vs. OPT-6.7B vs. Qwen rescoring across the 4 speech studies
python shared/tools/compare_rescorers.py

# 3) Inspect CLI flags for second-pass LLM rescoring
python shared/tools/opt_rescore.py --help
```
