# AGENTS.md — `20231201_fan/` (Fan et al., *NeurIPS* 2023 403-Day Plug-and-Play CORP Reproduction)

This directory reproduces **Fan et al., *NeurIPS* 36, 42258–42270 (2023)** (403-day unsupervised online recalibration for handwriting BCI in participant T5; Dryad DOI: `10.5061/dryad.hqbzkh1p6`). The study demonstrates that while a frozen seed RNN degrades from **2.65% CER** (Day 7) to **34.71% CER** (Day 403; static multi-session pool **50.85%**), unsupervised online pseudo-label recalibration (**CORP**) maintains a **6.20% mean raw CER** (paper: **6.16%**) and **6.10% post-LM WER** (paper: **6.20%**) across 10 follow-up sessions spanning 403 days on a single consumer GPU (**4.0 GB peak VRAM, 18.4 GB host RAM**).

---

## What Not to Do Here (Conventions & Guardrails)

1. **Do NOT delete `tier3_corp_sim_official_labels.py` (v4) or `tier3_corp_sim_v5.py`**:
   - Although `tier3_corp_sim_v6.py` produces the headline result reported in `README.md`, **both `v4` and `v5` are imported as libraries by `v6`, `v7`, and `v8`** (`v4` supplies `make_args`, `gather_trials`, `load_official_labels`, `forward_logits`, `greedy_decode`, and `ctc_loss`; `v5` supplies `cer` and the 500-step 70/30 pseudo-label/replay loop). Deleting `v4` or `v5` will break imports across `v6–v8`.
2. **Understand the `v4 -> v5 -> v6 -> v7 -> v8` controlled ablation chain (not redundant copies)**:
   - `v4`: initial low-budget input-layer-only simulation + shared library functions;
   - `v5`: adds the official 500-step update budget with 70/30 pseudo-label/replay mixing, still freezing the shared RNN backbone (mean post CER = 40.2%);
   - **`v6` (Headline Reproduction)**: unfreezes the full shared RNN decoder + per-session input layer (matching the paper's adaptation scope), achieving **6.20% mean raw CER**;
   - `v7`: holds out the final 6 trials of every session from the adaptation pool for a strictly leakage-free post-adaptation measurement;
   - `v8`: stress-tests self-greedy pseudo-labels (without LM correction) as a worst-case lower bound.
3. **Preserve TensorFlow session/graph garbage collection**:
   - Iterating adaptation across 10 sessions requires explicit TF graph/session cleanup between sessions to cap host RAM at 18.4 GB (vs. 40 GB upstream).

---

## Directory Contents & File Roles

```text
20231201_fan/
├── README.md                                     # Study summary (Tier 1–3 reproduction metrics)
├── code/
│   ├── README.md                                 # Tier architecture & v4–v8 ablation lineage guide
│   ├── rescore_official_outputs.py               # Tier 1: Kaldi bi-gram FST + GPT-2 rescoring over official shipped decode outputs
│   ├── tier2_seed_model_inference.py             # Tier 2: Evaluates the frozen pretrained seed RNN across all 10 follow-up sessions
│   ├── tier3_corp_sim_official_labels.py         # Tier 3 (v4) & core library imported by v5–v8 (model loading, forward, CTC loss)
│   ├── tier3_corp_sim_v5.py                      # Tier 3 (v5) & library: 500-step budget (input-layer-only update; exports cer())
│   ├── tier3_corp_sim_v6.py                      # Tier 3 (v6) headline script: full decoder + input layer adaptation (6.20% CER)
│   ├── tier3_corp_sim_v7.py                      # Tier 3 (v7): leakage-free evaluation holding out the last 6 trials per session
│   ├── tier3_corp_sim_v8.py                      # Tier 3 (v8): unassisted self-greedy pseudo-label robustness test
│   ├── diag_train_step.py                        # Diagnostic: single-step gradient flow inspection across layers
│   └── diag_train_budget.py                      # Diagnostic: step-budget sweep on Session 3 motivating the 500-step protocol
└── results/
    ├── rescore_official_outputs.json             # Tier 1 rescoring summary
    ├── tier2_seed_model_inference.json           # Tier 2 frozen seed RNN results (2.65% seed self-test vs. 50.85% static follow-up)
    ├── tier3_corp_sim_official_labels.json       # v4 archived metrics
    ├── tier3_corp_sim_v5.json                    # v5 archived metrics (input-layer-only: mean_post = 40.2%)
    ├── tier3_corp_sim_v6.json                    # v6 headline archived metrics (full decoder: mean_pre = 26.9% -> mean_post = 6.20%)
    ├── tier3_corp_sim_v7.json                    # v7 leakage-free holdout metrics
    └── tier3_corp_sim_v8.json                    # v8 self-greedy pseudo-label metrics
```

---

## Quick Inspection & Execution

* **Zero-download visualization**:
  Run Section 4 of [`../tutorials/01_zero_download_results_playground.ipynb`](../tutorials/01_zero_download_results_playground.ipynb) to plot pre- vs. post-CORP CER across all 10 follow-up sessions directly from `results/tier3_corp_sim_v6.json`.
* **Full re-execution (requires Dryad dataset & pretrained seed weights `doi:10.5061/dryad.hqbzkh1p6`)**:
  ```bash
  conda activate bci-repro
  python 20231201_fan/code/tier2_seed_model_inference.py
  python 20231201_fan/code/tier3_corp_sim_v6.py
  ```
