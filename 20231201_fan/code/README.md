# `20231201_fan/code/` — Code Guide

Reproduces the **Fan et al., *NeurIPS* 2023** CORP study (403-day plug-and-play
handwriting BCI, participant T5): the paper's key claim is that *online unsupervised
pseudo-label recalibration* keeps a decoder usable for a year. The reproduction is
organized in tiers.
Study-level context: [`../README.md`](../README.md).

## Pipeline order

```text
Tier 1 (no model):  rescore_official_outputs.py
      └─ rescore the OFFICIAL released online decode outputs (recal vs no-recal arms)
Tier 2 (frozen):    tier2_seed_model_inference.py
      └─ run the released pretrained seed RNN on all 10 follow-up sessions
Tier 3 (the CORP dynamic, an intentional version chain):
      tier3_corp_sim_official_labels.py (v4: official lm_decoded pseudo-labels;
              also the shared library imported by v5–v8)
        └─ tier3_corp_sim_v5.py  (500-step CORP-scale budget, input layer only)
             └─ tier3_corp_sim_v6.py  (full decoder trainable — README headline:
                    mean CER 6.20% vs paper 6.16%)
                  ├─ tier3_corp_sim_v7.py  (leakage-free final-6-trials holdout)
                  └─ tier3_corp_sim_v8.py  (worst-case self-greedy labels test)
      diag_train_step.py / diag_train_budget.py
      └─ one-step / budget diagnostics that justified v5's 500-step choice
```

The v-numbering is an **experiment log, not duplication**: each version states in its
module docstring what the previous one got wrong and what changed. v6 is the
README-documented result; v7/v8 are the methodological follow-ups (their JSONs are in
`../results/`). The retired v1 (greedy self-labels, retracted claim) was removed —
see `docs/CLEANUP_AUDIT.md`.

## File-by-file

| File | Role |
|---|---|
| `rescore_official_outputs.py` | Tier-1: rescores the official online decoded outputs shipped in the Dryad release (independent of running any model). |
| `tier2_seed_model_inference.py` | Tier-2: evaluates the frozen pretrained seed RNN across the 10 follow-up sessions (shows the drift the paper's CORP fixes). |
| `tier3_corp_sim_official_labels.py` | Tier-3 v4 + library: session-by-session online recalibration simulation using the official `lm_decoded` pseudo-labels; provides `make_args`, `gather_trials`, `load_official_labels`, `forward_logits`, `greedy_decode`, `ctc_loss`, … to v5–v8. |
| `tier3_corp_sim_v5.py` | v5: adds the CORP-scale update budget (500 steps/session, 70/30 pseudo/replay); provides `cer` used by v6–v8. |
| `tier3_corp_sim_v6.py` | v6: unfreezes the full decoder (matching the paper's update scope) — the headline reproduction run. |
| `tier3_corp_sim_v7.py` | v7: same protocol with the final 6 trials of every session held out of the training pool (leakage-free post measure). |
| `tier3_corp_sim_v8.py` | v8: re-tests the label axis with self-greedy pseudo-labels (robustness worst case). |
| `diag_train_step.py` | One-step gradient-flow diagnostic on the worst session. |
| `diag_train_budget.py` | Session-3-only update-budget experiment that motivated v5's 500 steps. |
