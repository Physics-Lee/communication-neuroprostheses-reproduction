# `20231201_fan` — Fan et al., *Nature Communications* 2023 (403-Day Plug-and-Play Handwriting CORP, T5)

- **Paper**: Fan, C. et al. Plug-and-play stability for intracortical brain-computer interfaces: a 1-year demonstration of seamless brain-to-text communication. *Nat. Commun.* **14**, 7965 (2023-12-01). [doi:10.1038/s41467-023-43858-3](https://doi.org/10.1038/s41467-023-43858-3)
- **Dataset & Pretrained Seed Weights**: Dryad [`doi:10.5061/dryad.hqbzkh1p6`](https://doi.org/10.5061/dryad.hqbzkh1p6)
- **Upstream Code**: [`cfan89/CORP`](https://github.com/cfan89/CORP)

## Consumer-PC Reproduction Pipeline
- **`code/tier2_seed_model_inference.py`**: Evaluates the released pretrained seed RNN across all 10 follow-up sessions (403 days).
- **`code/tier3_corp_sim_v6.py`**: Reproduces the full 403-day online unsupervised pseudo-label recalibration loop (CORP) on a single 16 GB GPU (peak 4.0 GB VRAM).
- **`code/rescore_official_outputs.py`**: Runs Kaldi bi-gram FST + GPT-2 rescoring over the decoded character lattices.

## Reproduction Summary (`results/`)
- **Frozen Seed Baseline**: Raw CER **2.65%** (Day 7) drifting to **34.71%** (Day 403) without adaptation.
- **403-Day CORP Online Adaptation (`tier3_corp_sim_v6.json`)**: Mean raw CER **6.20%** vs. paper **6.16%**; post-LM WER **6.10%** vs. paper **6.20%**.
