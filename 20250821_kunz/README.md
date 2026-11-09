# `20250821_kunz` — Kunz et al., *Cell* 2025 (Inner Speech in Motor Cortex, T12 / T15 / T16)

- **Paper**: Kunz, E. M. et al. Inner speech in motor cortex and implications for speech neuroprostheses. *Cell* **188**, 4658–4673 (2025-08-21). [doi:10.1016/j.cell.2025.06.015](https://doi.org/10.1016/j.cell.2025.06.015)
- **Dataset**: Dryad [`doi:10.5061/dryad.gf1vhhn1j`](https://doi.org/10.5061/dryad.gf1vhhn1j)
- **Upstream Code**: [`nptl-stanford/inner_speech`](https://github.com/nptl-stanford/inner_speech)

## Consumer-PC Reproduction & Gap Decomposition
- **`code/convert_kunz.py` & `code/train_kunz.py`**: Converts and trains attempted vs. inner vs. mixed speech models for participants T12, T15, and T16.
- **`code/20261028_gap_decomposition.py`**: Quantifies why training from scratch on the small released inner-speech evaluation subsets (1–3 sessions per participant) yields high PER (~60%) compared to the authors' longitudinal pre-trained online checkpoints.

## Reproduction Summary (`results/`)
- **Greedy PER (From-Scratch on Released Subsets)**: **61.4%** (T12) / **59.7%** (T15) / **61.5%** (T16) vs. paper online anchors **24.0% / 54.0% / 52.0%**.
- **Gap Attribution (`results/20261028_gap_decomposition.tsv`)**: Demonstrates that acoustic PER $> 25\%$ crosses the LM phase-transition boundary (panel i of the benchmark overview figure), causing n-gram + LLM rescoring to hallucinate insertions rather than reduce WER.
