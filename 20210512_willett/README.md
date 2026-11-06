# `20210512_willett` — Willett et al., *Nature* 2021 (Handwriting BCI, T5)

- **Paper**: Willett, F. R., Avansino, D. T., Hochberg, L. R., Henderson, J. M. & Shenoy, K. V. High-performance brain-to-text communication via handwriting. *Nature* **593**, 249–254 (2021-05-12). [doi:10.1038/s41586-021-03506-2](https://doi.org/10.1038/s41586-021-03506-2)
- **Dataset**: Dryad [`doi:10.5061/dryad.wh70rxwmv`](https://doi.org/10.5061/dryad.wh70rxwmv)
- **Upstream Code**: [`fwillett/handwritingBCI`](https://github.com/fwillett/handwritingBCI)

## Consumer-PC Memory & Framework Adaptations
1. **Host RAM Wall (56 GB → 26.9 GB)**: Replaced eager in-memory fold concatenation (`step3_make_synthetic.py`) with `numpy.memmap` streaming and lazy on-demand Kaldi `TLG.fst` lattice generation (`code/latgen_lazy.cc.txt`, `code/step6_decode_bigram.py`).
2. **Framework Migration**: Ported legacy TensorFlow 1.15 RNN training & HMM forced alignment to TensorFlow 2.20 (`compat.v1`) and PyTorch (`code/torch_port/`).

## Reproduction Summary (`results/`)
- **Acoustic / Character Stage (No LM)**: Raw CER **4.71%** (held-out blocks) / **5.10%** (held-out trials) vs. paper **2.78%**.
- **Full Pipeline (Kaldi Bi-gram FST + GPT-2 Rescoring)**: WER **2.21%** (held-out blocks) / **3.22%** (held-out trials) vs. paper **3.45%**.
