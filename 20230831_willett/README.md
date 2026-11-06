# `20230831_willett` — Willett et al., *Nature* 2023 (High-Performance Speech Neuroprosthesis, T12)

- **Paper**: Willett, F. R. et al. A high-performance speech neuroprosthesis. *Nature* **620**, 1031–1036 (2023-08-31). [doi:10.1038/s41586-023-06377-x](https://doi.org/10.1038/s41586-023-06377-x)
- **Dataset**: Dryad [`doi:10.5061/dryad.x69p8czpq`](https://doi.org/10.5061/dryad.x69p8czpq)
- **Upstream Code**: [`fwillett/speechBCI`](https://github.com/fwillett/speechBCI)

## Consumer-PC Memory & LLM Rescorer Adaptations
1. **Host RAM Wall (300 GB 5-gram WFST → 1.46 GB / 19.6 GB)**: Provided both a 1.46 GB KenLM 3-gram binary trie decoder (`code/decode_words_btot24_v2.py`, 41× RAM reduction) and a 19.6 GB `mmap` + `.idx` Kaldi 5-gram WFST loader (`../shared/tools/tlg_viterbi.cc`).
2. **GPU VRAM Wall (80 GB A100 → 13.0 GB RTX 5060 Ti)**: 5-layer GRU trained in FP16 (`code/train_btot24.py`, 3.5 GB VRAM); second-pass rescoring evaluated with OPT-6.7B (FP16, 13.0 GB VRAM) and modern 4-bit Qwen2.5-7B / Qwen3-8B (`../shared/tools/opt_rescore.py`).

## Reproduction Summary (`results/decode_outputs/`)
- **Acoustic Stage (No LM)**: Greedy PER **23.42%** vs. paper **19.7%**.
- **Full Pipeline (125k 5-gram WFST + OPT-6.7B)**: WER **23.22%** (with Qwen3-8B base: **22.14%**) vs. paper **23.8%**.
