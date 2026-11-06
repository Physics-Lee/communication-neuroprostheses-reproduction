# `shared` — Cross-Study C++ WFST Decoders, KenLM/LLM Rescorers & Finetune Utilities

Shared infrastructure used across the speech and handwriting reproduction pipelines:

- **`tools/tlg_viterbi.cc` & `tools/wenet_lattice.cc`**: Memory-mapped (`mmap` + `.idx`) C++ Kaldi OpenFST 5-gram WFST beam-search decoders that reduce host RAM peak from **300 GB** to **19.6 GB**.
- **`tools/opt_rescore.py` & `tools/qwen_rank_probe.py`**: Second-pass N-best lattice rescorers supporting Facebook OPT-6.7B (FP16, 13.0 GB VRAM) and 4-bit Qwen2.5-7B / Qwen3-8B on a single 16 GB GPU.
- **`tools/wer_from_tsv.py` & `tools/compare_rescorers.py`**: Case-insensitive, punctuation-normalized word error rate (WER) evaluators matching the official Brain-to-Text sidecar convention.
- **`finetune_chain/` & `vacation_tlg/`**: Cross-day adaptation and hyperparameter sweep scripts for acoustic scale, word insertion penalty (WIP), and LLM interpolation weights ($\alpha$, $\beta$).
