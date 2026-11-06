# `20260428_jude` — Jude et al., *Cell Reports* 2026b (Intended Speech in Locked-In Syndrome, T17)

- **Paper**: Jude, J. et al. Decoding intended speech with an intracortical brain-computer interface in a person with long-standing anarthria and locked-in syndrome. *Cell Rep.* **45**, 117167 (2026-04-28). [doi:10.1016/j.celrep.2026.117167](https://doi.org/10.1016/j.celrep.2026.117167)
- **Dataset**: Dryad [`doi:10.5061/dryad.vq83bk481`](https://doi.org/10.5061/dryad.vq83bk481)
- **Upstream Code**: [`justin-jude/anarthria-speech-bci`](https://github.com/justin-jude/anarthria-speech-bci)

## Consumer-PC Reproduction Pipeline
- **`code/convert_jude.py` & `code/train_jude.py`**: Extracts the 12 conversational 125,000-word vocabulary sessions of participant T17 and trains the 5-layer GRU phoneme decoder in 16 GB VRAM.
- **`code/eval_jude.py` & `results/decode_outputs/`**: Evaluates greedy PER, 3-gram KenLM decoding, and 125k 5-gram WFST + OPT-6.7B / Qwen3-8B rescoring.

## Reproduction Summary (`results/decode_outputs/`)
- **Acoustic Stage (No LM)**: Pooled greedy PER **39.97%** across 12 conversational sessions vs. paper **34.0%**.
- **Full Pipeline (125k 5-gram WFST + OPT-6.7B)**: Pooled WER **46.47%** (with Qwen3-8B rescoring: **45.2%**) vs. paper conversational benchmark **~50%**.
