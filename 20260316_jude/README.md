# `20260316_jude` — Jude et al., *Nature Neuroscience* 2026a (Touch-Typing Neuroprosthesis, T17 & T18)

- **Paper**: Jude, J. et al. A touch-typing neuroprosthesis for rapid communication in people with paralysis. *Nat. Neurosci.* (2026-03-16). [doi:10.1038/s41593-026-02216-y](https://doi.org/10.1038/s41593-026-02216-y)
- **Dataset**: Dryad [`doi:10.5061/dryad.cz8w9gjjk`](https://doi.org/10.5061/dryad.cz8w9gjjk)
- **Upstream Code**: [`justin-jude/typing-bci`](https://github.com/justin-jude/typing-bci)

## Consumer-PC Memory & PyTorch Port Adaptations
1. **Host RAM Wall (64+ GB → 22.5 GB)**: Bypassed eager preloading of the unpruned 5-gram character/word FST during beam-search initialization (`code/nightly_S6_build_lm_decoder.sh`).
2. **PyTorch BF16 Port (`code/pytorch_port/`, 11.5 GB Peak VRAM)**: Native PyTorch port of the character-level GRU decoder across all released T17 and T18 sessions.
3. **Exact CPM / WPM Recomputation (`code/g1_recompute_cpm.py`)**: Verified per-block communication rates directly from released trial timestamps and Source Data tables.

## Reproduction Summary (`results/`)
- **T18 (Intact Speech, SCI)**: Raw CER **5.3%** (paper $<12\%$); post-LM WER **2.2%** vs. paper **1.6%**.
- **T17 (Severe Dysarthria / LIS, ALS)**: Best-session raw CER **16.5%** vs. paper Source Data **18.6%**; post-LM WER **26.8%** vs. paper **12.5%**.
