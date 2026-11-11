# `20260316_jude` — Jude et al., *Nature Neuroscience* 2026a (Touch-Typing Neuroprosthesis, T17 & T18)

- **Paper**: Jude, J. et al. Restoring rapid natural bimanual typing with a neuroprosthesis after paralysis. *Nat. Neurosci.* **29**, 2613–2622 (2026-03-16). [doi:10.1038/s41593-026-02218-y](https://doi.org/10.1038/s41593-026-02218-y)
- **Dataset**: Dryad [`doi:10.5061/dryad.cz8w9gjjk`](https://doi.org/10.5061/dryad.cz8w9gjjk)
- **Upstream Code**: [`justin-jude/typingBCI`](https://github.com/justin-jude/typingBCI)

## Consumer-PC Memory & PyTorch Port Adaptations
1. **Host RAM Wall (64+ GB → 22.5 GB)**: Bypassed eager preloading of the unpruned 5-gram character/word FST during beam-search initialization (`code/nightly_S6_build_lm_decoder.sh`).
2. **PyTorch BF16 Port (`code/pytorch_port/`, 11.5 GB Peak VRAM)**: Native PyTorch port of the character-level GRU decoder across all released T17 and T18 sessions.
3. **Exact CPM / WPM Recomputation (`code/g1_recompute_cpm.py`)**: Verified per-block communication rates directly from released trial timestamps and Source Data tables.

## Reproduction Summary (`results/`)
- **T18 (Intact Speech, SCI)**: Multi-session + warm-start raw CER **7.69 ± 2.79%** across all 8 sessions (best **2.6%** on `12.04`, **6.0%** on `01.22` 20-sentence set; paper $<12\%$, Source Data **9.32 ± 2.41%**); post-LM CER **2.67 ± 1.76%** (best **0.0%** / **0.6%**); post-LM WER **6.66 ± 3.67%** across all 8 sessions (best **0.0%** / **2.4%**, 4-session copy-typing mean **5.17%**) vs. paper $<7\%$ (**6.97 ± 2.38%**).
- **T17 (Severe Dysarthria / LIS, ALS)**: Multi-session + warm-start raw CER **19.42 ± 5.80%** (best **8.2%**) vs. paper Source Data **18.6 ± 5.75%**; post-LM WER **11.41 ± 6.85%** across the 6 Fig. 3b online days (best **4.6%**, 5/8 sessions $<15\%$) vs. paper **12.5 ± 2.49%** (predominantly $<15\%$).
