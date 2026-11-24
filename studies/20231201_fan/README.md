# `studies/20231201_fan` — Fan et al., *NeurIPS 2023* (403-Day Plug-and-Play Handwriting CORP, T5)

Start with the [study notebook](tutorial.ipynb) and the [evaluation and input guide](docs/20261124_evaluation_protocol/README.md).

- **Paper**: Fan, C. et al. Plug-and-Play Stability for Intracortical Brain-Computer Interfaces: A One-Year Demonstration of Seamless Brain-to-Text Communication. In *Advances in Neural Information Processing Systems 36 (NeurIPS 2023)*, 42258–42270. [doi:10.52202/075280-1831](https://doi.org/10.52202/075280-1831) · [proceedings page](https://proceedings.neurips.cc/paper_files/paper/2023/hash/83a14a36de4502bac5b580db36e81858-Abstract-Conference.html)
- **Dataset & Pretrained Seed Weights**: Dryad [`doi:10.5061/dryad.hqbzkh1p6`](https://doi.org/10.5061/dryad.hqbzkh1p6)
- **Upstream Code**: [`cffan/CORP`](https://github.com/cffan/CORP)

## Consumer-PC Reproduction Pipeline
- **`code/tier2_seed_model_inference.py`**: Evaluates the released pretrained seed RNN across all 10 follow-up sessions (403 days).
- **`code/tier3_corp_sim_v6.py`**: Replays batch recalibration over the ten released follow-up sessions using the authors' online transcripts as pseudo-labels (peak 4.0 GB VRAM).
- **`code/rescore_official_outputs.py`**: Computes WER from the authors' released online transcripts.

## Reproduction Summary (`results/`)
- **Frozen Seed Baseline**: Raw CER **2.65%** (Day 7) drifting to **34.71%** (Day 403) without adaptation.
- **Batch adaptation replay (v6)**: Mean greedy CER **6.20%** across ten sessions. The last six trials used for scoring were eligible for adaptation through the released pseudo-labels. See [the v6 archive](results/20261124_import_corp_archive/tier3_corp_sim_v6.json).
- **Held-out adaptation replay (v7)**: Mean greedy CER **12.4%**, with the last six trials of each session excluded from adaptation. See [the v7 archive](results/20261124_import_corp_archive/tier3_corp_sim_v7.json).
- **Author online WER**: The paper reports **6.16%** for CORP. Rescoring the released online subset gives **4.10% pooled WER**. The earlier README claim of **6.10% reproduction WER** has no supporting output in the v6 archive. See [the paper](https://proceedings.neurips.cc/paper_files/paper/2023/file/83a14a36de4502bac5b580db36e81858-Paper-Conference.pdf) and [the online rescoring archive](results/20261124_import_corp_archive/rescore_official_outputs.json).
- **Matched-trial character-LM ablation**: The CUDA replay with a seed-trained character 5-gram gives **42.25% mean daily WER** versus **1.50%** for the released online outputs on the same 60 trials. The paper's online pipeline uses a 130,000-word Kaldi 3-gram followed by GPT-2 XL. See [the matched-trial results](../../figures/docs/20261124_evaluation_protocol/figure2cd.md#fan-corp-held-out-replay).
