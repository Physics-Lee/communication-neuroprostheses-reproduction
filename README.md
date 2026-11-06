# Communication Neuroprostheses — Consumer-GPU (16 GB VRAM / 64 GB RAM) Reproduction Suite

[![Verify & Render Benchmark](https://github.com/Physics-Lee/communication-neuroprostheses-reproduction/actions/workflows/verify_benchmark.yml/badge.svg)](https://github.com/Physics-Lee/communication-neuroprostheses-reproduction/actions/workflows/verify_benchmark.yml)

End-to-end reproduction code, host-RAM / GPU-VRAM wall mitigations, per-trial decode outputs, and benchmark figure generator for all **8 open-data large-vocabulary (>71,400 word forms) communication neuroprosthesis studies** on a single consumer workstation (**1× NVIDIA RTX 5060 Ti 16 GB VRAM, 64 GB DDR5 RAM**).

![Communication Neuroprostheses Reproduction Benchmark](communication_neuroprostheses_reproduction.png)

---

## Overview & Directory Naming Convention (`yyyymmdd_author`)

Each study directory is named by its **official journal issue publication date (`yyyymmdd`) + first author surname**, providing a strict chronological ordering without arbitrary numbering or suffix collisions:

| Study Directory | Publication | Modality & Participant(s) | Dataset DOI | Official vs. Our Memory Wall Mitigation | Acoustic / Synthesis Stage (Ours vs. Paper) | Full-Pipeline WER (Ours vs. Paper) |
|---|---|---|---|---|---|---|
| [`20210512_willett/`](20210512_willett/) | Willett et al., *Nature* 593, 249–254 (2021-05-12) | Handwriting (T5) | [`10.5061/dryad.wh70rxwmv`](https://doi.org/10.5061/dryad.wh70rxwmv) | 56 GB → **26.9 GB RAM** (`mmap` fold assembly + lazy Kaldi FST) | Raw CER: **4.71%** vs. 2.78% | **2.21%** vs. 3.45% (bi-gram + GPT-2) |
| [`20230831_willett/`](20230831_willett/) | Willett et al., *Nature* 620, 1031–1036 (2023-08-31) | Speech (T12) | [`10.5061/dryad.x69p8czpq`](https://doi.org/10.5061/dryad.x69p8czpq) | 300 GB 5-gram WFST / 80 GB A100 → **1.46 GB KenLM trie** (or 19.6 GB `mmap`) & **13.0 GB VRAM** | Greedy PER: **23.42%** vs. 19.7% | **23.22%** vs. 23.8% (5-gram + OPT-6.7B) |
| [`20231201_fan/`](20231201_fan/) | Fan et al., *Nat. Commun.* 14, 7965 (2023-12-01) | Handwriting CORP (T5) | [`10.5061/dryad.hqbzkh1p6`](https://doi.org/10.5061/dryad.hqbzkh1p6) | 40 GB rig → **4.0 GB VRAM** (seed RNN + 403-day online pseudo-label adaptation) | CORP v6 CER: **6.20%** vs. 6.16% | **6.10%** vs. 6.20% (bi-gram + GPT-2) |
| [`20240815_card/`](20240815_card/) | Card et al., *N. Engl. J. Med.* 391, 609–618 (2024-08-15) | Speech (T15) | [`10.5061/dryad.dncjsxm85`](https://doi.org/10.5061/dryad.dncjsxm85) | 300 GB 5-gram WFST → **19.6 GB RAM** (`mmap` + `.idx`) & **13.0 GB VRAM** | Greedy PER: **10.19%** vs. 9.54% | **7.02%** vs. 3.40% (10-model ens.) |
| [`20250612_wairagkar/`](20250612_wairagkar/) | Wairagkar et al., *Nature* 642, 417–424 (2025-06-12) | Voice synthesis (T15) | [`10.5061/dryad.dncjsxm85`](https://doi.org/10.5061/dryad.dncjsxm85) | 3×24 GB RTX 3090 + A6000 → **1.5 GB VRAM** (causal 40-bin Mel encoder) | Spectral distortion $1-r$: **31.4%** ($r=0.69$) vs. 17.0% ($r=0.83$) | Voice synthesis (acoustic stage only) |
| [`20250821_kunz/`](20250821_kunz/) | Kunz et al., *Cell* 188, 4675–4689 (2025-08-21) | Inner & attempted speech (T12, T15, T16) | [`10.5061/dryad.m905qfv9c`](https://doi.org/10.5061/dryad.m905qfv9c) | 48 GB VRAM / 60 GB WFST → **13.0 GB VRAM** & **1.46 GB KenLM** | Greedy PER: **61.4% / 59.7% / 61.5%** | **122.1% / 79.5% / 95.5%** (sparse 1–3 sess. inner-speech subsets) |
| [`20260316_jude/`](20260316_jude/) | Jude et al., *Nat. Neurosci.* (2026-03-16) | Touch typing (T17, T18) | [`10.5061/dryad.cz8w9gjjk`](https://doi.org/10.5061/dryad.cz8w9gjjk) | 64+ GB RAM → **22.5 GB RAM** (bypass unpruned FST preload) & **11.5 GB VRAM** | Raw CER: **16.5% (T17) / 5.3% (T18)** | **26.8% (T17) / 2.2% (T18)** vs. 12.5% / 1.6% |
| [`20260428_jude/`](20260428_jude/) | Jude et al., *Cell Rep.* 45, 117167 (2026-04-28) | Intended speech in LIS (T17) | [`10.5061/dryad.vq83bk481`](https://doi.org/10.5061/dryad.vq83bk481) | 48 GB VRAM / 300 GB WFST → **13.0 GB VRAM** & **19.6 GB RAM** | Greedy PER: **39.97%** vs. 34.0% | **46.47%** vs. ~50% (conversational 125k) |

---

## Repository Structure

```text
communication-neuroprostheses-reproduction/
├── 20210512_willett/                     # Willett et al., 2021 (Nature, handwriting T5)
├── 20230831_willett/                     # Willett et al., 2023 (Nature, speech T12)
├── 20231201_fan/                         # Fan et al., 2023 (Nat. Commun., 403-day handwriting CORP T5)
├── 20240815_card/                        # Card et al., 2024 (NEJM, speech & rapid calibration T15)
├── 20250612_wairagkar/                   # Wairagkar et al., 2025 (Nature, voice synthesis T15)
├── 20250821_kunz/                        # Kunz et al., 2025 (Cell, inner speech T12/T15/T16)
├── 20260316_jude/                        # Jude et al., 2026a (Nat. Neurosci., touch typing T17/T18)
├── 20260428_jude/                        # Jude et al., 2026b (Cell Rep., intended speech in LIS T17)
├── shared/                               # Shared C++ TLG/WeNet decoders, KenLM/OPT/Qwen rescorers, plots
├── reproduction_benchmark.yaml           # Authoritative structured dataset for Figure 2 (a)-(h)
├── reproduction_benchmark.json           # Auto-synced JSON mirror of reproduction_benchmark.yaml
├── session_per_vs_wer.json               # Session-level (PER, post-LM WER) pairs for Figure 2 (i)
├── build_session_per_vs_wer.py           # Recomputes session_per_vs_wer.json from per-trial decode TSVs
└── render_figure2.py                     # Standalone renderer + overlap verifier for Figure 2 (.png/.pdf)
```

---

## Quick Verification (No GPU Required)

All per-trial decode outputs (`*.tsv`, `*.csv`, `*.json`) produced by our single-GPU runs are archived inside each study's `results/` directory. Anyone can recompute the session-level phase diagram (`session_per_vs_wer.json`) and regenerate Figure 2 in seconds:

```bash
pip install -r requirements.txt
python build_session_per_vs_wer.py
python render_figure2.py
```

---

## Companion Repository

- **Weekly-Updated Living Timeline (Figure 1)**: [`Physics-Lee/communication-neuroprostheses-timeline`](https://github.com/Physics-Lee/communication-neuroprostheses-timeline)

## License

Released under the [MIT License](LICENSE). Official upstream repositories remain under their respective authors' licenses and are referenced (not redistributed) in each study directory.
