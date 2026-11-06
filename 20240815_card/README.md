# `20240815_card` — Card et al., *New England Journal of Medicine* 2024 (Speech Neuroprosthesis in ALS, T15)

- **Paper**: Card, N. S. et al. An accurate and rapidly calibrating speech neuroprosthesis. *N. Engl. J. Med.* **391**, 609–618 (2024-08-15). [doi:10.1056/NEJMoa2314132](https://doi.org/10.1056/NEJMoa2314132)
- **Dataset**: Dryad [`doi:10.5061/dryad.dncjsxm85`](https://doi.org/10.5061/dryad.dncjsxm85)
- **Upstream Code**: [`Neuroprosthetics-Lab/nejm-brain-to-text`](https://github.com/Neuroprosthetics-Lab/nejm-brain-to-text)

## Consumer-PC Memory & Day-Index Calibration Fixes
1. **300 GB 5-gram WFST → 19.6 GB Resident RAM**: Memory-mapped OpenFST `TLG.fst` with pre-indexed state offsets (`../shared/tools/wenet_lattice.cc`).
2. **Day-1 Rapid Calibration Curve (`code/calib_online_sim.py`, `code/plot_rapid_calibration_curve.py`)**: Reproduces the sentence-by-sentence Day-1 online calibration trajectory on T15.
3. **Day-Index Alignment Fix (`results/decode_outputs/p4_fixed_opt_fx.tsv`)**: Corrects held-out session day-specific input layer routing across the 41 evaluation sessions.

## Reproduction Summary (`results/`)
- **Acoustic Stage (No LM)**: Greedy PER **10.19%** vs. benchmark baseline **9.54%**.
- **Full Pipeline (125k 5-gram WFST + OPT-6.7B)**: Single-seed WER **7.02%** vs. 10-model ensemble anchor **3.40%** (online paper aggregate **4.9%**).
