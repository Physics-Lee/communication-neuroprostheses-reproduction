# Fresh-machine setup

A new reader cloning this repository has six questions. This page answers them in order — one section per question — so the page doubles as an acceptance checklist: if every section below leads you to a command you can run, the repository is ready for a new machine. All times and sizes come from the 2026-11-25/26 clean-checkout acceptance run ([report](docs/20261124_repository_migration/overnight_reader_acceptance_report.md)).

Three companions hold the details this page maps to: the [reader guide](docs/20261124_repository_migration/reader_workflow.md) (shared Windows/WSL commands), the [naming map](docs/20261124_repository_migration/naming_map.md) (every public command, path, and archived result per study), and each study's `tutorial.ipynb` (the canonical per-study commands).

**Before question 1: the environment (once, ~30 minutes).** Follow [the reader guide](docs/20261124_repository_migration/reader_workflow.md): `conda env create -f environment.yml`, activate `bci-repro`, register the notebook kernel, install the pronunciation resources (`python -m nltk.downloader cmudict averaged_perceptron_tagger averaged_perceptron_tagger_eng`), and install the upstream checkouts with `python shared/tools/setup_upstream.py`. When testing the documented default paths, clear `BCI_DATA_ROOT`, `BCI_ARCHIVE_ROOT`, and `BCI_RUN_ID` first; the tutorials assume them unset.

## 1. Will I see a unified file tree?

Yes — all eight studies use one layout, so everything you learn in one study transfers to the next seven:

```text
studies/yyyymmdd_author/            # eight of these, structurally identical
├── tutorial.ipynb                  # THE per-study guide: prepare data → reproduce → score
├── code/                           # train.py · infer.py · decode.py · evaluate.py (same names everywhere)
├── data/
│   ├── raw/<release>/              # publisher archives land here, never modified
│   ├── extracted/<release>/        # unmodified extraction
│   ├── derived/<release>/...      # converted trainer inputs (study-specific format)
│   ├── processed/<participant>/    # common session files (one .h5 per recording session)
│   └── splits/                     # tracked CSVs: the papers' train/test assignments (see Q5)
├── results/<run_id>/               # run.yaml + trials.csv (+ summary.json) per dated run
├── log/<run_id>/
└── docs/                           # per-study evaluation protocol and code guide
```

Two facts make the tree predictable: every directory is named from the same `<yyyymmdd>_<author>` + `<release>` + `<format>` + `<decoder>` scheme ([naming rules](docs/20261124_repository_migration/README.md#repository-layout)), and every new result you generate goes into a dated `results/<run_id>/` with a `run.yaml` that records what produced it. Nothing a fresh run writes ever overwrites an archived baseline.

## 2. Where do I download the data, where do I unpack it, and which code converts it?

Each tutorial's **"Prepare the data"** is the same four steps, and every command writes into the directories above — download → extract → convert → export:

| Step | Command family | Reads → writes |
| --- | --- | --- |
| 1. Download | manual, from the publisher DOI | → `data/raw/<release>/<release>__<publisher_filename>` |
| 2. Extract | `tar -xzf` / `python -m zipfile -e` (exact command in each tutorial) | → `data/extracted/<release>/` |
| 3. Convert | `code/prepare_data.py` (Card and Wairagkar read the release directly and skip this) | → `data/derived/<release>/<format>/...` |
| 4. Export sessions | `code/export_sessions.py --session <id>` | → `data/processed/<participant>/<release>_<session>.h5` |

The eight downloads (all Dryad; the tutorial's step 1 names the same files):

| Study | Download (Dryad) | Save as (under `data/raw/<release>/`) | Size measured |
| --- | --- | --- | --- |
| Willett 2021 | `handwritingBCIData.tar.gz` ([10.5061/dryad.wh70rxwmv](https://doi.org/10.5061/dryad.wh70rxwmv)) | `dryad_wh70rxwmv__handwritingBCIData.tar.gz` | — |
| Willett 2023 | `competitionData.tar.gz` ([10.5061/dryad.x69p8czpq](https://doi.org/10.5061/dryad.x69p8czpq)) | `dryad_x69p8czpq__competitionData.tar.gz` | — |
| Willett 2023 (WER only) | `languageModel_5gram.tar.gz` (same release) | `dryad_x69p8czpq__languageModel_5gram.tar.gz` | ~83 GB archive + graph |
| Fan 2023 | `CORP_data_release.zip` ([10.5061/dryad.hqbzkh1p6](https://doi.org/10.5061/dryad.hqbzkh1p6)) | `dryad_hqbzkh1p6__CORP_data_release.zip` | — |
| Card 2024 | `t15_copyTask_neuralData.zip` ([10.5061/dryad.dncjsxm85](https://doi.org/10.5061/dryad.dncjsxm85)) | `dryad_dncjsxm85__t15_copyTask_neuralData.zip` | 11 GB |
| Card 2024 (optional) | `t15_pretrained_rnn_baseline.zip` (same release) | `dryad_dncjsxm85__t15_pretrained_rnn_baseline.zip` | 463 MB |
| Wairagkar 2025 | `t15_neural_data.zip` ([10.5061/dryad.2280gb64f](https://doi.org/10.5061/dryad.2280gb64f)) | `dryad_2280gb64f__t15_neural_data.zip` | 25 GB |
| Kunz 2025 | `sentenceData.zip` ([10.5061/dryad.gf1vhhn1j](https://doi.org/10.5061/dryad.gf1vhhn1j)) | `dryad_gf1vhhn1j__sentenceData.zip` | 3.7 GB |
| Jude 2026a | `Data.zip` + `LanguageModels.zip` ([10.5061/dryad.cz8w9gjjk](https://doi.org/10.5061/dryad.cz8w9gjjk)) | `dryad_cz8w9gjjk__<name>.zip` | graph inside ≈18 GB |
| Jude 2026b | `Data.zip` ([10.5061/dryad.vq83bk481](https://doi.org/10.5061/dryad.vq83bk481)) | `dryad_vq83bk481__Data.zip` | — |

The two `LanguageModels` archives are only needed for fresh word decodes (Q6); every study's greedy metric and every archived metric runs without them. After step 4, verify the export with one command — it re-reads every trial from its publisher file and compares:

```text
python shared/tools/validate_session.py <study>/data/processed/<participant>/<file>.h5 --source trials
```

## 3. Which code runs the converted data?

`code/train.py` reads the converted input and `code/infer.py` exports per-trial predictions — the same two command names in all eight studies (the [naming map](docs/20261124_repository_migration/naming_map.md) lists the mapping to each underlying script). The reader-length form of each, as documented in the tutorials and measured in the acceptance run:

| Study | `train.py` reads | Reader-length command (measured time) |
| --- | --- | --- |
| Willett 2021 | synthetic `.npy` folds | `train.py --partition HeldOutTrials --synth --nBatches 2000` (≈22 min) |
| Willett 2023 | trainer HDF5 | `train.py 1000 RUN 1000` (6 min) |
| Fan 2023 | released seed TFRecords — no fresh training | replay runs the released model directly |
| Card 2024 | released HDF5 directly | `train.py 1000 RUN 1000` (≈6 min at archived rate) |
| Wairagkar 2025 | released ZIP directly | released Keras encoders run via `infer.py` (no training) |
| Kunz 2025 | trainer HDF5 | `train.py t12 mixed 1000 RUN 500` (3.5 min) |
| Jude 2026a | per-session NPZ | `train.py --participant t17` (joint) or `pytorch_port/train.py --npz FILE` (single) |
| Jude 2026b | trainer HDF5 | `train.py 1000 RUN 500` (21 min) |

Then `infer.py` exports only the evaluation split's predictions (880 trials for W2023, 1426 for Card, 417 for Jude 2026b, 100 for Kunz, 63 sentences for W2021 — see Q5 for why those exact counts). Full arguments for every command are each tutorial's "Reproduce and score" section.

## 4. Which code computes the metrics?

`code/evaluate.py` — again the same name everywhere. What it prints and what else each study needs:

| Study | `evaluate.py` prints | Fresh WER comes from |
| --- | --- | --- |
| Willett 2021 | raw CER/WER from errCounts | WSL bigram decode (`step6_decode_ours.py` → `latgen_lazy2`) |
| Willett 2023 | pooled greedy PER | WSL 5-gram decode + `wer_from_tsv.py` |
| Fan 2023 | pooled WER (greedy / char-5gram / online) directly | built in — Windows CPU |
| Card 2024 | PER during training | WSL 5-gram decode + `wer_from_tsv.py` |
| Wairagkar 2025 | Mel-band correlations (Tier-1) | `evaluate.py` → WSL LPCNet synthesis + Mel r |
| Kunz 2025 | greedy PER by vocab | WSL 5-gram decode + `wer_from_tsv.py` |
| Jude 2026a | CER (`final_report.json` / `multisession_summary.json`) | WSL typing 5-gram (`decode.py`, py310lm) |
| Jude 2026b | greedy PER (CL/OL) | WSL 5-gram decode + `wer_from_tsv.py` |

Two things compute on plain Windows before any of the WSL pieces exist: the eight notebooks' archive-scoring cells recompute the published metrics — greedy rates *and* the archived WERs — from files shipped with the repository (3–25 s each, ~1.5 min for all eight), and the fresh greedy metrics of the table above. Scoring any decode TSV is one shared command:

```text
python shared/tools/wer_from_tsv.py --no-sidecar <any final.tsv>
```

Fresh post-LM WER is the one metric that does not exist without the WSL toolchain — see Q6 for why and how, and Q5 for comparing it with the paper.

## 5. How did the paper split the test set, how does my reproduction split it, and how do the two WERs compare?

**What the paper did** is recorded trial by trial in tracked CSVs under each `data/splits/`: one `archived_evaluation.csv` listing exactly which trials each archived score covered (with reference text), plus a **partition manifest with the paper's own train/test assignment** (`train_test_partitions.csv`, `source_partitions.csv`, `paired_v7_holdout.csv`, `calibration_order.csv`, `conditions.csv`/`t16_folds.csv`, `validation_blocks.csv`, `test_trials.csv`, `evaluation_trials.csv` — named per study in each `data/splits/README.md`). Both are rebuilt and cross-verified by the study's `build_split_manifest.py`, not hand-copied. The per-study [evaluation protocol](studies/20230831_willett/docs/20261124_evaluation_protocol/README.md) (each study has one at `studies/<study>/docs/20261124_evaluation_protocol/`) states which of those trials the paper's figure scores.

**What the reproduction does**: it reuses the paper's split — it never invents one. Each release ships its partition (W2021's released `HeldOutTrials`/`HeldOutBlocks` MAT files; the `test`/`validation` partitions of W2023, Card, Jude 2026b; Fan's withheld final six trials per session; Kunz's and Wairagkar's evaluation-day blocks; Jude 2026a's held-out blocks), and the conversion/export/train/infer commands read those partitions by construction. You do nothing to "create" the split: `train.py` writes a `train_val_trials.json` naming the paper's train and eval trials, and `infer.py` exports only the eval side.

**How the WERs compare** — three rules:

1. **Same trials.** The fresh decode scores exactly the trials the archived score did (the counts in Q3; `data/splits/<release>__archived_evaluation.csv` is the audited row-for-row list to check a fresh `trials.csv` against).
2. **Same convergence.** The 1000–2000-batch reader runs are smoke-length: they prove the chain but land far above the paper (Jude 2026b at 1000 batches → WER 83.3%; the archived full training → 50.6%; Kunz's short decode even exceeds 100%, as unconverged CTC can). For the paper comparison, use the score-bearing run lengths named in each study's evaluation protocol (`studies/<study>/docs/20261124_evaluation_protocol/README.md`), or load the released/`runtime/` checkpoint — e.g. Card's released checkpoint decodes to 8.47% against its archived 8.70%, and W2023's archived 60k checkpoint re-decodes to 24.9% against its archived 25.7%.
3. **Two studies headline a different metric, compared the same way**: Wairagkar correlates predicted vs target Mel bands on the released evaluation days (paper r ≈ 0.83 after removing silences; this suite's day-level means 0.47–0.69), and the paper's listener-transcription WER is a separate archived Figure 2c value the notebook recomputes from the released listening-test data.

The [acceptance report](docs/20261124_repository_migration/overnight_reader_acceptance_report.md) tabulates the fresh WER next to the archived WER for every study.

## 6. Why does part of this run in WSL — and how do I use it?

The word error rates are the headline metrics (Q5), and **fresh post-LM WER exists only in WSL**: the 5-gram/bigram/typing decodes and LPCNet synthesis are Linux-native. Three concrete reasons, not preference:

1. **The word decoders are Linux-native C++ linked against Kaldi/OpenFST.** The speech 5-gram path is `wenet_lattice.cc` (mmap + `.idx` decoder on Kaldi libraries); the handwriting bigram is `latgen_lazy2`; the typing 5-gram is the authors' `lm_decoder` (their pinned `speechBCI`, PyTorch 1.13.1). Upstream `speechBCI`/`LPCNet` build systems support Linux only — LPCNet has no Windows build at all.
2. **The papers' graphs exceed Windows-friendly tooling.** The speech `TLG.fst` is a 42 GB OpenFST binary the decoder must mmap with random access; the typing graph is 16 GB. The repository's memory-wall design keeps them on disk (never loaded whole), and that mmap machinery lives in the Linux binaries above.
3. **The Windows GPU environment stays clean.** Acoustic training/inference run natively on Windows CUDA; the C++ decoders live in WSL, so neither environment bloats the other.

**How to use it day-to-day**: everything runs from the Windows repository root — the PowerShell scripts call `wsl.exe` for you (`step6_decode_ours.py` shells out to `latgen_lazy2`; `decode_wfst.py` wraps `wenet_lattice`). You enter WSL yourself only to build the toolchain once and to run the shared decoder/typing commands in the [reader guide](docs/20261124_repository_migration/reader_workflow.md); both sides see the same checkout at `/mnt/<drive>/...` (verified to read and write the same files).

It is the slowest piece to build (hours, dominated by Kaldi), so start it in a side terminal while the Windows passes below run. Build order and the pitfalls the acceptance run hit (all documented in the reader guide): Kaldi + OpenFST (then `wenet_lattice`, `latgen_lazy2`), [LPCNet](docs/20261124_repository_migration/reader_workflow.md#voice-synthesis), and the typing decoder's `py310lm` + `lm_decoder` (eight `runtime/server/x86` symlink relinks; CMake 4 needs `CMAKE_POLICY_VERSION_MINIMUM=3.5`; GCC 15 needs `-Wno-template-body`). For Kaldi itself, follow the upstream `tools/INSTALL` and `src/INSTALL`. Two honest notes from the acceptance run:

- The reference machine carries a **prebuilt** tree (`~/kaldi` with `tools/openfst` and `tools/openfst-1.8.4`); the reader commands point at those paths. A different Kaldi build means adjusting the compile commands' `-I`/`-L` paths to match.
- The acceptance run **did not build Kaldi from scratch**; it only compiled `lm_decoder`. On a brand-new Ubuntu whose compiler rejects older Kaldi sources, an older GCC (`apt install g++-12`, `CC`/`CXX` pointing at it) is the usual remedy — untested here. Until the toolchain is ready, every greedy and archived metric — including the archived WERs — is still computable on Windows (Q4).

Two graph-placement rules measured the hard way: copy each decoding graph to the WSL home filesystem first (`~/tg5` for speech, `~/typing_tg` for typing) — random graph reads through `/mnt/<drive>` are effectively unusable — and run decodes serially (the speech graph's page cache is ~20 GB, up to ~60 GB with both graphs resident; `wsl --shutdown` returns it to Windows at once).

## Suggested first evening (measured order)

Start the Q6 WSL build in a side terminal before pass 1; the order assumes it finishes during the Windows passes.

| Pass | What | Wall time measured |
| --- | --- | --- |
| 1 | All eight notebooks' archive-scoring cells, including the archived WER tables (zero download) | 3–25 s each |
| 2 | Fan one-session replay (released model, 6 held-out trials) | minutes |
| 3 | W2023 reader train 1000 batches → infer 880 → greedy eval | 6 min + <1 min + <1 min |
| 4 | Kunz t12-mixed reader train 1000 batches → infer/eval 100 | 3.5 min + <1 min |
| 5 | Jude 2026b reader train 1000 batches → infer 417 → eval | 21 min + ~1 min |
| 6 | W2021 synth train 2000 batches → infer → bigram decode 63 | ≈22 min + minutes + 17.7 min |
| 7 | Wairagkar Tier-1 released-encoder inference (2 days, CPU) → LPCNet Mel eval | ≈48 min + ≈20 min |
| 8 | Card reader train 1000 batches | ≈6 min at the archived rate (87 min measured while three other jobs shared the disk) |
| 9 | Fresh post-LM WER — first 5-gram decode after `~/tg5` copy + `.idx`: W2023 880 / Card 1426 / Jude26b 417 / Kunz 100 | 23 / 19 / 17 / 6 min |
| 10 | Typing post-LM decode: graph load ≈2–3 min per process, then seconds per session | — |

One-time costs: `.idx` index build ≈22 min for the speech graph; large-graph copies to the WSL home filesystem ≈20–30 min each.

## Disk and memory budget

Measured components on the acceptance machine: the full eight-study data tree (extracted + derived + sessions + raw) was ~444 GiB before the language models; the speech `wfst_5gram` tree adds ~127 GB on the data disk (incl. the 75 GB `G_no_prune.fst`, which only N-best rescoring uses), the typing graph ~18 GB; WSL-side working copies add ~47 GB (`~/tg5` + `.idx`) + 16 GB (`~/typing_tg`) inside the WSL virtual disk. Budget ~650–700 GB free, or park per-study data on a second disk with `BCI_DATA_ROOT` (the tutorials document it).

The WSL virtual disk grows with the graph copies and does not shrink automatically; `wsl --shutdown` releases the page cache to Windows at once, and compacting the VHDX needs the standard export/import or `Optimize-VHD` route.
