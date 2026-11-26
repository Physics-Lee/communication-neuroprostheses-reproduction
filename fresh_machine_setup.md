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

| Step | Command family | Reads → writes | Consumed by |
| --- | --- | --- | --- |
| 1. Download | manual, from the publisher DOI | → `data/raw/<release>/<release>__<publisher_filename>` | step 2 |
| 2. Extract | `tar -xzf` / `python -m zipfile -e` (exact command in each tutorial) | → `data/extracted/<release>/` | step 3/4 |
| 3. Convert | `code/prepare_data.py` (Card and Wairagkar read the release directly and skip this) | → `data/derived/<release>/<format>/...` | **`train.py`/`infer.py` — the whole model→metric chain runs on this** |
| 4. Export sessions | `code/export_sessions.py --session <id>` | → `data/processed/<participant>/<release>_<session>.h5` | **not `train.py`** — the audit layer: `validate_session.py`, `build_split_manifest.py` (the split CSVs of Q5), and cross-study tooling |

Steps 3 and 4 both produce HDF5, but they are different files for different readers: step 3 keeps the upstream trainer's native layout (names like `input_features`, train/val files), step 4 re-stores the same trials in the repository's common schema with per-trial `source_*` provenance attributes. Verified in the acceptance run: every `train.py` reads step-3 outputs only (a historical `data/processed/hdf5` string inside the trainers is a path alias that resolves into `data/derived/`).

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

> **⚠️ Before quoting any paper-vs-reproduction WER comparison: "same test set" means same sessions *and* same trials — every individual trial, in the same order. Matching recording dates alone is NOT enough.** Verify the trial lists below first.

**Where to verify, per study — two tracked CSVs under `data/splits/`, plus one machine check:**

| Check | Where | What to compare |
| --- | --- | --- |
| Trials the archived score covered | `<release>__archived_evaluation.csv` (one row per scored trial: session, trial id, archive row, reference text) | your fresh `trials.csv` — session + reference text must match **row for row, in order** |
| The paper's own train/test assignment | the study's partition manifest (e.g. `<release>__validation_blocks.csv`, `__train_test_partitions.csv`; names listed in that folder's `README.md`), keyed by `trial_id` | your run's `train_val_trials.json` — its eval trial IDs must be this manifest's eval rows |
| **Automated check (use this)** | `python studies/<study>/code/build_split_manifest.py` | **fails** unless the exported evaluation trials equal the archived evaluation rows in content and order — run it instead of eyeballing |

The machine check is how the acceptance run itself confirmed trial-level equality (e.g. Jude 2026b: 417/417 rows identical in session and reference text). Each `data/splits/README.md` documents its study's columns, and each study's evaluation protocol (`studies/<study>/docs/20261124_evaluation_protocol/`) states which trials the paper's figure scores.

Short answer to the rest: **the paper's split ships inside its data release, the reproduction reuses it by construction, and the two WERs are comparable when the trial lists and the training length match.** The three sub-questions, each with where the answer lives:

| Sub-question | Answer | Where to look |
| --- | --- | --- |
| How did the paper split? | Recorded **trial by trial** in tracked CSVs: an `archived_evaluation.csv` (which trials each archived score covered, with reference text) + a partition manifest (the paper's own train/test rows with `trial_id`s) | each `data/splits/` + its `README.md` |
| How does my run split? | Identically — each release ships its partition, and convert/export/train/infer read it; you never create a split. `train.py` writes the exact train/eval trial IDs it used into `train_val_trials.json`; `infer.py` scores only the eval side | your run's `train_val_trials.json`; rebuild/verify with `build_split_manifest.py` |
| How do the WERs compare? | Only on the **same sessions *and* the same trials in the same order** (rule 1 below), at the same convergence (rule 2) | each study's `docs/20261124_evaluation_protocol/README.md` |

**The paper's split, per study** (all recorded in the partition CSVs named above):

| Study | The paper's held-out evaluation (the exact trials, not just the days) |
| --- | --- |
| Willett 2021 | released `HeldOutTrials` / `HeldOutBlocks` MAT partitions (63 held-out test sentences in `HeldOutTrials`) |
| Willett 2023 | the release's `test` partition — 880 labeled validation trials across 24 sessions |
| Fan 2023 | the final six trials of each session, withheld from CORP adaptation |
| Card 2024 | the release's validation split — 1426 labeled trials; calibration days in file order |
| Wairagkar 2025 | the released evaluation-day blocks (days 179/195 scored here; 20 trials per day for Mel eval) |
| Kunz 2025 | per-condition evaluation blocks (e.g. the T12 mixed eval: 100 trials); T16 uses 3 folds |
| Jude 2026a | the conversion manifest's held-out blocks (20 test trials per session × 8 sessions) |
| Jude 2026b | the release's validation partition — 417 sentences (highest-numbered block of each day) |

**Three rules for comparing your WER with the paper's:**

1. **Same sessions *and same trials* — check row by row, not by day.** A fresh decode scores exactly the trials the archived score did, and "same" means identical trial lists, not the same recording dates. Verify it with the pair of tracked CSVs: `data/splits/<release>__archived_evaluation.csv` is the audited row-for-row list of the trials each archived score covered (one row per trial, with reference text and archive row number), and the partition manifest (e.g. `validation_blocks.csv`) lists the same trials with their `trial_id`s. Compare a fresh `trials.csv` against the archived list — session and reference text must match row for row and in order. This is checked for you in one place: each study's `build_split_manifest.py` fails unless the export's evaluation trials equal the archived evaluation rows in content and order (verified again in the acceptance run, e.g. Jude 2026b: 417/417 rows identical).
2. **Same convergence.** The 1000–2000-batch reader runs are smoke-length: they prove the chain but land far above the paper. Real comparisons, as measured in the acceptance run:

| Study | Smoke-length WER (this chain) | Paper-comparable WER | How the comparable number was obtained |
| --- | --- | --- | --- |
| Willett 2023 | 48.4% greedy PER @1000 batches | 24.9% | decode the archived **60k-batch checkpoint** (archived first-pass: 25.7%) |
| Card 2024 | 40.0% val PER @1000 batches | **8.47%** WER | decode the **released checkpoint** (archived: 8.70%) |
| Jude 2026b | 83.3% WER @1000 batches | 50.6% | the archived **full-training** decode (25k batches) |
| Kunz 2025 (T12 mixed) | 138.6% WER @1000 batches | 106.9% | the archived full-training decode (this arm is insertion-heavy even at full training) |
| Fan 2023 | — | 0.0–43.8% | released seed model replays score directly (6/60 held-out trials for the one-session check) |

3. **Two studies headline a different metric, compared the same way**: Wairagkar correlates predicted vs target Mel bands on the released evaluation days (paper r ≈ 0.83 after removing silences; this suite's day-level means 0.47–0.69), and its listener-transcription WER is a separate archived Figure 2c value the notebook recomputes from the released listening-test data.

The [acceptance report](docs/20261124_repository_migration/overnight_reader_acceptance_report.md) tabulates the fresh WER next to the archived WER for every study.

## 6. Why does part of this run in WSL — and how do I use it?

Short answer: **the word decoders are Linux-native C++ with no Windows build, so fresh post-LM WER exists only in WSL; day-to-day you stay in PowerShell and the scripts call WSL for you — you only enter WSL yourself to build the toolchain once.**

| What runs where | Where | Why |
| --- | --- | --- |
| Data prep, conversion, session export | Windows | pure Python |
| Training + inference (greedy metrics) | Windows | native CUDA; PyTorch/TF |
| Archived metrics incl. archived WERs | Windows | notebook cells over shipped files |
| **5-gram speech decode, bigram, typing decode, LPCNet synthesis** | **WSL** | Linux-native C++: `wenet_lattice`/`latgen_lazy2`/`lm_decoder` link Kaldi+OpenFST; LPCNet has no Windows build at all; the 42 GB speech graph / 16 GB typing graph must be mmap'd, never loaded whole |

**How to use WSL** — two modes, only the first needs you inside Linux:

| Mode | When | What you do |
| --- | --- | --- |
| Scripts call WSL for you | every day | run PowerShell from the repository root — `step6_decode_ours.py` shells out to `latgen_lazy2`, `decode_wfst.py` wraps `wenet_lattice`; decoders write TSVs back into the same checkout (both sides share `/mnt/<drive>/...`) |
| You are inside WSL | once per machine, plus the shared speech/typing decode commands | build the toolchain (below), then run the reader guide's decoder commands |

**Build order and measured pitfalls** (each documented with exact commands in the [reader guide](docs/20261124_repository_migration/reader_workflow.md)):

| Piece | Measured pitfall (already in the guide) | One-time cost |
| --- | --- | --- |
| Kaldi + OpenFST | reference machine uses a prebuilt `~/kaldi`; from-scratch was **not** exercised by the acceptance run — if a new Ubuntu's compiler rejects old Kaldi, `apt install g++-12` + point `CC/CXX` at it (untested) | hours |
| `wenet_lattice` + `latgen_lazy2` | none beyond Kaldi being ready | minutes |
| LPCNet | Linux only, no Windows build exists | minutes |
| typing `py310lm` + `lm_decoder` | relink eight `runtime/server/x86` Git symlinks into `runtime/core`; CMake 4 → `CMAKE_POLICY_VERSION_MINIMUM=3.5`; GCC 15 → `-Wno-template-body`; cached glog zip was a 0-byte failed download | 1–2 h |
| Graph placement | copy graphs to the WSL home filesystem (`~/tg5`, `~/typing_tg`) — random reads through `/mnt/<drive>` are effectively unusable; decode serially (page cache ~20 GB speech, ~60 GB with both) | 20–30 min per copy; `.idx` build ≈22 min |

Until the toolchain is ready, every greedy and archived metric — including the archived WERs — is still computable on Windows (Q4).

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
