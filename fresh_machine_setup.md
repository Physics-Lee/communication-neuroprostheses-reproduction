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
│   └── splits/                     # tracked CSVs: source partitions and reproduction split rules (see Q5)
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

Two things compute on plain Windows before any of the WSL pieces exist: the eight notebooks' archive-scoring cells recompute the archived metrics — greedy rates *and* the archived WERs — from files shipped with the repository (3–25 s each, ~1.5 min for all eight), and the fresh greedy metrics of the table above. Scoring any decode TSV is one shared command:

```text
python shared/tools/wer_from_tsv.py --no-sidecar <any final.tsv>
```

Fresh post-LM WER is the one metric that does not exist without the WSL toolchain — see Q6 for why and how, and Q5 for comparing it with the paper.

## 5. How do I use the same test trials as the paper to compare WER?

**Use the same individual neural recordings for both results.** The same dates, the same number of trials, or even the same sentence texts are insufficient. Each reference and prediction must be linked to the original recording by session, source file, block/partition, and trial index. A sentence can have been recorded more than once.

Two tasks must be distinguished:

- **Reproduce an archived offline score:** run on the trials that a saved offline result used. This checks your implementation against our archive.
- **Compare offline reproduction with the paper's online decoding:** run on the trials used for that particular online result. Matching our offline archive does not establish this second comparison.

For Figure 2, the intended comparison uses **every online trial selected for Figure 2c**. This released-data cohort is not automatically identical to every trial behind the paper's printed headline WER. State which result and cohort you are reproducing before quoting a comparison.

**A. How the online and offline evaluation sets differ, study by study**

Online evaluation scores the output produced during the experiment. Its training and calibration history belongs to the online protocol; it need not coincide with a later offline `train`/`test` split. For an adapting decoder, preserve the original predict/update order: adapting after a trial is different from training on it before scoring it.

The table combines trial selection, offline split rules, and coverage. Study links lead to the available split lists and their definitions.

- **Fig2c WER trials:** released trials actually included in our Figure 2c WER calculation, not the paper's total online trials or all trials in the release.
- **Fig2f WER trials:** archived offline trials included in Figure 2f on those online dates, not the number of raw recordings available. The two counts are not an overlap fraction.

| Study | Participant | Fig2c selection | Fig2c WER trials | Offline split | Fig2f WER trials | Trial match |
| --- | --- | --- | ---: | --- | ---: | --- |
| [Willett 2021](studies/20210512_willett/data/splits/README.md) | T5 | Closed-loop copy typing; 5 days | 163 | `HeldOutTrials` | 33 | Different |
| [Willett 2023](studies/20230831_willett/data/splits/README.md) | T12 | Switchboard Stage 2; attempted speaking; 5 days | 400 | Released `test`; used for validation | 200 | Different |
| [Fan 2023](studies/20231201_fan/data/splits/README.md) | T5 | Recalibration sessions; 10 days | 377 | Replay: final 6 trials/session | — | Unavailable |
| [Card 2024](studies/20240815_card/data/splits/README.md) | T15 | 125k-word copy task; 14 Table S3 dates | 1,628 | Released validation HDF5; BiGRU arm | 514 | Different |
| [Wairagkar 2025](studies/20250612_wairagkar/data/splits/README.md) | T15 | Listener-transcription benchmark | — | Acoustic evaluation; separate listener indices | — | Unavailable |
| [Kunz 2025](studies/20250821_kunz/data/splits/README.md) | T12 | Closed-loop inner speech; 50 words; 1 day | 100 | Condition-specific `_eval` inputs | 100 | Matched |
| [Kunz 2025](studies/20250821_kunz/data/splits/README.md) | T15 | Closed-loop inner speech; 125k words; 2 days | 55 | Condition-specific `_eval` inputs | 55 | Matched |
| [Kunz 2025](studies/20250821_kunz/data/splits/README.md) | T16 | Closed-loop inner speech; 125k words; 1 day | 35 | Condition-specific `_eval` inputs | 35 | Matched |
| [Jude 2026a](studies/20260316_jude/data/splits/README.md) | T17 | Figure 3b,c blocks; 6 days | 161 | Seed-9 shuffle; last `num_test_trials` | 104 | Different |
| [Jude 2026a](studies/20260316_jude/data/splits/README.md) | T18 | Figure 3b,c blocks; 7 days | 580 | Seed-9 shuffle; last `num_test_trials` | 140 | Different |
| [Jude 2026b](studies/20260428_jude/data/splits/README.md) | T17 | Unconstrained-vocabulary blocks; 4 days | 469 | Last eligible sentence block/day | 131 | Different |

**Matched** means the same individual trials have been verified. **Different** means the current trial sets differ. **Unavailable** means there is no corresponding offline WER comparison. For Wairagkar, the online count is left as a dash because its listener-median benchmark is handled separately; the offline dash indicates no reproduced listener WER.

For example, the selected Jude 2026a T17 blocks contain **165 released trials**, but **161 contribute to Figure 2c WER**: four have no recoverable decoder output. The table reports 161, not 165. Counts for the selected released blocks and the scored trials are recorded separately as `released_trials` and `wer_trials` in the [daily source CSV](figures/results/20261124_repository_migration/figure2cd_daily.csv).

Details that affect how to interpret this table:

- **Willett 2021:** `HeldOutTrials` has 63 archived trials overall; only five of its 33 trials on the online dates match online trials. The separate `HeldOutBlocks` partition assigns all 163 online trials to test, and its decode archive already matches 158 of them.
- **Offline archive totals:** Willett 2023 has 880 trials over 24 sessions; Card has 1,426 trials; Jude 2026b has 417 trials. These totals span a different cohort from the online-date counts above. All 469 Jude 2026b online recordings are present in the local exported data: our split assigns 131 to validation and 338 to training.
- **Separate experiments:** Fan's 60-trial replay uses a character LM; Card rapid calibration and Kunz T16 three-fold block holdout are separate experiments. Wairagkar's Mel correlation is not listener-transcription WER.
- **Selection exposure:** Jude 2026a's held-out cues were used for checkpoint validation. Kunz's OPT weights were selected on the scored evaluation trials. Matching trial identities does not make these untouched test benchmarks.

For the six unmatched groups, Figure 2f and Table 2 currently have date alignment only. They do **not** yet supply an identical-test-set performance comparison. The [online daily counts](figures/results/20261124_repository_migration/figure2cd_daily.csv), [offline daily counts](figures/results/20261124_repository_migration/figure2f_daily.csv), and [trial-pair records](figures/results/20261124_repository_migration/figure2cd_paired_trials.csv) document the current coverage. Trial-pair records can refer to a different model or decoder from the plotted arm; they are not a complete online test manifest. See the [evaluation protocol](figures/docs/20261124_evaluation_protocol/figure2cd.md) for source selections and known release-versus-paper differences.

**B. How to prepare the correct test split for a new run**

1. **Identify the target online trials first.** Use the original session/block/trial identifiers and references for the selected online result. The selection rules and source fields are in [extract_online_trials.py](figures/code/extract_online_trials.py). That script exports daily totals; a daily CSV alone is not a trial list. Freeze the complete target list before training. Do not reduce it to the predictions already in our archive and call it the original test set.
2. **Map those trials to the neural inputs.** Use the common session files from Q2 and their `trial_id`, `source_file`, `source_block` where present, and `source_index` attributes. Establish an unambiguous source mapping across formats; IDs from two different formats may have different strings. Repeated prompt text alone cannot identify a recording.
3. **Assign training, validation, and test roles explicitly.** The target trials must be test trials for the offline run, excluded from offline fitting, checkpoint selection, and LM parameter tuning. Use suitable other data for training and validation, and document differences from the original online training/calibration protocol. If an existing checkpoint used target trials for fitting or selection, scoring it again does not create an independent test result: a suitable new training/selection run is needed.
4. **Infer and decode every target trial.** Verify exactly one offline prediction and the same reference for each online test trial. Check missing, extra, and duplicate source IDs; all three counts must be zero. Preserve the trial/reference/prediction mapping and any chronological order needed for adaptation. If predictions are missing, distinguish missing recordings, preprocessing exclusions, missing inference, and missing decoding before deciding what to rerun.

The repository's current default offline splits do **not** implement all these changes automatically. For the unmatched groups, a newcomer can reproduce the existing archive using the tutorial, but cannot yet obtain a complete Figure 2c trial-matched comparison just by running that tutorial unchanged.

**C. What the existing split checks prove**

After Q2's session export, inspect the study's `data/splits/README.md`. Its partition manifest records trial roles; `<release>__archived_evaluation.csv` records a particular archived offline evaluation. Neither file should be assumed to be the paper's complete online test list.

For example, run from the repository root:

```text
python studies/20260428_jude/code/build_split_manifest.py --require-all
```

This checks exported **offline validation references against the offline archive**, including coverage of every archived session. A successful 417-row check does not prove equality with the 469 online trials. Willett 2023 and Fan also support `--require-all`. Jude 2026a checks the conversion's TFRecords and archived references for covered sessions. Willett 2021 writes publisher partition assignments; Card writes rapid-calibration roles; Kunz writes condition/fold assignments and checks fold inputs when present; Wairagkar writes evaluation roles and checks listener-index interpretation. Run those scripts without `--require-all` and read their coverage or skip messages. **There is no existing single command that certifies complete online/offline test equality for all eight studies.**

**D. How to compare the two WERs after the test trials match**

Score both outputs against the same references with the same text normalization. For each day, calculate `WER (%) = 100 * total word edits / total reference words`. Across the common evaluation days, report the equally weighted mean and sample SD (denominator `N - 1`); omit SD for a single day. Label pooled WER separately. PER, CER, and Mel correlation are not WER.

Record the actual test list, training/validation roles, checkpoint, decoder, word-error counts, and reference-word counts alongside the result. A printed paper WER is a direct numerical comparator only when its test coverage and calculation match; otherwise report the online WER recomputed on the stated released cohort and identify it as such. The [acceptance report](docs/20261124_repository_migration/overnight_reader_acceptance_report.md) verifies pipeline runs and fresh-versus-archive scores, not completion of the trial matching above.

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
