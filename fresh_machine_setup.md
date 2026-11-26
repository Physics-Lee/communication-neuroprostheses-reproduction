# Fresh machine setup

Everything a new reader needs between `git clone` and the first metric, in one place. The per-study commands live in each study's `tutorial.ipynb` (all eight follow the same four-step "Prepare the data" structure); this page maps what to do first, in what order, and what each piece costs in time and disk.

The per-study path and command reference is [naming_map.md](naming_map.md); the shared reader guide with all Windows/WSL command details is [reader_workflow.md](reader_workflow.md).

## 1. Windows environment (once, ~30 minutes)

Follow [the reader guide](reader_workflow.md): `conda env create -f environment.yml`, activate `bci-repro`, register the notebook kernel, and install the pronunciation resources for the three speech conversions:

```powershell
python -m nltk.downloader cmudict averaged_perceptron_tagger averaged_perceptron_tagger_eng
```

Then install the upstream reference checkouts once with `python shared/tools/setup_upstream.py` (or per study with `--study <id>`; `--check` verifies existing ones offline).

When testing the documented default paths, clear `BCI_DATA_ROOT`, `BCI_ARCHIVE_ROOT`, and `BCI_RUN_ID` first; the tutorials assume them unset.

## 2. What to download and where it goes

All eight studies keep publisher archives at `studies/<study>/data/raw/<release>/<release>__<publisher_filename>`, extract to `data/extracted/<release>/`, convert to `data/derived/...`, and export session files to `data/processed/<participant>/`. The exact commands are each Notebook's "Prepare the data" section; the table is the one-page view.

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

The two `LanguageModels` archives are only needed for fresh word decodes; every study's greedy metric and every archived metric runs without them.

## 3. Metrics without any WSL toolchain

Two things work on plain Windows with nothing else installed beyond step 1:

1. **All eight notebooks' archive-scoring cells** recompute the published metrics — greedy error rates *and* the archived word-level WERs — from the files shipped with the repository. Total time for all eight: about a minute and a half.
2. **Greedy metrics from fresh short runs**: each "Reproduce and score" train command at the Notebook's documented reader length (1000–2000 batches) plus `evaluate.py` needs nothing but this environment. The fresh post-LM WER needs the step-4 toolchain.

For example, scoring an archived decode TSV directly:

```text
python shared/tools/wer_from_tsv.py --no-sidecar studies/20230831_willett/results/20261124_import_btot24/wl_5g_opt_official.tsv
```

## 4. WSL toolchain for the word-level WER (required for the headline metric, slowest to build — start it early)

The word error rates are the studies' headline metrics: every per-study "Calculate metrics" section ends by scoring a fresh word decode, and the benchmark figures pair post-LM WER with raw PER/CER. Greedy error rates come out on Windows (step 3), but **fresh post-LM WER does not exist without this toolchain**: only the 5-gram/bigram/typing decodes and LPCNet synthesis run in WSL.

**Why WSL at all?** Three concrete reasons, not preference:

1. **The word decoders are Linux-native C++ linked against Kaldi/OpenFST.** The speech 5-gram path is `wenet_lattice.cc` (mmap + `.idx` decoder built on Kaldi libraries); the handwriting bigram is `latgen_lazy2`; the typing 5-gram is the authors' `lm_decoder` (built against their pinned `speechBCI`, PyTorch 1.13.1). The upstream `speechBCI`/`LPCNet` build systems only support Linux — LPCNet has no Windows build at all.
2. **The papers' graphs exceed Windows-friendly tooling.** The speech `TLG.fst` is a 42 GB OpenFST binary the decoder must mmap with random access; the typing graph is 16 GB. The repository's memory-wall design keeps them on disk (never loaded whole), and that mmap machinery lives in the Linux binaries above.
3. **The Windows GPU environment stays clean.** Acoustic training/inference run natively on Windows CUDA; keeping the C++ decoders in WSL means one environment never bloats the other. `wsl.exe` calls them straight from PowerShell, and the decoders write plain TSVs back to the shared checkout.

**How to use it day-to-day**: everything runs from the Windows repository root — PowerShell scripts call `wsl.exe` for you (e.g. `step6_decode_ours.py` shells out to `latgen_lazy2`; `decode_wfst.py` wraps `wenet_lattice`). You only enter WSL yourself to build the toolchain once and to run the shared decoder/typing decode commands in [reader_workflow.md](reader_workflow.md) (both checkouts are visible at `/mnt/<drive>/...`, verified to read and write the same files).

It is the slowest piece to build (hours, dominated by Kaldi), so start the WSL install below while the Windows passes of step 5 run, rather than after them. Install Ubuntu, then build the pieces in [reader_workflow.md](reader_workflow.md): Kaldi + OpenFST (then `wenet_lattice` and `latgen_lazy2`), [LPCNet](reader_workflow.md#voice-synthesis), and the typing decoder's `py310lm` + `lm_decoder` (including the eight `runtime/server/x86` symlink relinks and the CMake 4 / GCC 15 flags documented there).

For Kaldi itself, follow the upstream `tools/INSTALL` and `src/INSTALL` (clone, `extras/check_dependencies.sh` and its `apt-get` line, `make` in `tools/`, then `./configure --shared && make` in `src/`). Two honest notes from the acceptance run:

- The reference machine carries a **prebuilt** tree (`~/kaldi` with `tools/openfst` and `tools/openfst-1.8.4`); the reader commands point at those paths. If your Kaldi builds a different OpenFST version, adjust the two compile commands' `-I`/`-L` paths to match.
- The acceptance run **did not build Kaldi from scratch**; it only compiled `lm_decoder` (GCC 15 needed `-Wno-template-body`, documented in the reader guide). On a brand-new Ubuntu whose compiler rejects older Kaldi sources, a slightly older GCC via `apt install g++-12` with `CC/CXX` pointing at it is the usual remedy — untested here. Until the toolchain is ready, step 3 covers every study's greedy and archived metrics, including the archived WERs.

Two graph-placement rules that the acceptance run measured the hard way (details in the reader guide):

- Copy each decoding graph to the WSL home filesystem first (`~/tg5` for speech, `~/typing_tg` for typing); random graph reads through `/mnt/<drive>` are effectively unusable.
- Run decodes serially; the shared speech graph's page cache (~20 GB, up to ~60 GB with both graphs resident) is per-host. `wsl --shutdown` returns it to Windows immediately.

## 5. Suggested first-pass order (times measured on the acceptance run)

Start the step-4 WSL build in a side terminal before pass 1; the order below assumes it finishes during the Windows passes.

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

## 6. Test splits: what the paper held out, what you hold out, and how the WERs compare

Every study ships its evaluation split as tracked CSVs under `data/splits/` — an `archived_evaluation.csv` listing exactly which trials each archived score covered (with reference text), plus a **partition manifest recording the paper's own train/test assignment trial by trial** (`train_test_partitions.csv`, `source_partitions.csv`, `paired_v7_holdout.csv`, `calibration_order.csv`, `conditions.csv`/`t16_folds.csv`, `validation_blocks.csv`, `test_trials.csv`, `evaluation_trials.csv` — one or two per study, named in each `data/splits/README.md`). Both kinds are rebuilt and re-verified by each study's `build_split_manifest.py`, not hand-copied. A per-study [`docs/20261124_evaluation_protocol/README.md`](docs/20261124_evaluation_protocol/README.md) (one folder up from each tutorial) states which trials the paper's figure scores. Three facts connect them:

1. **The reproduction reuses the paper's own held-out trials — it does not invent a new split.** Each publisher release carries its partition: the released `HeldOutTrials`/`HeldOutBlocks` MAT files (Willett 2021), the release's `test`/`validation` partition (Willett 2023, Card, Jude 2026b), the withheld last-six-trials of each session (Fan), the per-day evaluation blocks (Kunz, Wairagkar), and the conversion manifest's held-out blocks (Jude 2026a). The conversion and export commands read those partitions, so the `train_val_trials.json` your `train.py` writes and the trials your `infer.py` scores are the paper's trials by construction. The per-study `data/splits/README.md` documents each column.
2. **To compare against the paper's WER, decode and score exactly those trials.** The per-study "Read the archived results" notebook cell shows the reference WER for the paper's evaluation arm; a fresh run replaces its input logits, not its split. Concretely: `infer.py`/`dump_logits` exports only the split's trials (W2023: 880 validation; Card: 1426 validation; Jude 2026b: 417 validation; Kunz T12-mixed eval: 100; Willett 2021: the 63 `HeldOutTrials` sentences), the WSL decode scores that same TSV, and `wer_from_tsv.py` prints the comparable number. `data/splits/<release>__archived_evaluation.csv` is the audited row-for-row list of what each archived score covered — check a fresh `trials.csv` against it when in doubt.
3. **Match convergence before comparing numbers.** The documented 1000–2000-batch reader runs are smoke-length: they verify the chain end-to-end and land far above the paper (e.g. Jude 2026b at 1000 batches decodes to WER 83.3% where the archived full training reaches 50.6%; Kunz's 1000-batch decode even exceeds 100%, as insertion-heavy unconverged CTC can). The paper-comparable WER comes from the score-bearing run lengths named in each [evaluation protocol](docs/20261124_evaluation_protocol/README.md) — e.g. W2023's archived 60k-batch checkpoint (WER 25.7% first-pass; 24.9% re-decoded on this chain) and Card's released checkpoint (8.47% vs its archived 8.70%). Reproduce those lengths (or load the released/`runtime/` checkpoints) when the comparison is the point; keep the reader-length runs for pipeline checks.

Two studies use a different headline metric and are compared the same way: Wairagkar correlates predicted vs target Mel bands on the released evaluation days (paper r ≈ 0.83 after removing silences; this suite's day-level means: 0.47–0.69), and the paper's listener-transcription WER is a separate archived Figure 2c value that the notebook recomputes from the released listening-test data.

The acceptance report ([overnight_reader_acceptance_report.md](docs/20261124_repository_migration/overnight_reader_acceptance_report.md)) tabulates, per study, the fresh WER next to the archived WER for exactly this comparison.

## Disk and memory budget

Measured components on the acceptance machine: the full eight-study data tree (extracted + derived + sessions + raw) was ~444 GiB before the language models; the speech `wfst_5gram` tree adds ~127 GB on the data disk (incl. the 75 GB `G_no_prune.fst`, which only N-best rescoring uses), the typing graph ~18 GB; WSL-side working copies add ~47 GB (`~/tg5` + `.idx`) + 16 GB (`~/typing_tg`) inside the WSL virtual disk. Budget ~650–700 GB free, or park per-study data on a second disk with `BCI_DATA_ROOT` (the tutorials document it).

The WSL virtual disk grows with the graph copies and does not shrink automatically; `wsl --shutdown` releases the page cache to Windows at once, and compacting the VHDX needs the standard export/import or `Optimize-VHD` route.
