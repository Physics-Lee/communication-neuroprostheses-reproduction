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

1. **All eight notebooks' archive-scoring cells** recompute the published metrics from the files shipped with the repository. Total time for all eight: about a minute and a half.
2. **Greedy metrics from fresh short runs**: each "Reproduce and score" train command at the Notebook's documented reader length (1000–2000 batches) plus `evaluate.py` needs nothing but this environment. Fresh word-level WER from fresh logits needs step 4.

For example, scoring an archived decode TSV directly:

```text
python shared/tools/wer_from_tsv.py --no-sidecar studies/20230831_willett/results/20261124_import_btot24/wl_5g_opt_official.tsv
```

## 4. WSL toolchain for fresh word decodes (optional, slowest to build)

Only the 5-gram/bigram/typing decodes and LPCNet synthesis need WSL. Install Ubuntu, then build the pieces in [reader_workflow.md](reader_workflow.md): Kaldi + OpenFST (then `wenet_lattice` and `latgen_lazy2`), [LPCNet](reader_workflow.md#voice-synthesis), and the typing decoder's `py310lm` + `lm_decoder` (including the eight `runtime/server/x86` symlink relinks and the CMake 4 / GCC 15 flags documented there).

For Kaldi itself, follow the upstream `tools/INSTALL` and `src/INSTALL` (clone, `extras/check_dependencies.sh` and its `apt-get` line, `make` in `tools/`, then `./configure --shared && make` in `src/`). Two honest notes from the acceptance run:

- The reference machine carries a **prebuilt** tree (`~/kaldi` with `tools/openfst` and `tools/openfst-1.8.4`); the reader commands point at those paths. If your Kaldi builds a different OpenFST version, adjust the two compile commands' `-I`/`-L` paths to match.
- The acceptance run **did not build Kaldi from scratch**; it only compiled `lm_decoder` (GCC 15 needed `-Wno-template-body`, documented in the reader guide). On a brand-new Ubuntu whose compiler rejects older Kaldi sources, a slightly older GCC via `apt install g++-12` with `CC/CXX` pointing at it is the usual remedy — untested here. Until the toolchain is ready, step 3 keeps every study scorable.

Two graph-placement rules that the acceptance run measured the hard way (details in the reader guide):

- Copy each decoding graph to the WSL home filesystem first (`~/tg5` for speech, `~/typing_tg` for typing); random graph reads through `/mnt/<drive>` are effectively unusable.
- Run decodes serially; the shared speech graph's page cache (~20 GB, up to ~60 GB with both graphs resident) is per-host. `wsl --shutdown` returns it to Windows immediately.

## 5. Suggested first-pass order (times measured on the acceptance run)

| Pass | What | Wall time measured |
| --- | --- | --- |
| 1 | All eight notebooks' archive-scoring cells (zero download) | 3–25 s each |
| 2 | Fan one-session replay (released model, 6 held-out trials) | minutes |
| 3 | W2023 reader train 1000 batches → infer 880 → greedy eval | 6 min + <1 min + <1 min |
| 4 | Kunz t12-mixed reader train 1000 batches → infer/eval 100 | 3.5 min + <1 min |
| 5 | Jude 2026b reader train 1000 batches → infer 417 → eval | 21 min + ~1 min |
| 6 | W2021 synth train 2000 batches → infer → bigram decode 63 | ≈22 min + minutes + 17.7 min |
| 7 | Wairagkar Tier-1 released-encoder inference (2 days, CPU) → LPCNet Mel eval | ≈48 min + ≈20 min |
| 8 | Card reader train 1000 batches | ≈6 min at the archived rate (87 min measured while three other jobs shared the disk) |
| 9 | First 5-gram decode after `~/tg5` copy + `.idx`: W2023 880 / Card 1426 / Jude26b 417 / Kunz 100 | 23 / 19 / 17 / 6 min |
| 10 | Typing decode: graph load ≈2–3 min per process, then seconds per session | — |

One-time costs: `.idx` index build ≈22 min for the speech graph; large-graph copies to the WSL home filesystem ≈20–30 min each.

## Disk and memory budget

Measured components on the acceptance machine: the full eight-study data tree (extracted + derived + sessions + raw) was ~444 GiB before the language models; the speech `wfst_5gram` tree adds ~127 GB on the data disk (incl. the 75 GB `G_no_prune.fst`, which only N-best rescoring uses), the typing graph ~18 GB; WSL-side working copies add ~47 GB (`~/tg5` + `.idx`) + 16 GB (`~/typing_tg`) inside the WSL virtual disk. Budget ~650–700 GB free, or park per-study data on a second disk with `BCI_DATA_ROOT` (the tutorials document it).

The WSL virtual disk grows with the graph copies and does not shrink automatically; `wsl --shutdown` releases the page cache to Windows at once, and compacting the VHDX needs the standard export/import or `Optimize-VHD` route.
