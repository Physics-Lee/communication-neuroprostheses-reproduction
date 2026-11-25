# Repository migration plan

2026-11-24

## Repository layout

All eight studies use `studies/<yyyymmdd>_<author>/` and the same public path and filename templates below. Create a stage directory when the study has an input or output for that stage. See the [eight-study path table](naming_map.md#data-paths) for the files each study currently uses.

```text
studies/<yyyymmdd>_<author>/
├── README.md
├── AGENTS.md
├── tutorial.ipynb
├── code/
│   ├── README.md
│   ├── prepare_data.py
│   ├── train.py
│   ├── infer.py
│   ├── decode.py
│   └── evaluate.py
├── open_source_code_for_reference/<repo>/
├── data/
│   ├── sources.yaml
│   ├── raw/<release>/<release>__<publisher_filename>
│   ├── extracted/<release>/<publisher_path>
│   ├── splits/<release>__<split_id>.csv
│   ├── processed/<participant>/<release>_<session>.h5
│   ├── derived/<release>/<format>/<variant>/
│   ├── checkpoints/<release>/<asset_id>/
│   └── language_models/<decoder>/
├── results/<run_id>/
│   ├── run.yaml
│   ├── trials.csv
│   └── summary.json
├── log/<run_id>/
└── docs/<yyyymmdd>_<topic>/
```

Each study exposes the public commands for the stages it runs. Native datasets and trainer packages keep the internal filenames required by their readers. Archived results are cited by their existing filenames. Cross-study code and figures live in `shared/` and `figures/` at the repository root.

## Release, format, and decoder names

- **`<release>`** is the verified provider release ID when available. Until the local version is verified, use the dataset ID, such as `dryad_wh70rxwmv`, in paths and file prefixes and record `release_id: null` in provenance. Save a verified provider version as part of `release_id`, for example `dryad_x69p8czpq_20230901`. `sources.yaml` records the dataset DOI and the source path.
- **`<format>`** is the input format the command actually reads, such as `hdf5`, `tfrecord`, `npz`, or `zip`. `.h5` is the filename extension for the common session export. Native trainer inputs and experiment-specific folds use `data/derived/` as they are prepared for that layout. Willett 2021 uses memory-mapped `.npy` for its synthetic folds.
- **`<decoder>`** is the decoding method, for example `char_start`, `ctc_greedy`, `wfst_bigram`, `kenlm_3gram`, `char_5gram_beam`, or `wfst_5gram`. Write `none` in the run directory name when the run has no text decoder. Record `decoder`, `rescorer`, `vocoder`, and the exact asset and vocabulary in the run record as applicable. For example, a Wairagkar acoustic run can use `decoder: null` and `vocoder: lpcnet` when it renders audio.

For new published runs, use `<run_id> = <yyyymmdd>_<release>__<format>__<decoder>__<experiment>`. For example, a Willett 2023 greedy run can publish to `results/20261124_dryad_x69p8czpq__hdf5__ctc_greedy__checkpoint_eval/`. The run record identifies the input, split, checkpoint, and scoring configuration.

## Willett 2023 on disk

| Role | Current path within the study |
| --- | --- |
| Released input | `data/extracted/dryad_x69p8czpq/competitionData/` |
| Native trainer input | `data/derived/dryad_x69p8czpq/hdf5/trainer_input/` |
| Native trainer index | `data/derived/dryad_x69p8czpq/hdf5/dryad_x69p8czpq__session_description.csv` |
| Archived evaluation list | `data/splits/dryad_x69p8czpq__archived_evaluation.csv` |
| Decoder assets | `data/language_models/kenlm_3gram/`, `data/language_models/wfst_5gram/` |
| Common session file | `data/processed/t12/dryad_x69p8czpq_2022-05-19.h5` |

Keep a downloaded `competitionData.tar.gz` at `data/raw/dryad_x69p8czpq/dryad_x69p8czpq__competitionData.tar.gz` if the archive is available locally. The extracted tree and both decoder assets already exist in this checkout. Existing model workspaces such as the local `run_60k` still use their historical paths. The [naming map](naming_map.md) records the current paths for the other seven studies.

## Common session files

Every study exports one `.h5` file per recording session in the same layout, defined in [`shared/session_format.py`](../../shared/session_format.py) (schema `bci-session`, version 1). A file stores each trial's neural features as released, its task targets, and its timing, and names the publisher file and position the trial came from. A numeric target that names a label set holds zero-based positions in it. Split assignments stay in `data/splits/` lists that reference `trial_id`, so different experiments can share the exported trials.

| Step | Command from the repository root |
| --- | --- |
| Export one session | `python studies/<study>/code/export_sessions.py --session <id>` |
| List the sessions of a study and their export state | `python studies/<study>/code/export_sessions.py --list` |
| Check a file and trace every trial to its source | `python shared/tools/validate_session.py <file> --source trials` |
| Check every exported file | `python shared/tools/validate_session.py` |
| Rebuild a study's split lists from its exported files | `python studies/<study>/code/build_split_manifest.py` |
| List the export state of every session of every study | `python shared/tools/session_export_inventory.py` |

The file name is `<release>_<session>.h5` inside `data/processed/<participant>/`. The session part is the native identifier without its participant code and with dots replaced by dashes: `t12.2022.05.19` gives `t12/dryad_x69p8czpq_2022-05-19.h5`, `t15_day195` gives `t15/dryad_2280gb64f_day195.h5`, and `t12.2023.11.28_attempted_50words` gives `t12/dryad_gf1vhhn1j_2023-11-28_attempted_50words.h5`. [`shared/study_paths.py`](../../shared/study_paths.py) builds the path with `session_export_path`. Session files are local data and are not tracked by Git.

The format was piloted on one text-decoding study and one voice-synthesis study: [Willett 2023](pilot_willett2023_session_export.md) and [Wairagkar 2025](pilot_wairagkar2025_session_export.md), then applied to the other six studies. The [export inventory](session_export_inventory.md) lists the state of every session of every study, and [onboarding a study](onboarding_a_study.md) shows how a new study registers its source, exports one session, and validates it with the same command.

## Implementation order

1. **Unify public names.** Use the release-scoped paths and standard output names. Record verified provider versions as provenance becomes available.
2. **Unify session exports.** Pilot the common `.h5` on a Willett 2023 text session and a Wairagkar voice session. Validate trial IDs, counts, targets, and native timing. Store one file per session and place trial roles, fold assignments, and adaptation order in the split list. Record a schema version and stable source trial IDs in the exports.
3. **Exercise new-study onboarding.** Add a study using the common names, source manifest, split manifest, validated session export, run record, and tutorial. Recompute a result before adding it to the benchmark.

## Upstream code

Each official repository lives under its study's `open_source_code_for_reference/`, using the upstream repository name. The [upstream manifest](../../shared/upstream_sources.json) records the origin, pinned commit, runtime consumers, and any patch. Card's `nejm-brain-to-text` checkout is shared by Willett 2023, Card, Kunz, and Jude 2026b. Its [runtime patch](../../studies/20240815_card/code/upstream_patches/nejm-brain-to-text.patch) preserves the two locally modified training files.

Run `python shared/tools/setup_upstream.py` from the repository root to install all eight checkouts. Use `--study 20230831_willett` to install that study's reference checkout and its runtime dependencies. Use `--check` to verify existing checkouts offline. The full-runtime instructions in each study tutorial use these pinned paths.

Four historical source snapshots have no recorded Git commit. The manifest distinguishes those unknown historical commits from the commits selected for this setup. The Wairagkar and Jude 2026a runtime files were compared against those pinned checkouts. On Windows, the three repositories with bundled native language-model code use sparse checkouts for the C, C++, and Java source files in those subtrees. Their Git commits retain the source objects.

## Migration rules

- **Code.** Cross-study tools live in `shared/`.
- **Data.** Git tracks `sources.yaml` and split lists. Keep large inputs local.
- **Paths.** Tutorials document the paths the current scripts accept. Apply the target layout one study at a time and update the scripts, manifest, and tutorial together. The study directory is the default base. `BCI_DATA_ROOT` can supply the same layout on another disk. `BCI_ARCHIVE_ROOT` selects the historical checkout layout explicitly.
- **Runs and figures.** Put new outputs and trained checkpoints in `results/<run_id>/` and logs in `log/<run_id>/`. Record old-to-new paths when moving published result files. Use `figures/reproduction_benchmark.yaml` to select figure inputs and regenerate the derived tables and renders with the figure code.

## Migration and verification record

- The original checkouts are under `D:\workspace\communication_neuroprostheses_archive\`. The Willett 2023 pilot copied 63 competition files and matched every SHA-256 hash. The subsequent same-volume moves relocated 22,834 files totaling 443.71 GiB.
- The naming migration introduced 33 public commands and 22 standardized paths across the eight studies. The Willett 2023 pilot converted and evaluated 20 test trials through the public commands. Its 245 phoneme errors over 637 reference phonemes match the published trial archive for that session.
- The eight studies passed input checks through the new paths. Representative conversions succeeded for Willett 2023, Kunz 2025, and Jude 2026b. All eight zero-download notebooks and the Figure 2 and 3 rebuilds passed. The [archived file manifest](archived_files.csv) records the 252 verified result and log files.
- The release-scoped naming update verified 26 local renames across the eight studies, including seven evaluation lists. The current and historical input paths resolved for all eight studies. The Willett 2023 public commands rebuilt and evaluated the 20-trial sample from the new data paths, matching all input tensors, trial predictions, and 245 errors over 637 phonemes. Figure 2 and Figure 3 rebuilt with identical numerical tables. All 252 archived artifact hashes matched.
- Full GPU retraining and WSL 5-gram decoding are outside these verification results. Verify the affected commands and zero-download tutorial each time a study adopts the target subdirectories.
- The common session format (schema `bci-session`, version 1) is exported by all eight studies through `studies/*/code/export_sessions.py`. 131 session files hold 28,724 trials in 28.46 GiB. Not exported are 35 Wairagkar days, the 11 Fan seed sessions, and 3 Card days outside the archived evaluation; Jude 2026b's `t17.2024.03.25` and `t17.2024.05.23` are excluded because they hold no sentence block. The export took 29 GiB and left 103.7 GiB free on the data disk. The [export inventory](session_export_inventory.md) lists every session.
- `python shared/tools/validate_session.py --source trials` re-reads every trial from its publisher file and compares it with the stored trial. It passed for all 131 files, checked in batches between 08:03 and 10:04 on 2026-11-25: each of the 28,724 stored trials is identical to the trial re-read from its publisher file, and the provenance attributes match. A full pass reads every publisher file again, so it takes tens of minutes on a hard disk; run it when no other job reads the same disk.
- Each study's split lists were rebuilt from the exported files by `studies/*/code/build_split_manifest.py` and checked against the archived evaluation lists: Willett 2023 (24 sessions, 880 validation trials), Willett 2021 (63 `HeldOutTrials` test sentences), Fan 2023 (60 holdout rows), Card 2024 (five calibration days in file order), Kunz 2025 (three T16 folds, each matching its existing fold input), Jude 2026a (16 sessions, each matching its TFRecord conversion, 275 test trials), and Jude 2026b (12 sessions, 417 validation sentences).
- All eight study tutorials pass `python shared/tools/check_study_tutorials.py --run`: data setup comes first, every path is a download target, an existing input, or a command output, and every public command answers `--help`. They execute without errors through `python shared/tools/run_notebook.py` and store the output of the cell that reads their session file. The six general tutorials also executed without errors on temporary copies. Figure 2 and Figure 3 were rebuilt into a temporary directory: the nine published tables are byte-identical and no tracked figure file changed.
- Differences to keep in mind. The Kunz T12 session files hold the continuous 128-electrode arrays, while `convert_kunz.py` trains on the publisher's 8-bit per-trial copy with another electrode order; for T15 and T16 the two are identical in all 1,170 trials. The Fan release states no bin size; its files use 20 ms, the size the figure code assumes, which the trial lengths support. Label ids are zero-based in every study; Fan's released ids are one-based and are shifted by its exporter.
