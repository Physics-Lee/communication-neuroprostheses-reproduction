# Twelve-hour reader acceptance plan

The goal is to let a reader start with a fresh checkout and the publisher downloads, follow any of the eight study Notebooks, and produce a metric in the shared file tree. Test the smallest meaningful run for each study. Use the corresponding evaluation protocol to identify the split, units, and expectations for a full benchmark run.

Start at [the reader guide](reader_workflow.md). Use [the repository layout](README.md#repository-layout) and [the eight-study naming map](naming_map.md) to check paths. The previous pass traced the eight example HDF5 files back to their publisher inputs, reproduced nine benchmark tables byte for byte, and generated fresh HDF5 files in clean directories for Fan 2023 and Willett 2023. The Fan clean-directory run also scored six held-out trials. Treat those as prior evidence and rerun an affected step if tonight's changes touch it. The Windows session returned `Wsl/E_ACCESSDENIED`, so give the first usable WSL machine to the Linux-dependent stages.

## Schedule

Times are elapsed from the start of the shift.

| Time | Work | Evidence to record |
| --- | --- | --- |
| 0–1 h | Record the candidate branch, commit and local changes. Prepare a clean checkout containing the candidate files, including any new files that have yet to be committed. Check Windows Python, available GPU/RAM/disk, and `wsl.exe --list --quiet`. If WSL works, verify it can read and write a temporary file in the same clean checkout. Begin checking decoder and vocoder prerequisites. | Candidate revision or patch, environment versions, free disk space, and exact Windows/WSL checkout paths. |
| 1–4 h | Run the clean start for the six studies still missing that evidence. Follow the Notebook through archive placement, extraction, upstream setup, conversion and HDF5 source trace for one example session. Use verified publisher downloads already on disk if available. Process the largest releases one at a time. | One trial-count receipt per study. Mark precisely where an unavailable download, dependency or disk limit stops a run. |
| 4–8 h | In WSL, run the shared indexed 5-gram decoder on a small labeled batch first, then run each of the other three speech inputs through it and score WER. Reuse the graph and its `.idx`. Also run one handwriting bigram decode, one typing decode from joint-session logits, and one Wairagkar voice-synthesis check. If WSL is blocked, proceed immediately with Windows inference, scoring and reader-path fixes. | Actual WER/CER numerators and denominators for text decoding, or finite Mel correlation and number of voiced trials for Wairagkar. Log missing WSL prerequisites per study. |
| 8–11 h | Finish the model-to-metric gap for each study. Prefer a short training run or the released checkpoint where the Notebook supports it. When a documented step fails, fix the reader-facing command or underlying code in the candidate, then rerun from the affected stage in a new run directory. Compare generated paths with the shared tree. | A fresh run directory and score, plus the command and exit code. Identify where an existing checkpoint or logits were used to test a later stage. |
| 11–12 h | Run the eight-Notebook checker, execute the archive-scoring cells, rebuild figures into a new temporary directory, and compare the published CSV/JSON tables. Review the final diff. | One eight-row acceptance report, the changed files, and the exact first failing command for each outstanding blocker. |

## Per-study acceptance

For every row, a reader must be able to locate the exact publisher archive, know where to put and unpack it, generate the tutorial's first HDF5, find the model input and output, and run the documented scoring command. Use `python shared/tools/validate_session.py <file> --source trials` and record the expected and observed trial counts.

| Study | Smallest useful model-to-metric check | WSL step |
| --- | --- | --- |
| Willett 2021 | Infer on `HeldOutTrials`. Score the raw predictions, then run the lazy bigram decoder on labeled trials and report CER/WER. | Handwriting bigram |
| Willett 2023 | Export speech logits, score greedy PER, then produce and score a fresh 5-gram decode TSV. | Shared indexed speech decoder |
| Fan 2023 | Restore the released seed model, run one adaptation update, and score the six held-out trials. Record the replay settings. | Windows CPU suffices |
| Card 2024 | Infer with the corrected day routing, then score PER and WER. Record the checkpoint source and split. | Shared indexed speech decoder |
| Wairagkar 2025 | Run the released encoder, synthesize audio with LPCNet, and calculate Mel-band correlation and `1 - r`. | LPCNet |
| Kunz 2025 | Use one labeled participant/condition, such as T12 mixed. Score greedy PER and indexed WER with that split recorded. | Shared indexed speech decoder |
| Jude 2026a | Generate the typing NPZ, run the single-session CER check, then score post-LM CER/WER from at least one joint-session logits file. Record the settings from the conversion manifest. | Typing 5-gram |
| Jude 2026b | Generate intended-speech logits, score greedy PER, then produce and score an indexed-WER TSV. | Shared indexed speech decoder |

## Working rules

- A clean checkout starts with the candidate code, installed dependencies, and the publisher archives staged at the paths in the Notebook. Record whether each archive was downloaded during the run or copied from a verified local publisher download. Clear `BCI_DATA_ROOT`, `BCI_ARCHIVE_ROOT`, and `BCI_RUN_ID` before testing defaults. Verify the same data paths in Windows and WSL when both are used.
- The test must run the documented commands. If only the converter or decoder can run, record which input was supplied and mark the preceding stages unverified. Archived metrics establish that the published results can be read and scored. Fresh results establish that a reader can generate new predictions and score them.
- Inspect the runtime outputs as well as the source tree. Publisher archives belong in `data/raw/<release>/`, unmodified extracted trees in `data/extracted/<release>/`, session files in `data/processed/<participant>/`, and training inputs in `data/derived/<release>/`. Use the dated run ID in `results/` and `log/` for new outputs. Keep tool-native files in the run's `runtime/` directory and record the trial table or summary the reader should open.
- Keep the 16 GB VRAM and 64 GB RAM limits. Use the shared memory-mapped decoder for the large speech graph. Preserve archived results and figures. Keep data, graphs, and weights out of Git. Use the remaining time on reader-path failures before expanding session coverage or running full training.
- If the current session cannot start WSL, use a machine that can, if available. Otherwise report the WSL stages as blocked, quote the error and affected command, and finish the Windows checks. Leave the candidate changes ready for review.

## Morning report

Return one row per study with these columns. Use `pass`, `blocked`, or `pending` for each stage and include a path to the supporting log or result. The final row for each study should make it clear where a reader can start and what they can run successfully.

| Study | File tree | Download and extract | First HDF5 and source trace | Model and decoder | Metric and trial count | First blocker or evidence path |
| --- | --- | --- | --- | --- | --- | --- |

The shift is complete when each of the eight studies has a recorded outcome and every fix made during the shift has been rerun. A study passes the reader acceptance check when a fresh checkout reaches a new score and the resulting files follow the common layout.
