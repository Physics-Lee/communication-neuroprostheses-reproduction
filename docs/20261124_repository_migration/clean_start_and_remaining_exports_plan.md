# Clean-start acceptance and remaining session exports

## Goal

Prove that a new reader can create and validate the Fan 2023 example HDF5 using the tutorial and the publisher ZIP. After that passes, finish the pending session exports and update the inventory. Use the `migration/20261124-studies-layout` branch. The latest verified count was **131 exported, 49 pending, and 2 excluded**. Refresh the counts before running.

## 1. Verify the Fan tutorial from a clean checkout

1. Record the branch and commit. Create a separate checkout of the same tracked files under a new directory in `D:\workspace`. A local clone of the migration branch is sufficient. Clear `BCI_DATA_ROOT` and `BCI_ARCHIVE_ROOT` in the verification shell so the scripts read and write within that checkout. Keep the main checkout available as the source of the publisher ZIP.
2. Activate `bci-repro`. Before copying, check free space on the pilot volume. The current Fan ZIP is 3.32 GiB and its members total 13.61 GiB uncompressed. Allow space for both, the checkout, and the HDF5 export. Copy the single ZIP from the main checkout to the exact location in step 1 of `studies/20231201_fan/tutorial.ipynb`. Confirm that the clean checkout has no extracted Fan data or exported Fan HDF5 files before extraction.
3. From the clean checkout root, follow the Notebook's extraction, export, and source validation commands `as written`. The upstream CORP setup is for the separate replay exercise. The HDF5 exporter reads the extracted TFRecords and the MAT entries inside the original ZIP.

   ```powershell
   python -m zipfile -e "studies/20231201_fan/data/raw/dryad_hqbzkh1p6/dryad_hqbzkh1p6__CORP_data_release.zip" "studies/20231201_fan/data/extracted/dryad_hqbzkh1p6"
   python studies/20231201_fan/code/export_sessions.py --session t5.2022.09.29
   python shared/tools/validate_session.py studies/20231201_fan/data/processed/t5/dryad_hqbzkh1p6_2022-09-29.h5 --source trials
   ```

4. Run the Notebook's first Python cell from the clean checkout. It should open the new HDF5 and report 60 trials for `t5.2022.09.29`. Record the commands, exit codes, resulting path, and the validator's `60/60 trials re-read and identical` line.

**Pass condition** The instructions work from a clean directory with the publisher ZIP in place. If a documented command or path fails, fix the relevant instructions or path resolution in the main checkout and rerun the affected steps with clean data before continuing. Record any external dependency that prevents the trial from running.

## 2. Finish pending exports in the main checkout

Switch back to the main checkout and confirm the active data root points to the intended local data tree. Start by collecting the exact pending IDs and paths with the exporters.

```powershell
python studies/20240815_card/code/export_sessions.py --list
python studies/20231201_fan/code/export_sessions.py --list
python studies/20250612_wairagkar/code/export_sessions.py --list
```

| Order | Study | Pending at handoff | First check |
| --- | --- | ---: | --- |
| 1 | Card 2024 | 3 | Begin with `t15.2024.03.03`. The other IDs are `t15.2024.04.25` and `t15.2024.04.28`. Validate the labeled trials and keep counts of any withheld-label trials in the file metadata. |
| 2 | Fan 2023 | 11 | Begin with seed session `t5.2022.05.18`. Verify the source trace for the seed data before proceeding. The clean-start pilot uses a follow-up session. |
| 3 | Wairagkar 2025 | 35 | Begin with `t15_day025`. Read the existing ZIP directly. Use the first completed file's size and the available disk space to plan the remaining batches. Session sizes can vary. |

- Run one export process at a time. Start with `--session <ID>` for the first row of each study, then pass repeated `--session` arguments for manageable batches of pending IDs. Use the output paths from `--list`. Existing session files are preserved by the default command. Keep the 16 GB VRAM and 64 GB RAM ceilings.
- Validate every newly written file with `python shared/tools/validate_session.py <file> --source trials`. Multiple explicitly named files can share one invocation. Save the output or use `--json <path>` for a batch receipt. Every file must pass and each source trace must cover every stored trial.
- Before each batch, check free disk space. If the exporter reports `STOP`, retain completed sessions and record which IDs remain. If a session fails export or validation, record the ID, error, and source, then inspect the cause before resuming that study. Resume from a fresh `--list` and the saved validation receipts so an existing file still gets validated if an earlier run stopped after writing it.
- Keep the original ZIPs, extracted inputs, archived baseline results, and existing split assignments. Store new HDF5 files only under the study `data/processed/<participant>/` directories and keep them out of Git. The two already-excluded Jude 2026b sessions stay excluded.

## 3. Record the outcome

Once the new exports have passed their trial traces, run the full inventory from the main checkout. Its default file-level source check complements the trial-level receipts for the newly exported files.

```powershell
python shared/tools/session_export_inventory.py --write
python shared/tools/check_study_tutorials.py --run
git diff --check
```

- The expected inventory if all 49 pending sessions are completed is **180 exported, 0 pending, 0 invalid, and 2 excluded**. If work stops early, report the actual counts and affected IDs. Regenerate the full inventory with the script, which writes both the JSON and Markdown.
- Add a short verification entry to the [migration README](README.md#migration-and-verification-record) with the tested commit, Fan clean-start result, counts and disk usage by study, and any remaining issue. Keep generated HDF5 files and temporary verification data local. Confirm the Git diff contains the intended documentation, any narrow fixes uncovered by the pilot, and the regenerated inventory.
- Hand back the clean-start command results and the final inventory counts. A successful clean-start shows the archive-to-HDF5 workflow works for the Fan example. The three export batches establish coverage for the remaining sessions in the inventory.
