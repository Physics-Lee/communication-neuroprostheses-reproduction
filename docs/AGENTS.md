# AGENTS.md — `docs/` (Repository Governance & Cleanup Audit Documentation)

This directory contains repository-level engineering audits and maintenance documentation.

---

## Contents

| File | Description |
|---|---|
| [`20261124_repository_migration/README.md`](20261124_repository_migration/README.md) | Study and figure migration plan and verification record. |
| [`20261124_repository_migration/naming_map.md`](20261124_repository_migration/naming_map.md) | Public commands, data paths, and result names across the eight studies. |
| [`20261124_repository_migration/reader_workflow.md`](20261124_repository_migration/reader_workflow.md) | Fresh-clone instructions and shared Windows and WSL requirements for all eight study Notebooks. |
| [`20261124_repository_migration/fresh_machine_setup.md`](20261124_repository_migration/fresh_machine_setup.md) | One-page fresh-machine map: all eight studies' downloads and raw paths, the no-WSL metric path, WSL toolchain prerequisites, and measured first-pass timings and disk budget. |
| [`20261124_repository_migration/overnight_reader_acceptance_plan.md`](20261124_repository_migration/overnight_reader_acceptance_plan.md) | Twelve-hour clean-checkout acceptance plan for the eight study Notebooks. |
| [`20261124_repository_migration/overnight_reader_acceptance_report.md`](20261124_repository_migration/overnight_reader_acceptance_report.md) | Results of that acceptance run: per-study pass/blocked rows, fresh metrics vs archived baselines, fixes made, and known residuals. |
| [`20261124_repository_migration/overnight_plan.md`](20261124_repository_migration/overnight_plan.md) | Bilingual 16-hour plan for naming verification, HDF5 pilots, and study onboarding. |
| [`20261124_repository_migration/clean_start_and_remaining_exports_plan.md`](20261124_repository_migration/clean_start_and_remaining_exports_plan.md) | Clean-checkout acceptance for the Fan tutorial and execution plan for the remaining session exports. |
| [`20261124_repository_migration/workspace_cleanup.md`](20261124_repository_migration/workspace_cleanup.md) | Local workspace cleanup and verification plan. |
| [`20261124_repository_migration/workspace_cleanup_review.md`](20261124_repository_migration/workspace_cleanup_review.md) | Relocated inputs, independent Windows verification, and measured space reclaimed from the approved cleanup. |
| [`20261124_repository_migration/pilot_willett2023_session_export.md`](20261124_repository_migration/pilot_willett2023_session_export.md) | First common session file (Willett 2023, day `t12.2022.05.19`): source trace, bit-identical trainer-input features, and the repeated 245/637 greedy evaluation. |
| [`20261124_repository_migration/pilot_wairagkar2025_session_export.md`](20261124_repository_migration/pilot_wairagkar2025_session_export.md) | Common session file for a voice-synthesis study (Wairagkar 2025, day `t15_day195`): neural and acoustic time axes, silence intervals, and the recovered listener-test split. |
| [`20261124_repository_migration/onboarding_a_study.md`](20261124_repository_migration/onboarding_a_study.md) | Six steps for adding a study: register the source, write an exporter, export one session, and validate it with the shared command. |
| [`20261124_repository_migration/session_export_inventory.md`](20261124_repository_migration/session_export_inventory.md) | State of every session file of every study: exported, invalid, pending, or excluded. Written by `shared/tools/session_export_inventory.py --write`. |
| [`CLEANUP_AUDIT.md`](20261108_cleanup_audit/CLEANUP_AUDIT.md) | Per-file audit report from the 2026-11-08 repository cleanup: documents all 147 tracked `.py` files, cites the exact evidence (module docstrings, `diff` output, import graphs) for the 22 superseded/one-shot scripts and 3 duplicate tool files that were removed, and lists the canonical role of every retained file. |

---

## Conventions

1. **Do NOT restore scripts listed as deleted in `CLEANUP_AUDIT.md`**:
   - Files such as `studies/20230831_willett/code/decode_words_btot24.py` (v1 buggy lexicon lookup), `studies/20231201_fan/code/tier3_corp_sim.py` (v1 retracted greedy self-label run), and `studies/20260316_jude/code/patch_table_*.py` (already-applied one-shot markdown patchers) were intentionally removed with full provenance.
   - Whenever an AI agent or contributor wonders why a specific version number is canonical (e.g., `decode_words_btot24_v2.py`, `tier3_corp_sim_v6.py`, `pipeline_watchdog_v4.py`), check [`CLEANUP_AUDIT.md`](20261108_cleanup_audit/CLEANUP_AUDIT.md) first.
2. **Adding new repository-wide technical notes**:
   - Place cross-study audit or design documents here and register a one-line summary in the table above.
