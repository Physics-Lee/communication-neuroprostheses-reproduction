# AGENTS.md — `docs/` (Repository Governance & Cleanup Audit Documentation)

This directory contains repository-level engineering audits and maintenance documentation.

---

## Contents

| File | Description |
|---|---|
| [`20261124_repository_migration/README.md`](20261124_repository_migration/README.md) | Study and figure migration plan and verification record. |
| [`20261124_repository_migration/naming_map.md`](20261124_repository_migration/naming_map.md) | Public commands, data paths, and result names across the eight studies. |
| [`20261124_repository_migration/overnight_plan.md`](20261124_repository_migration/overnight_plan.md) | Bilingual 16-hour plan for naming verification, HDF5 pilots, and study onboarding. |
| [`CLEANUP_AUDIT.md`](20261108_cleanup_audit/CLEANUP_AUDIT.md) | Per-file audit report from the 2026-11-08 repository cleanup: documents all 147 tracked `.py` files, cites the exact evidence (module docstrings, `diff` output, import graphs) for the 22 superseded/one-shot scripts and 3 duplicate tool files that were removed, and lists the canonical role of every retained file. |

---

## Conventions

1. **Do NOT restore scripts listed as deleted in `CLEANUP_AUDIT.md`**:
   - Files such as `studies/20230831_willett/code/decode_words_btot24.py` (v1 buggy lexicon lookup), `studies/20231201_fan/code/tier3_corp_sim.py` (v1 retracted greedy self-label run), and `studies/20260316_jude/code/patch_table_*.py` (already-applied one-shot markdown patchers) were intentionally removed with full provenance.
   - Whenever an AI agent or contributor wonders why a specific version number is canonical (e.g., `decode_words_btot24_v2.py`, `tier3_corp_sim_v6.py`, `pipeline_watchdog_v4.py`), check [`CLEANUP_AUDIT.md`](20261108_cleanup_audit/CLEANUP_AUDIT.md) first.
2. **Adding new repository-wide technical notes**:
   - Place cross-study audit or design documents here and register a one-line summary in the table above.
