# Fig. 2 Reconciliation Workfiles (2026-11-26)

This directory archives the cross-repository reconciliation of **Table 2 ("Released data
and decoding performance") and Figure 2** between this reproduction repository and the
paper manuscript. It was migrated here from `tmp/` on 2026-11-28; subdirectory names
keep their original `fig2_*` prefixes so paths quoted in older commits and documents
remain greppable.

## Background

The paper manuscript does **not** live in this repository. Its LaTeX source and the
`SourceData_Figure2.xlsx` workbook live in the separate paper repository
(`D:/repositories/20261012_Comment_on_Recent_Progress_of_Language_BCI/latex/`, hardcoded
in each `update_source_data.mjs`). These workfiles bridge the two sides: rows computed
from this repository's archived per-trial decode outputs, the paper-snapshot LaTeX used
to check table edits, and before/after screenshots evidencing each workbook write-back.

## The three candidate protocols

The Table 2 "Reproduced WER" column was computed under three evaluation protocols; each
subdirectory is a self-contained candidate (same 1,560-line paper snapshot, only the
reproduced-WER column and its caption wording differ):

| Subdirectory | Protocol | Example (Willett 2021, T5) |
|---|---|---|
| [`fig2_matched_dates/`](fig2_matched_dates/) | Reproduced WER averaged over the same evaluation dates as the paper's online set | 6.57 ± 3.55 |
| [`fig2_offline_wer/`](fig2_offline_wer/) | Reproduced WER over all offline dates in the release | 7.09 ± 2.82 |
| [`fig2_online_wer/`](fig2_online_wer/) | Reproduced WER computed directly on the online trial set | 10.0 ± 3.58 |

**Outcome: `fig2_matched_dates` is the adopted protocol.** The HTML Table 2 rendered by
`figures/code/render_figure2.py` (caption: "Online and reproduced WERs use the same
evaluation dates") carries its numbers; the other two directories are retained as the
runners-up record.

## Per-directory contents (typical)

- `main.tex` — full snapshot of the paper manuscript at reconciliation time (the table
  lives near line 804, `\label{tab:repro}`).
- `resolved.json` / `offline_rows.json` / `data_fig2_reproduction.json` — row data
  computed from `studies/*/results/` archived decode outputs, traceable to source.
- `SourceData_Figure2.xlsx` — local copy of the paper's source-data workbook;
  `*.inspect.ndjson` is a per-cell dump used for verification.
- `original_hashes.json` / `original_workbook_values.json` — pre-write baseline snapshots
  proving the write-back scripts touched only the intended cells.
- `update_source_data.mjs` — Node script (`@oai/artifact-tool`) writing the new numbers
  into the paper repository's workbook, rendering `source_before/after.png` receipts.
- `before_main_fig_2.png`, `source_before/after.png` — before/after visual evidence.
- `figure2_final_page.png` (top level) — reference render of the paper's final Figure 2
  page.

## Provenance

- 2026-11-26 `c31a676` / `d3b490b` — workfiles committed under `tmp/` (an accidental
  `node_modules/` intake was reverted from the index in `142333d`).
- 2026-11-27 `f1b2a72` — Table 2 "Released blocks" column added; the three `main.tex`
  snapshots were kept cell-for-cell in sync with the HTML table.
- 2026-11-28 — migrated to `docs/20261126_fig2_reconciliation/` (this directory).
