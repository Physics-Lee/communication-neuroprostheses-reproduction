# Jude 2026b · intended speech T17

The offline model used the last eligible sentence block for validation. Figure 3a pools the greedy phoneme and post-LM word errors from that block by recording day.

## Archived verification

Run the [study notebook](../../tutorial.ipynb) on the files described in
[\`data/sources.yaml\`](../../data/sources.yaml). The benchmark lists the input
files for each plotted evaluation arm.

## Full pipeline

Follow the [study tutorial](../../tutorial.ipynb) to prepare the data under
`studies/20260428_jude/` and activate `bci-repro` at the repository root.
For a separate disk, set `BCI_DATA_ROOT` to the directory containing the
same study directories. The [code guide](../../code/README.md) covers the
stage-specific upstream code and model requirements. Set
`BCI_RUN_ID=yyyymmdd_<experiment>` for a new run and use its dated results
directory for explicit output arguments.

1. Run `python studies/20260428_jude/code/prepare_data.py`.
2. Run `python studies/20260428_jude/code/train.py <batches> <new-run-dir>`.
3. Run `python studies/20260428_jude/code/evaluate.py <run-dir>`.

The commands above identify the study entry points. The code guide supplies
the intermediate conversions, decoder choices, and hardware requirements.
