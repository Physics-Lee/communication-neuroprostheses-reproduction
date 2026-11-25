# Kunz 2025 · inner speech T12, T15, T16

These are the archived mixed-training evaluations. The three arms remain available for supplementary analysis. Earlier Figure 2f OPT weights were selected on the full archived evaluation set.

## Archived verification

Run the [study notebook](../../tutorial.ipynb) on the files described in
[\`data/sources.yaml\`](../../data/sources.yaml). The benchmark lists the input
files for each plotted evaluation arm.

## Full pipeline

Follow the [study tutorial](../../tutorial.ipynb) to prepare the data under
`studies/20250821_kunz/` and activate `bci-repro` at the repository root.
For a separate disk, set `BCI_DATA_ROOT` to the directory containing the
same study directories. The [code guide](../../code/README.md) covers the
stage-specific upstream code and model requirements. Set
`BCI_RUN_ID=yyyymmdd_<experiment>` for a new run and use its dated results
directory for explicit output arguments.

1. Run `python studies/20250821_kunz/code/prepare_data.py`.
2. Run `python studies/20250821_kunz/code/train.py t12 mixed <batches> <new-run-dir>`.
3. Run `python studies/20250821_kunz/code/evaluate.py <run-dir> t12`.

The commands above identify the study entry points. The code guide supplies
the intermediate conversions, decoder choices, and hardware requirements.
