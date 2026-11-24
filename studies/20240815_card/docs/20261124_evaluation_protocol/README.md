# Card 2024 · speech T15

Figure 3a uses the validation trials and day-index-corrected decode from the P4 GRU. Figure 2f uses the BiGRU comparison. The rapid-calibration results are a separate archive.

## Archived verification

Run the [study notebook](../../tutorial.ipynb) on the files described in
[\`data/sources.yaml\`](../../data/sources.yaml). The benchmark lists the input
files for each plotted evaluation arm.

## Full pipeline

Follow the [study tutorial](../../tutorial.ipynb) to prepare the data under
`studies/20240815_card/` and activate `bci-repro` at the repository root.
For a separate disk, set `BCI_DATA_ROOT` to the directory containing the
same study directories. The [code guide](../../code/README.md) covers the
stage-specific upstream code and model requirements. Set
`BCI_RUN_ID=yyyymmdd_<experiment>` for a new run and use its dated results
directory for explicit output arguments.

1. Run `python studies/20240815_card/code/train.py <batches> <new-run-dir>`.
2. Run `python studies/20240815_card/code/infer.py <run-dir>`.

The commands above identify the study entry points. The code guide supplies
the intermediate conversions, decoder choices, and hardware requirements.
