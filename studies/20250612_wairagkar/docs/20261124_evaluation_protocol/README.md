# Wairagkar 2025 · voice synthesis T15

The acoustic output is continuous speech. The reproduction archives per-day mean Mel-band correlations. The paper's listener word-transcription median is a separate Figure 2c value.

## Archived verification

Run the [study notebook](../../tutorial.ipynb) on the files described in
[\`data/sources.yaml\`](../../data/sources.yaml). The benchmark lists the input
files for each plotted evaluation arm.

## Full pipeline

Follow the [study tutorial](../../tutorial.ipynb) to prepare the data under
`studies/20250612_wairagkar/` and activate `bci-repro` at the repository root.
For a separate disk, set `BCI_DATA_ROOT` to the directory containing the
same study directories. The [code guide](../../code/README.md) covers the
stage-specific upstream code and model requirements. Set
`BCI_RUN_ID=yyyymmdd_<experiment>` for a new run and use its dated results
directory for explicit output arguments.

1. Run `python studies/20250612_wairagkar/code/infer.py`.
2. Run `python studies/20250612_wairagkar/code/evaluate.py`.

The commands above identify the study entry points. The code guide supplies
the intermediate conversions, decoder choices, and hardware requirements.
