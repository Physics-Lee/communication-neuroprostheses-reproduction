# Willett 2021 · handwriting T5

Use the released HeldOutTrials indices. Figure 3b pairs greedy CER extracted from the PyTorch inference MAT files with bigram word errors on those same trials. The per-trial character errors in the decode CSV describe its bigram output.

## Archived verification

Run the [study notebook](../../tutorial.ipynb) on the files described in
[\`data/sources.yaml\`](../../data/sources.yaml). The benchmark lists the input
files for each plotted evaluation arm.

## Full pipeline

Follow the [study tutorial](../../tutorial.ipynb) to prepare the data under
`studies/20210512_willett/` and activate `bci-repro` at the repository root.
For a separate disk, set `BCI_DATA_ROOT` to the directory containing the
same study directories. The [code guide](../../code/README.md) covers the
stage-specific upstream code and model requirements. Set
`BCI_RUN_ID=yyyymmdd_<experiment>` for a new run and use its dated results
directory for explicit output arguments.

1. Run `python studies/20210512_willett/code/evaluate.py`.
2. Run `python studies/20210512_willett/code/train.py --partition HeldOutTrials --synth --outputDir <new-run-dir>`.

The commands above identify the study entry points. The code guide supplies
the intermediate conversions, decoder choices, and hardware requirements.
