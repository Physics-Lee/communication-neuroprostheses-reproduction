# Fan 2023 · handwriting CORP T5

The final six trials of each session were withheld from CORP adaptation. Figure 3b compares greedy CER to WER from a character 5-gram beam. This replay covers an alternate decoder and has a separate full-pipeline status in Figure 2f.

## Archived verification

Run the [study notebook](../../tutorial.ipynb) on the files described in
[\`data/sources.yaml\`](../../data/sources.yaml). The benchmark lists the input
files for each plotted evaluation arm.

## Full pipeline

Follow the [study tutorial](../../tutorial.ipynb) to prepare the data under
`studies/20231201_fan/` and activate `bci-repro` at the repository root.
For a separate disk, set `BCI_DATA_ROOT` to the directory containing the
same study directories. The [code guide](../../code/README.md) covers the
stage-specific upstream code and model requirements. Set
`BCI_RUN_ID=yyyymmdd_<experiment>` for a new run and use its dated results
directory for explicit output arguments.

1. Run `python studies/20231201_fan/code/evaluate.py --study studies/20231201_fan --output <new-run-dir>`.

The commands above identify the study entry points. The code guide supplies
the intermediate conversions, decoder choices, and hardware requirements.
