# Jude 2026a · touch typing T17 and T18

The released test trial split was used for checkpoint validation. Figure 3b applies a minimum of three trials per session. Raw greedy CER is available at session level. The per-trial character and word counts below describe the post-LM output.

## Archived verification

Run the [study notebook](../../tutorial.ipynb) on the files described in
[\`data/sources.yaml\`](../../data/sources.yaml). The benchmark lists the input
files for each plotted evaluation arm.

## Full pipeline

Follow the [study tutorial](../../tutorial.ipynb) to prepare the data under
`studies/20260316_jude/` and activate `bci-repro` at the repository root.
For a separate disk, set `BCI_DATA_ROOT` to the directory containing the
same study directories. The [code guide](../../code/README.md) covers the
stage-specific upstream code and model requirements. Set
`BCI_RUN_ID=yyyymmdd_<experiment>` for a new run and use its dated results
directory for explicit output arguments.

1. Run `python studies/20260316_jude/code/prepare_data.py <session> <block-csv> <test-trial-count>`.
2. Run `python studies/20260316_jude/code/train.py --participant t17 --data-root <converted-data> --out-root <new-run-dir>`.

The commands above identify the study entry points. The code guide supplies
the intermediate conversions, decoder choices, and hardware requirements.
