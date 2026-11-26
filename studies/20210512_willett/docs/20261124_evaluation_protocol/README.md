# Willett 2021 · handwriting T5

Use the released HeldOutTrials indices. Figure 3b pairs greedy CER extracted from the PyTorch inference MAT files with bigram word errors on those same trials. The per-trial character errors in the decode CSV describe its bigram output.

## Archived verification

Run the [study notebook](../../tutorial.ipynb) on the files described in
[\`data/sources.yaml\`](../../data/sources.yaml). The benchmark lists the input
files for each plotted evaluation arm.

## Full pipeline

Open the [study Notebook](../../tutorial.ipynb) and follow **Prepare the data** and **Reproduce and score**. The [common reader guide](../../../../docs/20261124_repository_migration/reader_workflow.md) covers the environment, run directories, and WSL setup. See the [code guide](../../code/README.md) for the model and experiment variants.
