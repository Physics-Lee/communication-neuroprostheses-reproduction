# Card 2024 · speech T15

Figure 3a uses the validation trials and day-index-corrected decode from the P4 GRU. Figure 2f uses the BiGRU comparison. The rapid-calibration results are a separate archive.

For a word-decoder replay on the authors' exact released Figure 2 trials,
see [Same-trial WER](same_trial_replay.md).
For from-scratch GRU training and the separate mapped neural cohort,
see [Figure 2 neural reproduction](neural_reproduction.md).

## Archived verification

Run the [study notebook](../../tutorial.ipynb) on the files described in
[\`data/sources.yaml\`](../../data/sources.yaml). The benchmark lists the input
files for each plotted evaluation arm.

## Full pipeline

Open the [study Notebook](../../tutorial.ipynb) and follow **Prepare the data** and **Reproduce and score**. The [common reader guide](../../../../docs/20261124_repository_migration/reader_workflow.md) covers the environment, run directories, and WSL setup. See the [code guide](../../code/README.md) for the model and experiment variants.
