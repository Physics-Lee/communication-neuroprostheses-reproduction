# Evaluation trials

The [dryad_wh70rxwmv__archived_evaluation.csv](dryad_wh70rxwmv__archived_evaluation.csv) file records the scored evaluation rows recoverable from the archived outputs. The 1-based archive row points into its source or within its JSON session. Blank original IDs indicate that the exported archive gives the row order only. Verify training, calibration, tuning, and fold assignments against the original release before reusing or extending a split.

# Train and test partitions

[dryad_wh70rxwmv__train_test_partitions.csv](dryad_wh70rxwmv__train_test_partitions.csv) lists the sentences of every exported session that the publisher's two cross-validation partitions assign, with the role `train` or `test` in `HeldOutTrials` and `HeldOutBlocks`. The partition indices are 0-based rows into the per-sentence arrays of the session's `sentences.mat`. Excluded sentences are in neither role. Rows are keyed by `trial_id`.

Build the file with `python studies/20210512_willett/code/build_split_manifest.py`. It covers the exported sessions, so run it again after exporting more.
