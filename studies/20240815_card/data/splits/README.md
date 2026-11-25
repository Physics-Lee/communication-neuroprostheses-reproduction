# Evaluation trials

The [dryad_dncjsxm85__archived_evaluation.csv](dryad_dncjsxm85__archived_evaluation.csv) file records the scored evaluation rows recoverable from the archived outputs. The 1-based archive row points into its source or within its JSON session. Blank original IDs indicate that the exported archive gives the row order only. Verify training, calibration, tuning, and fold assignments against the original release before reusing or extending a split.

# Calibration order

[dryad_dncjsxm85__calibration_order.csv](dryad_dncjsxm85__calibration_order.csv) lists every trial of every exported session with its role in the rapid-calibration stage. On a target day, the `train` trials are `calibration` trials and `order` is their zero-based position in the file, so a budget of N sentences takes the rows with `order` below N. The `val` trials of a target day are `validation`. Every other trial is `other`. Rows are keyed by `trial_id`.

Build the file with `python studies/20240815_card/code/build_split_manifest.py`.
