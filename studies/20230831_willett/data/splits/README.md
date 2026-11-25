# Evaluation trials

The [dryad_x69p8czpq__archived_evaluation.csv](dryad_x69p8czpq__archived_evaluation.csv) file records the scored evaluation rows recoverable from the archived outputs. The 1-based archive row points into its source or within its JSON session. Blank original IDs indicate that the exported archive gives the row order only. Verify training, calibration, tuning, and fold assignments against the original release before reusing or extending a split.

# Source partitions

[dryad_x69p8czpq__source_partitions.csv](dryad_x69p8czpq__source_partitions.csv) lists every trial of every exported session file with the partition the publisher assigned (`train`, `test`, `competitionHoldOut`) and the role this reproduction gives that partition. The `test` partition is the validation set of the offline evaluation, and `competitionHoldOut` is unlabeled. Rows are keyed by `trial_id`.

Build the file from the exported session files with `python studies/20230831_willett/code/build_split_manifest.py`. The script also checks that the validation trials of each exported session equal the archived evaluation rows in content and order. Add `--require-all` to fail when an archived session has no session file.
