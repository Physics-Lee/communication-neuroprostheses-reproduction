# Evaluation trials

The [dryad_vq83bk481__archived_evaluation.csv](dryad_vq83bk481__archived_evaluation.csv) file records the scored evaluation rows recoverable from the archived outputs. The 1-based archive row points into its source or within its JSON session. Blank original IDs indicate that the exported archive gives the row order only. Verify training, calibration, tuning, and fold assignments against the original release before reusing or extending a split.

# Validation blocks

[dryad_vq83bk481__validation_blocks.csv](dryad_vq83bk481__validation_blocks.csv) lists every exported trial with the role `training` or `validation`. The sentence block with the highest number of each day is the validation split, as in `convert_jude.py`. `vocab` and `loop` repeat the dataset README table. Rows are keyed by `trial_id`.

Build the file with `python studies/20260428_jude/code/build_split_manifest.py`. The script also checks that the validation sentences of each exported session equal the archived evaluation rows in content and order. Add `--require-all` to fail when an archived session has no session file.
