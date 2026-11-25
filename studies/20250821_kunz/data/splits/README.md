# Evaluation trials

The [dryad_gf1vhhn1j__archived_evaluation.csv](dryad_gf1vhhn1j__archived_evaluation.csv) file records the scored evaluation rows recoverable from the archived outputs. The 1-based archive row points into its source or within its JSON session. Blank original IDs indicate that the exported archive gives the row order only. Verify training, calibration, tuning, and fold assignments against the original release before reusing or extending a split.

# Conditions

[dryad_gf1vhhn1j__conditions.csv](dryad_gf1vhhn1j__conditions.csv) lists every trial of every exported session with the arms of the pipeline that use its file. A trial of an `_eval` file has the role `evaluation`, which every arm validates on. Any other trial has `training_arm`, and `arms` names the arms that select the file. A trial that the converter's length or finiteness filter drops has `dropped` and the reason in `drop_reason`. Rows are keyed by `trial_id`.

# T16 folds

[dryad_gf1vhhn1j__t16_folds.csv](dryad_gf1vhhn1j__t16_folds.csv) lists the trials of the session the T16 block holdout reads, once per fold. `fold` is the fold whose held-out block holds the trial. In fold f, the rows with `fold` equal to f are validation and every other row is training.

Build both files with `python studies/20250821_kunz/code/build_split_manifest.py`.
