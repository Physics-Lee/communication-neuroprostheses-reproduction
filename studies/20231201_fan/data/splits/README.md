# Evaluation trials

The [dryad_hqbzkh1p6__archived_evaluation.csv](dryad_hqbzkh1p6__archived_evaluation.csv) file records the scored evaluation rows recoverable from the archived outputs. The 1-based archive row points into its source or within its JSON session. Blank original IDs indicate that the exported archive gives the row order only. Verify training, calibration, tuning, and fold assignments against the original release before reusing or extending a split.

# Paired v7 holdout

[dryad_hqbzkh1p6__paired_v7_holdout.csv](dryad_hqbzkh1p6__paired_v7_holdout.csv) lists every trial of every exported session with its role in the tier-3 replay. The replay reads the `recalibration` arm: its final six trials are the `holdout` and its other trials are `adaptation`. Trials of the `no_recalibration` arm are `frozen`, and the seed partitions are `seed_training` and `seed_validation`. `order` is the zero-based position of the trial in its arm. Rows are keyed by `trial_id`.

Build the file with `python studies/20231201_fan/code/build_split_manifest.py`. The script also checks every archived evaluation row of an exported session against a holdout row. Add `--require-all` to fail when an archived session has no session file.
