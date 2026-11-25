# Evaluation trials

The [dryad_cz8w9gjjk__archived_evaluation.csv](dryad_cz8w9gjjk__archived_evaluation.csv) file records the scored evaluation rows recoverable from the archived outputs. The 1-based archive row points into its source or within its JSON session. Blank original IDs indicate that the exported archive gives the row order only. Verify training, calibration, tuning, and fold assignments against the original release before reusing or extending a split.

# Test trials

[dryad_cz8w9gjjk__test_trials.csv](dryad_cz8w9gjjk__test_trials.csv) lists every exported trial with the role the official conversion gives it. The cues the conversion keeps are shuffled with seed 9: the last `num_test_trials` of them are `test` and the rest are `train`. A cue it leaves out has the role `excluded` and the reason in `exclusion_reason`. Rows are keyed by `trial_id`.

Build the file with `python studies/20260316_jude/code/build_split_manifest.py`. It writes nothing unless the exported trials agree with the TFRecords of each existing conversion.
