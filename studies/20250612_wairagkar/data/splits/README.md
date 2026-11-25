# Evaluation trials

The voice-synthesis archive records per-day acoustic summaries with trial counts. Trial-level membership is not recorded in those summaries. The two lists below recover it from the released inputs.

# Evaluation trials

[dryad_2280gb64f__evaluation_trials.csv](dryad_2280gb64f__evaluation_trials.csv) lists all 8,489 trials of the 37 recording days with their role. The release marks 1,216 trials as evaluation trials and names the decoder that ran each of the nine evaluation days. Rows are keyed by `trial_id`.

# Listener benchmark

[dryad_2280gb64f__listener_benchmark.csv](dryad_2280gb64f__listener_benchmark.csv) maps the 90 `random_test_idx` and 38 `video_test_idx` values of the release's `test_idx.mat` to trials. The release does not number them. Read as zero-based positions in the concatenation of all days in day order, every value falls on an evaluation trial. Read as one-based, one falls on a training trial. One trial is in both sets, so the file holds 127 distinct trials.

Build both files with `python studies/20250612_wairagkar/code/build_split_manifest.py`. The script repeats the index check on every run and stops if the zero-based reading no longer holds. The evidence is in the [pilot report](../../../docs/20261124_repository_migration/pilot_wairagkar2025_session_export.md).
