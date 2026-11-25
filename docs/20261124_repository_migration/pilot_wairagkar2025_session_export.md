# Pilot report: Wairagkar 2025 session export

2026-11-24

The common session format was applied to a study whose output is continuous
speech, to check that the format carries time-aligned acoustic targets and
silence timing as well as text targets.

## What was exported

| Item | Value |
| --- | --- |
| Session | `t15_day195`, one of the two days in the archived evaluation |
| Session file | `studies/20250612_wairagkar/data/processed/t15/dryad_2280gb64f_day195.h5` (718.6 MiB) |
| Trials | 305: 130 `Evaluation trial`, 7 `Evaluation trial (free response)`, 10 `Evaluation trial (mimed speech)`, 146 `Training data (50-word vocab)`, 12 `Training data` |
| Decoder of the day | `brain2voice_t15_day193.h5` (root attribute `brain2voice_decoder`) |
| Neural features | `spike_pow`, `tx`, each `(frames, 256)` as released, 10 ms bins |
| Acoustic targets | `lpcnet_word`, `lpcnet_syllable`, each `(frames, 20)` LPCNet features |
| Timing | `word` (1,697 intervals) and `syllable` (1,954 intervals), each `(n, 2)` onset and offset frames |
| Provenance | every trial names the publisher ZIP and the three members it was read from |

The exporter is `studies/20250612_wairagkar/code/export_sessions.py`. It reads
the publisher's ZIP directly; the study's pipeline never needs an extracted
copy.

## Commands

Run from the repository root with `bci-repro` active.

```text
python studies/20250612_wairagkar/code/export_sessions.py --session t15_day195
python shared/tools/validate_session.py studies/20250612_wairagkar/data/processed/t15/dryad_2280gb64f_day195.h5 --source trials
python studies/20250612_wairagkar/code/check_session_timing.py studies/20250612_wairagkar/data/processed/t15/dryad_2280gb64f_day195.h5
python studies/20250612_wairagkar/code/build_split_manifest.py
```

The timing check reads the exported HDF5 file. The verified result and its
run record are in
`studies/20250612_wairagkar/results/20261125_dryad_2280gb64f__hdf5__none__session_timing_check/`.

## Results

| Check | Result |
| --- | --- |
| Trial count | 305 trials, equal to the number of per-trial arrays in the three members of the day. |
| Source trace | 305 of 305 trials were re-read from the ZIP and are identical to the stored data. |
| Neural time axis | In 305 of 305 trials, `spike_pow` and `tx` have the trial's number of 10 ms frames. |
| Acoustic time axis | In 305 of 305 trials, `lpcnet_word` and `lpcnet_syllable` have the same number of frames as the neural features. |
| Frame timing | In 305 of 305 trials, every word and syllable interval lies inside the trial and ends after it starts. |
| Silence intervals | Over 407,466 frames, the frames outside every word interval and the silent frames of the acoustic targets agree on 99.57 % of frames. 60 silent target frames lie inside a word interval and 1,680 voiced target frames lie outside the word intervals. |

A target frame counts as silent when its first LPCNet coefficient, the log
energy, sits on the lowest value of the session.

## Finding: the word intervals are zero-based

The release does not state the base of `word_indices` and `syl_indices`. The
data settle the onset. In 1,694 of the 1,697 word intervals the frame before
the onset is silent, and the frame at the onset is voiced in 1,640. The
onsets are therefore zero-based. Reading them as one-based places 1,754
silent target frames inside word intervals, against 60 when they are read as
zero-based.

The offset does not mark the last voiced frame. The frame at the offset is
voiced in all 1,697 intervals, and the frame after it still is in 1,666. The
check counts both bounds as inside the word, which is the closer of the two
simple readings.

The two existing evaluation scripts read the tables differently.
`tier1_official_model_inference.py` keeps frames from `onset - 1` to `offset`
inclusive, which adds one silent frame to most intervals.
`tier2_mel_full_trial.py` keeps frames from `onset` to `offset` exclusive,
which drops the voiced offset frame. The difference touches about 1 % of the
voiced frames. Its effect on the archived correlations was not measured. The
session file stores the tables as released and leaves the choice to the
reader.

## Finding: the listener-test indices are global and zero-based

`test_idx.mat` holds 90 `random_test_idx` and 38 `video_test_idx` values
between 4,267 and 8,485. The release readme says they select evaluation trials
but not how they are numbered. The release has 8,489 trials in all 37 days
together. Reading each value as a zero-based position in that concatenation,
in day order, puts all 128 on evaluation trials. Reading them as one-based
puts one on a training trial.

The 23 values that fall on day 195 were checked against the exported session
file: each one resolves to a trial with the same sentence and the same trial
description. This closes the open item in the study's split notes, which said
trial-level membership of the listener test could not be recovered from the
archived summaries.

## Split manifests

| File | Rows | Content |
| --- | --- | --- |
| `data/splits/dryad_2280gb64f__evaluation_trials.csv` | 8,489 | Every trial of every day with its role (1,216 evaluation, 7,273 training), the recorded trial description, and the decoder of the day |
| `data/splits/dryad_2280gb64f__listener_benchmark.csv` | 128 | The listener-test trials by trial id (127 distinct; one trial is in both sets) |

## Limits of this pilot

- Only day 195 was exported for this report. The manifests cover all days
  because they are read from the ZIP, not from the session files. The state of
  the other days is in the [export inventory](session_export_inventory.md).
- The closed-loop audio released for 147 trials of the day is not stored in
  the session file. Each trial records only its sample count.
- Neither tier of the synthesis evaluation was repeated from the session file.
