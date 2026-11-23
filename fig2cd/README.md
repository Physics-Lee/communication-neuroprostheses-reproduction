# Figure 2(c,d) online WER and communication speed

Open the [English notebook](explore_figure2cd.ipynb) to reproduce the panel values and inspect the daily counts and plots for a participant. It runs on the committed CSV.

[Daily counts and timings](daily_metrics.csv) are the inputs used to draw the online word error rate (WER) and speed bars. The records come from the authors’ released, real-time decoder outputs and trial timing. Rows follow paper publication order, then participant and calendar recording date. The task, vocabulary, and available days vary by study.

## How the bars are calculated

- **WER, panel (c).** Sum the word edit counts for the day, divide by the total reference words, and multiply by 100. Word edits include insertions, deletions, and substitutions. Text is case-folded and split on whitespace after translating release-specific symbols, then punctuation is stripped within tokens except for apostrophes.
- **Speed, panel (d).** Divide the total reference count by total trial minutes for that day. Speech uses reference words. Handwriting and typing use five reference characters, including spaces, per word equivalent. This is task throughput, so it is independent of whether a word was decoded correctly. Unless noted below, the duration is from the go cue to the recorded trial end.
- **Across days.** The bar is the arithmetic mean of the daily rates. The whisker is one sample standard deviation across those days, with a denominator of *n* − 1. Single-day results have a bar and no whisker. Missing rates are left blank.
- **Voice synthesis.** Wairagkar 2025 reports a **43.75% listener transcription median** from open transcription of 90 sampled synthesized-speech trials. The panel marks that value and shows it without a whisker. The Dryad file inventory lists neural data, speech timing, target speech features, and the models used online. The paper’s supplementary videos contain examples of the synthesized audio.

The CSV contains the daily numerators and denominators. `released_trials` counts trials in the selected released blocks or conditions. `wer_trials` counts trials with a usable reference and a recoverable output record. `speed_trials` counts scored trials with a positive, finite duration. The `speed_unit` field specifies which count goes into the WPM conversion. The final two columns, `wer_pct` and `speed_wpm`, contain rates calculated from the counts and rounded to six decimal places. Missing rates are blank. The figure computes its summaries directly from the counts at full precision.

## Data selection

| Study | Online trial selection and timing | Public source |
|---|---|---|
| Willett 2021, T5 | Five published copy-typing days. Use `CL Corpus` and `CL Pandarinath`, drop the release’s excluded trials, and compare `rnn_decodedText` with `intendedText`. Timing uses the go cue and sentence end in 10 ms bins. | [Dryad](https://doi.org/10.5061/dryad.wh70rxwmv) |
| Willett 2023, T12 | Five vocal evaluation days. Use `CL Switchboard Stage 2` blocks, `ngramFinalOutput`, and `sentenceDurations` in 20 ms bins. | [Dryad](https://doi.org/10.5061/dryad.x69p8czpq) |
| Fan 2023, T5 | Ten released recalibration evaluation sessions. Compare `lm_decoded` with `sentences`. Each `tx_feats` row is one 20 ms bin in the trial’s go period. | [Dryad](https://doi.org/10.5061/dryad.hqbzkh1p6) |
| Card 2024, T15 | The 14 evaluation days in Supplementary Table S3. Use the 125,000-word trials in `t15_copyTask.pkl` with their cue, decoded sentence, and recorded speech duration. | [Dryad](https://doi.org/10.5061/dryad.dncjsxm85) |
| Wairagkar 2025, T15 | Paper Fig. 2l human open-transcription median. Seven listeners transcribed each of the 90 sampled trials. The paper grouped trials into sentence pairs when scoring. | [Paper](https://doi.org/10.1038/s41586-025-09127-3) · [Dryad](https://doi.org/10.5061/dryad.2280gb64f) |
| Kunz 2025, T12/T15/T16 | Closed-loop imagined speech in the released `*_eval.mat` files. T12 uses one day with the 50-word vocabulary. T15 and T16 use two days and one day, respectively, with the large vocabulary. The figure shows no speed for these conditions. | [Dryad](https://doi.org/10.5061/dryad.gf1vhhn1j) |
| Jude 2026a, T17/T18 | Use the blocks named in the paper’s Fig. 3b,c speed sheets. Score the released trials in those blocks and combine copy-typing and personal-use trials on the same date. Three T17 blocks have different paper and release counts, listed below. | [Dryad](https://doi.org/10.5061/dryad.cz8w9gjjk) · [Paper Source Data](https://www.nature.com/articles/s41593-026-02218-y#Sec17) |
| Jude 2026b, T17 | Use the same unconstrained-vocabulary blocks for WER and speed on trial days 33, 68, 75, and 89, matching the cohort used for the earlier WER bar. Use all scored trials in those blocks and the recorded go-cue-to-end time for speed. | [Dryad](https://doi.org/10.5061/dryad.vq83bk481) |

For both Jude studies, the extraction uses a final decoder message assigned by the Redis clock. When multiple finals are recorded between a trial’s go cue and the next go cue, it takes the final closest to trial end. If a trial has no final, it uses the last recorded partial within that trial. An explicit empty per-trial output contributes deletion errors. A tick-based trial with no recoverable output is left unscored. This excludes four of the 165 released T17 typing trials in the selected blocks. The source-data workbook identifies the Jude 2026a blocks and provides the published comparison.

The T17 block counts in the source sheet and in the Dryad files differ as follows.

| T17 free-typing block | Paper speed sheet | Released trials |
|---|---:|---:|
| 2024-06-04, block 9 | 7 | 5 |
| 2024-07-09, block 4 | 8 | 2 |
| 2024-07-10, block 3 | 7 | 8 |

## Reproduce the numbers

From the repository root, use the committed daily table to verify the inputs and print the panel summaries. This step requires only Python.

```bash
python fig2cd/recompute.py
python render_figure2.py
```

To regenerate the table from the public releases, install `numpy`, `scipy`, `editdistance`, and `openpyxl`. Create a source manifest using the keys below. Paths can be absolute or relative to the manifest. Use the original Dryad `t15_copyTask.pkl` for the Card input.

```json
{
  "willett21_datasets": "handwritingBCIData/Datasets",
  "willett23_sentences_archive": "sentences.tar.gz",
  "fan23_archive": "CORP_data_release.zip",
  "card24_copy_task": "t15_copyTask.pkl",
  "kunz25_sentence_data": "sentenceData",
  "jude26a_data": "jude-typing/Data",
  "jude26a_figure3_source_data": "SourceData_Figure3.xlsx",
  "jude26b_data": "jude-speech/Data"
}
```

```bash
python fig2cd/recompute.py --sources sources.json
python render_figure2.py
```

The rebuild reads one recording at a time. For Willett 2023 it streams the archive and stages one selected MAT file at a time. The renderer derives the plotted values and JSON mirror from the same CSV, while the benchmark YAML supplies the row order, labels, and Wairagkar’s published exception.

## Changes from the earlier figure

- **Kunz WER.** The earlier per-day calculation averaged the individual sentence WERs. The current calculation weights by the number of reference words. The T12 and T16 results each have one day, so their earlier `±0.0` labels have been removed.
- **Jude 2026a WER.** The earlier figure averaged per-sentence percentages downloaded from the paper. The current figure starts from the recorded online text and uses daily word counts. The speed bars also use the sum of released trial durations within each day.
- **Jude 2026b speed.** The earlier script read only the first trial in each block and included small-vocabulary blocks. The current speed uses the same four large-vocabulary days and blocks as WER, with all eligible timed trials and actual word counts.
- **Other source changes.** Willett 2023 uses the released trial data in place of digitized figure markers. Fan speed uses the ten released recalibration days, matching its WER cohort, in place of the paper’s fifteen-day summary converted from characters per minute. Willett 2021 speed uses the released trial window, and both Card metrics use released trials for the same fourteen Table S3 days.

| Study | Earlier WER (%) | Revised WER (%) | Earlier speed (WPM) | Revised speed (WPM) |
|---|---:|---:|---:|---:|
| Willett 2021, T5 | 25.1 ± 8.6 | 23.9 ± 8.4 | 15.4 ± 2.2 | 14.3 ± 1.8 |
| Willett 2023, T12 | 23.9 ± 1.8 | 23.8 ± 1.8 | 65.3 ± 0.9 | 65.9 ± 0.9 |
| Fan 2023, T5 | 4.0 ± 1.2 | 4.0 ± 1.2 | 14.6 ± 2.2 | 14.4 ± 1.6 |
| Card 2024, T15 | 4.9 ± 3.3 | 4.9 ± 3.3 | 32.5 ± 2.7 | 32.6 ± 2.6 |
| Wairagkar 2025, T15 | 43.8 (median) | 43.8 (median) | — | — |
| Kunz 2025, T12 | 23.7 ± 0.0 | 22.9 | — | — |
| Kunz 2025, T15 | 51.3 ± 33.1 | 43.3 ± 23.8 | — | — |
| Kunz 2025, T16 | 46.6 ± 0.0 | 44.4 | — | — |
| Jude 2026a, T17 | 12.5 ± 2.5 | 10.5 ± 3.1 | 6.8 ± 1.8 | 5.7 ± 1.7 |
| Jude 2026a, T18 | 3.4 ± 0.8 | 3.5 ± 0.6 | 14.0 ± 4.5 | 13.0 ± 4.2 |
| Jude 2026b, T17 | 66.5 ± 18.1 | 66.5 ± 18.1 | 34.1 ± 7.7 | 31.9 ± 1.4 |

The Willett 2021 WER also changes because the common word normalization strips punctuation. The revised Jude 2026b rate is based on **469 trials over four days**. An eight-day, mixed-vocabulary recalculation in the earlier audit answers a different question from the four-day, large-vocabulary bar shown here.

## Paired online and offline WER

[Paired daily counts](paired_wer_daily.csv) and [the included trials](paired_wer_trials.csv) compare the authors’ real-time output with archived reproduction output on their shared trials. Each pair uses a single reference and the same word normalization as panel (c). For each day, divide the summed edit counts for each output by the summed reference word count for the shared trials. Across days, take the arithmetic mean and the sample standard deviation. The trial table records both source identifiers and the two edit counts. Numeric source indices are zero-based. TSV identifiers follow the archived `key` column.

| Study | Reproduction output and trial linkage |
|---|---|
| Willett 2021 | The [held-out block](../20210512_willett/results/m3_own_decode_HeldOutBlocks.csv) and [held-out trial](../20210512_willett/results/m3_own_decode_HeldOutTrials.csv) archives identify each trial by session and `sidx`. The archived reference was checked against the released prompt. Both outputs were scored against the intended text used in panel (c). The held-out block archive covers 158 of the 163 panel-(c) trials and is the primary comparison. The held-out trial archive covers five and is listed as a separate comparison. |
| Willett 2023 | Join [test-trial metadata](../20230831_willett/results/decode_outputs/test_partition_greedy_per_trial.csv) and [archived 5-gram + OPT output](../20230831_willett/results/decode_outputs/wl_5g_opt_official.tsv) in their verified reference order. Match to the online data by session, block, and a reference unique on both sides. |
| Fan 2023 | The archived adaptation results provide [session-level CER](../20231201_fan/results/tier3_corp_sim_v6.json). The [archived WER table](../20231201_fan/results/rescore_official_outputs.json) scores the authors’ online output. A paired WER requires our own per-trial decoded text. |
| Card 2024 | Use the [day-routing-corrected 5-gram + OPT output](../20240815_card/results/decode_outputs/p4_fixed_opt_fx.tsv) with the [validation metadata](../20240815_card/results/decode_outputs/val_greedy_per_trial.csv). Match by date and a reference unique in both the day’s released copy-task records and the offline archive. The resulting pairs follow the same within-day block and trial order. The online ID uses the row index in the copy-task pickle. |
| Kunz 2025 | Use `off_final_full.tsv` from each participant’s [mixed-training archive](../20250821_kunz/results/decode_outputs). Verify that the entire ordered reference list for each selected evaluation file matches the release. These three comparisons cover every corresponding trial in panel (c). |
| Jude 2026a | Use the [multisession decoder archive](../20260316_jude/results/runs_multisess/lm_decode_summary.json). Reconstruct the test indices from the recorded TFRecord conversion parameters and the converter’s fixed shuffle. Verify the full ordered test reference list against the archive. Identify each day’s panel-(c) blocks by the unique subset that reproduces the saved trial, word, edit, character, and timing totals. |
| Jude 2026b | Join [validation metadata](../20260428_jude/results/decode_outputs/val_greedy_per_trial.csv) and [archived 5-gram + OPT output](../20260428_jude/results/decode_outputs/off_5g_opt_official.tsv) in their verified reference order. Match the complete ordered reference list of each held-out block against its released online output. |

The paired online rates use the archived coverage shown in the CSV. Panel (c) uses the full set of available online trials for its selected conditions and days.

### Kunz OPT rescoring archive

The research archive’s `vac/t12_opt_fx.tsv`, `vac/t15_opt_fx.tsv`, and `vac/t16_opt_fx.tsv` reproduce the earlier panel-(f) values. The run log `vac_t5c_progress.log` records the chosen OPT rescoring weights (0.50, 0.55, and 0.50) and pooled error counts. Each weight was selected by WER on its full archived evaluation set. The `kunz25_*_opt_fx` rows use these transcripts on the paired trials. The three unsuffixed rows use `off_final_full.tsv` from the mixed-training archives.

| Participant | Earlier pooled WER, full archive | Paired trials | Paired OPT WER |
|---|---:|---:|---:|
| T12 | 384 / 407 words = 94.35% (100 trials) | 100 | 94.35% |
| T15 | 420 / 538 words = 78.07% (105 trials) | 55 | 91.65 ± 1.71% |
| T16 | 332 / 403 words = 82.38% (85 trials) | 35 | 105.61% |

The earlier T15 and T16 pooled totals each include 50 trials from an additional 50-word session. The paired results use the large-vocabulary blocks selected for panel (c). A second T15 rescoring file, `t15a0325_opt_fx.tsv`, ties at 420 / 538 words on the full archive. The comparison here uses the original report’s 0.55-weight configuration saved in `t15_opt_fx.tsv`.
