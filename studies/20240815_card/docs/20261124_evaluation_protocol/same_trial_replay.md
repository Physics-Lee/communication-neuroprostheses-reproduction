# Card 2024 same-trial WER

The replay compares our 1-best 5-gram WFST to the authors' online output
using the authors' own acoustic logits for **all 1,628 released 125k-word
Figure 2 copy-task trials** on the 14 Table S3 dates. This isolates the word
decoder; it does not reproduce acoustic training or online adaptation.
The online comparator is recalculated on the released cohort, not taken
from the paper's 2.5% final-five-session headline.

| Output | Equal-day WER (mean ± sample SD) | Pooled WER |
|---|---:|---:|
| Authors' released online | 4.87% ± 3.29 pp | 4.50% (465 / 10,337) |
| Our 5-gram WFST replay | 5.69% ± 1.70 pp | 5.67% (586 / 10,337) |

The replay was 0.82 percentage points higher on the equal-day mean
and 1.17 points higher pooled (121 additional edits). Both arms cover the
same 1,628 pickle rows and references, with no missing, extra, or duplicate
predictions.

[Daily counts](../../results/20261126_dryad_dncjsxm85__npy__wfst_5gram__online_logit_replay/daily.csv),
[run metadata](../../results/20261126_dryad_dncjsxm85__npy__wfst_5gram__online_logit_replay/run.yaml),
and [neural linkage audit](../../results/20261126_dryad_dncjsxm85__npy__wfst_5gram__online_logit_replay/neural_linkage.csv).

## Archived overlap (not the full cohort)

For context, the 276 validation cue anchors cover ten of the fourteen
days. Both offline archives and the released online outputs can be
scored against the same 1,719 words on this subset:

| Output | Equal-day WER (mean ± sample SD) | Pooled WER |
|---|---:|---:|
| Authors' released online | 6.05% ± 5.34 pp | 5.24% (90 edits) |
| Archived P4 GRU + OPT | 2.84% ± 1.56 pp | 2.85% (49 edits) |
| Archived BiGRU + OPT | 2.13% ± 1.25 pp | 2.44% (42 edits) |

These are archive comparisons on cue anchors, not results on the full
cohort or an untouched independent test split.

## Why this is not an end-to-end result

The online pickle has a unique row index but no crosswalk to the neural
HDF5 session/block/trial IDs. Searching the native neural files found 1,061
unique normalized cue matches in `train` and 276 in `val`, all in consistent
within-day order. That leaves 291 rows unresolved. The `test` files have
no cue labels, and order alone does not solve the linkage: for example,
online day 76 ends with a gap of 25 rows and 45 unlabeled candidates. The
audit treats all cue matches as diagnostic anchors, not verified IDs.
To obtain an independent end-to-end comparison, resolve every target
recording, hold it out from fitting and selection, and retrain; do not
substitute the 7.02% GRU or 6.25% BiGRU validation archive WER.

## Re-run

From the repository root in PowerShell with `bci-repro` active and the
indexed WSL graph prepared as in the fresh-machine guide:

```powershell
$data = "studies/20240815_card/data"
$run = "studies/20240815_card/results/NEW_RUN_ID"
$replay = "studies/20240815_card/code/replay_online_wer.py"
python $replay prepare "$data/raw/dryad_dncjsxm85/dryad_dncjsxm85__t15_copyTask.pkl" $run
python $replay audit-neural $run "$data/extracted/dryad_dncjsxm85/hdf5_data_final"
```

Then in WSL, from the same repository root (substitute the same run ID):

```bash
run=studies/20240815_card/results/NEW_RUN_ID
python3 shared/tools/decode_wfst.py --binary ~/bin/wenet_lattice \
  --graph ~/tg5/TLG.fst --words ~/tg5/words.txt \
  --trials-dir "$run/runtime/logits" --meta "$run/runtime/meta.tsv" \
  --output "$run/final.tsv" --nbest 1
```

Back in PowerShell, `python $replay score $run` verifies complete one-to-one
coverage and identical references before writing both sets of word edits,
daily counts and a summary.
