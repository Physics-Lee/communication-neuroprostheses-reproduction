# Card 2024 Figure 2 neural reproduction

This is a separate **from-scratch** run from the [word-decoder
replay](same_trial_replay.md). It trains a five-layer GRU on native neural
features, uses the day-specific layer for each test recording, then decodes
with our fixed 1-best 5-gram WFST. The authors' logits contribute only their
frame counts for record linkage and are not used for training or inference.
This is an offline comparison, not a reconstruction of the authors'
chronological training, adaptation, or second-pass language-model reranking.

## Result

| Output | Mean daily WER ± sample SD | Pooled WER |
|---|---:|---:|
| Authors' released online | 4.85% ± 3.35 pp | 4.42% (454 / 10,266) |
| Our from-scratch GRU + 5-gram WFST | 5.67% ± 1.19 pp | 5.79% (594 / 10,266) |

On the same 1,617 mapped trials, the reproduction was 0.82 percentage
points higher by the daily mean and 1.36 points higher pooled, with 140
more word edits. The scorer verified one prediction and the same reference
for each manifest ID. Training took 179.43 minutes and WFST decoding
took 20 minutes 10 seconds, excluding setup and inference.

[Per-day results](../../results/20261126_dryad_dncjsxm85__hdf5__gru_wfst__figure2_neural/daily.csv) ·
[Scored trials](../../results/20261126_dryad_dncjsxm85__hdf5__gru_wfst__figure2_neural/trials.csv) ·
[Metric summary](../../results/20261126_dryad_dncjsxm85__hdf5__gru_wfst__figure2_neural/summary.json)

## Cohort and split

The run contains [1,617 mapped test
trials](../../results/20261126_dryad_dncjsxm85__hdf5__gru_wfst__figure2_neural/cohort.csv):
1,337 unique-cue anchors and 280 unlabeled HDF5 recordings matched by the
relationship between online logit frame count and neural recording length
within ordered gaps. A timing pair is taken only if it is present in every
maximum-cardinality ordered match under the configured offset windows.
These are inferred links, not publisher IDs. Eleven rows are
[unresolved](../../results/20261126_dryad_dncjsxm85__hdf5__gru_wfst__figure2_neural/unresolved.csv).
Three of them have no unlabeled candidate even between their immediate
anchors: the expected trial index is absent or only a recording with a
different cue is present. Scoring all 1,628 would require a publisher
crosswalk and potentially additional neural recordings.

The fit uses 7,007 training trials and chooses the checkpoint using 1,085
validation trials. Entire blocks containing cue anchors or unlabeled
candidates are excluded from both roles. The fixed recipe runs 12,000
optimizer steps (shorter than the 120,000-step competition baseline), with
no training or language-model tuning on scored trials.

## Commands

Run from the repository root with `bci-repro` active:

```powershell
$s = "studies/20240815_card"
$data = "$s/data"
$replay = "$s/results/20261126_dryad_dncjsxm85__npy__wfst_5gram__online_logit_replay"
$neural = "$data/extracted/dryad_dncjsxm85/hdf5_data_final"
$run = "$s/results/NEW_RUN_ID"
python "$s/code/reproduce_neural_wer.py" $replay "$data/raw/dryad_dncjsxm85/dryad_dncjsxm85__t15_copyTask.pkl" $neural $run
python "$s/code/train.py" 12000 "$run/runtime/model" 1000 4 --split-view "$run/runtime/neural_view"
python "$s/code/infer.py" "$run/runtime/model" --manifest "$run/cohort.csv" --manifest-output "$run/runtime/inference" --neural-root $neural
```

From the same checkout in WSL:

```bash
run=studies/20240815_card/results/NEW_RUN_ID
python3 shared/tools/decode_wfst.py --binary ~/bin/wenet_lattice \
  --graph ~/tg5/TLG.fst --words ~/tg5/words.txt \
  --trials-dir "$run/runtime/inference/logits" --meta "$run/runtime/inference/meta.tsv" \
  --output "$run/final.tsv" --nbest 1
```

Back in PowerShell:

```powershell
python "$s/code/replay_online_wer.py" score $run --expected-count 1617 --arm from_scratch_gru_wfst
```
