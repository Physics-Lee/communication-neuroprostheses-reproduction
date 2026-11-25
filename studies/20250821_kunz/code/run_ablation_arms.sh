#!/bin/bash
# Ablation arms for the kunz dataset: imagined-only / attempted-only per participant
# (review A.4 task 1). Same protocol as the mixed runs: official rnn_args.yaml,
# best checkpoint by val greedy PER, eval on the closed-loop imagined *_eval blocks.
# Batches budgets match the mixed runs (t12 4k / t15 8k / t16 6k) so the only
# difference between arms is the training-file selection.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PY=${PYTHON:-python}
run_date=$(date +%Y%m%d)

run_arm () {
  local subj=$1 arm=$2 nb=$3
  local id="${run_date}_dryad_gf1vhhn1j__hdf5__ctc_greedy__${subj}_${arm}"
  local out="results/$id/runtime" log="log/$id"
  mkdir -- "results/$id"
  mkdir -p -- "$log"
  "$PY" code/train_kunz.py "$subj" "$arm" "$nb" "$out" 250 >> "$log/driver.log" 2>&1
  "$PY" code/eval_kunz.py "$out" "$subj" >> "$log/driver.log" 2>&1
  mv -- "$out/training_log" "$log/training_log"
  "$PY" ../../shared/tools/record_study_run.py 20250821_kunz "$out/record" \
    --input-format hdf5 --decoder ctc_greedy --split validation --trials "$out/val_greedy_per_trial.csv"
  mv -- "$out/record/run.yaml" "$out/record/trials.csv" "results/$id/"
  rmdir -- "$out/record"
}

run_arm t12 imagined  4000
run_arm t12 attempted 4000
run_arm t16 imagined  6000
run_arm t16 attempted 6000
run_arm t15 imagined  8000
run_arm t15 attempted 8000
