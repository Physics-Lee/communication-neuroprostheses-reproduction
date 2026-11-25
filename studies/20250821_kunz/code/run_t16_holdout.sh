#!/bin/bash
# T16 attempted-speech holdout baseline (review A.4 task 3):
# 3-fold block-level CV, 3,000 batches per fold, val every 100 batches
# (80 training trials per fold -> best checkpoint arrives within a few
# hundred batches; budget mirrors the t16 ablation arms' scale).
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PY=${PYTHON:-python}
run_date=$(date +%Y%m%d)

for fold in 1 2 3; do
  id="${run_date}_dryad_gf1vhhn1j__hdf5__ctc_greedy__t16_holdout_fold$fold"
  out="results/$id/runtime"
  log="log/$id"
  mkdir -- "results/$id"
  mkdir -p -- "$log"
  "$PY" code/train_t16_holdout.py "$fold" 3000 "$out" 100 >> "$log/driver.log" 2>&1
  "$PY" code/eval_t16_holdout.py "$out" "$fold" >> "$log/driver.log" 2>&1
  mv -- "$out/training_log" "$log/training_log"
  "$PY" ../../shared/tools/record_study_run.py 20250821_kunz "$out/record" \
    --input-format hdf5 --decoder ctc_greedy --split validation --trials "$out/val_greedy_per_trial.csv"
  mv -- "$out/record/run.yaml" "$out/record/trials.csv" "results/$id/"
  rmdir -- "$out/record"
done
