#!/bin/bash
# T16 attempted-speech holdout baseline (review A.4 task 3):
# 3-fold block-level CV, 3,000 batches per fold, val every 100 batches
# (80 training trials per fold -> best checkpoint arrives within a few
# hundred batches; budget mirrors the t16 ablation arms' scale).
PY=/c/Users/user/.conda/envs/decoder/python.exe
cd /d/repositories/shared
LOG=train_t16_holdout.log

for fold in 1 2 3; do
  echo "===== TRAIN fold$fold ($(date +%H:%M:%S)) =====" >> $LOG
  $PY train_t16_holdout.py "$fold" 3000 "/d/repositories/shared/data/20250821_kunz/run_t16_holdout_fold$fold" 100 >> $LOG 2>&1
  echo "===== EVAL fold$fold ($(date +%H:%M:%S)) =====" >> $LOG
  $PY eval_t16_holdout.py "/d/repositories/shared/data/20250821_kunz/run_t16_holdout_fold$fold" "$fold" >> $LOG 2>&1
done
echo "===== HOLDOUT DONE ($(date +%H:%M:%S)) =====" >> $LOG
