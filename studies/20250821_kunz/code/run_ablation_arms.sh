#!/bin/bash
# Ablation arms for the kunz dataset: imagined-only / attempted-only per participant
# (review A.4 task 1). Same protocol as the mixed runs: official rnn_args.yaml,
# best checkpoint by val greedy PER, eval on the closed-loop imagined *_eval blocks.
# Batches budgets match the mixed runs (t12 4k / t15 8k / t16 6k) so the only
# difference between arms is the training-file selection.
PY=/c/Users/user/.conda/envs/decoder/python.exe
cd /d/repositories/shared
LOG=train_ablation.log

run_arm () {
  local subj=$1 arm=$2 nb=$3
  echo "===== TRAIN $subj $arm ($(date +%H:%M:%S)) =====" >> $LOG
  $PY train_kunz.py "$subj" "$arm" "$nb" "/d/repositories/shared/data/20250821_kunz/run_${subj}_${arm}" 250 >> $LOG 2>&1
  echo "===== EVAL $subj $arm ($(date +%H:%M:%S)) =====" >> $LOG
  $PY eval_kunz.py "/d/repositories/shared/data/20250821_kunz/run_${subj}_${arm}" "$subj" >> $LOG 2>&1
}

run_arm t12 imagined  4000
run_arm t12 attempted 4000
run_arm t16 imagined  6000
run_arm t16 attempted 6000
run_arm t15 imagined  8000
run_arm t15 attempted 8000
echo "===== ALL ARMS DONE ($(date +%H:%M:%S)) =====" >> $LOG
