#!/bin/bash
# Rapid-calibration Step-3 chain: N-trial sweep on target day t15.2025.03.30,
# 2000 optimizer steps each, val every 100 (target-day val = 30 trials).
PY=/c/Users/user/.conda/envs/decoder/python.exe
cd /d/repositories/shared/code
LOG=/d/repositories/shared/result/logs/finetune_chain_20260904.log
mkdir -p /d/repositories/shared/result/logs

for N in 10 50 165; do
  echo "===== PREP N=$N $(date +%H:%M:%S) =====" >> $LOG
  $PY prep_rapid_calibration.py t15.2025.03.30 $N >> $LOG 2>&1
  echo "===== TRAIN N=$N $(date +%H:%M:%S) =====" >> $LOG
  $PY train_finetune.py "D:\\repositories\\20240815_card\\data\\20241009_card_BtoT25\\ft_t1520250330_N$N" 2000 100 >> $LOG 2>&1
done
echo "===== CHAIN DONE $(date +%H:%M:%S) =====" >> $LOG
