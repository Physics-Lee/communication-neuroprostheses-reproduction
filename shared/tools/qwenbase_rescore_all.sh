#!/bin/bash
# Qwen3.5-4B-Base rescoring sweep — re-run of qwen_rescore_all.sh with the
# BASE (non-instruct) checkpoint, per the 2026-09-30 section-6 conclusion that
# sentence-likelihood rescoring needs a base LM. willett23 + card24 + jude26.
# Same params as the OPT-6.7B official runs and the Qwen3.5-4B instruct sweep:
# acs=0.325, search-acscale=0.325, length-penalty=0, augment top20, uppercase.
set -u
PY=/d/repositories/20260910_reproduce_20260629_levy/.venv312/Scripts/python.exe
QW="../20260710_Design_of_Large_Corpus/LLM/Qwen3.5-4B-Base"
cd /d/repositories/shared
ALPHAS=0.3,0.4,0.5,0.55,0.65,0.75

run() {  # run <nbest.tsv> <out.tsv> <ref.tsv>
  echo "=== START $1 $(date '+%F %T')" | tee -a code/qwenbase_rescore_progress.log
  "$PY" code/opt_rescore.py "$1" "$2" --model "$QW" --dtype bfloat16 \
    --batch 96 --acs 0.325 --search-acscale 0.325 --ref-tsv "$3" \
    --sweep-alpha "$ALPHAS"
  echo "=== EXIT $? $2 $(date '+%F %T')" | tee -a code/qwenbase_rescore_progress.log
}

W=data/20230616_willett_BtoT24/run_60k
C=data/20241009_card_BtoT25/run_120k
J=data/20260428_jude/run_25k
run $W/wl_nbest_full.tsv   $W/wl_5g_opt_qwenbase.tsv           $W/wl_final_full.tsv
run $C/nbest_full.tsv      $C/word_decode_5g_opt_qwenbase.tsv  $C/final_full.tsv
run $J/off_nbest_full.tsv  $J/off_5g_opt_qwenbase.tsv          $J/off_final_full.tsv
echo "=== ALL DONE $(date '+%F %T')" | tee -a code/qwenbase_rescore_progress.log
