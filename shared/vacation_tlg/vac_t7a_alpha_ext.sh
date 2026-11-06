#!/bin/bash
# T7a evening GPU extras (2026-10-03): alpha-extension for OPT (Card best was
# at the 0.75 sweep boundary in deep pools) + Qwen3.5-4B on Jude deep pool.
set -u
PY=/d/repositories/20260910_reproduce_20260629_levy/.venv312/Scripts/python.exe
OPTM="../20260710_Design_of_Large_Corpus/LLM/opt-6.7b"
QW35="../20260710_Design_of_Large_Corpus/LLM/Qwen3.5-4B"
SW="0.6,0.7,0.75,0.8,0.85,0.9,0.95"
cd /d/repositories/shared
LOG=code/vac_t7a_progress.log
LOCK=result/logs/vac_gpu.lock
mkdir -p result/logs
if [ -f "$LOCK" ]; then
  OLD=$(head -1 "$LOCK")
  if tasklist //FI "PID eq $OLD" 2>/dev/null | grep -q "$OLD"; then
    echo "ABORT: GPU lock held by PID $OLD" | tee -a "$LOG"; exit 1
  fi
fi
echo "$$" > "$LOCK"; echo "t7a extras" >> "$LOCK"
trap 'rm -f "$LOCK"' EXIT

run() {  # run <model> <dtype> <nbest> <out> <ref> <extra...>
  local M=$1 DT=$2 NB=$3 OUT=$4 REF=$5; shift 5
  [ -s "$OUT" ] && { echo "skip $OUT" | tee -a "$LOG"; return; }
  echo "=== START $OUT $(date '+%F %T')" | tee -a "$LOG"
  "$PY" code/opt_rescore.py "$NB" "$OUT" --model "$M" --dtype "$DT" --batch 48 \
    --acs 0.325 --search-acscale 0.325 --ref-tsv "$REF" "$@" >> "$LOG" 2>&1
  echo "=== EXIT $? $OUT $(date '+%F %T')" | tee -a "$LOG"
}

C=data/20241009_card_BtoT25; W=data/20230616_willett_BtoT24; J=data/20260428_jude
run "$OPTM" float16 "$C/vac/nbest400.tsv" "$C/vac/nbest400_opt_fx_card_aext.tsv" "$C/vac/nbest400_final.tsv" --sweep-alpha "$SW"
run "$OPTM" float16 "$C/run_120k/nbest_full.tsv" "$C/vac/opt_fx_card_aext.tsv"   "$C/run_120k/final_full.tsv"  --sweep-alpha "$SW"
run "$OPTM" float16 "$W/vac/nbest400.tsv"     "$W/vac/nbest400_opt_fx_wl_aext.tsv" "$W/vac/nbest400_final.tsv" --sweep-alpha "$SW"
run "$QW35" bfloat16 "$J/vac/nbest400.tsv"    "$J/vac/nbest400_qwen35_fx_jude.tsv" "$J/vac/nbest400_final.tsv" --sweep-alpha "0.3,0.4,0.5,0.55,0.65,0.75"
echo "=== T7A ALL DONE $(date '+%F %T')" | tee -a "$LOG"
