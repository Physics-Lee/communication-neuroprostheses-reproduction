#!/bin/bash
# T2: Qwen3-4B (pure-text dense base LM) rescoring chain — architecture
# attribution vs the failed Qwen3.5-4B (hybrid VL). Same protocol as the
# OPT-6.7B official runs: acs=0.325, search-acscale=0.325, lp=0, augment
# top20, uppercase, alpha sweep. Sequential GPU jobs: Jude -> Willett -> Card.
# Run from repo root in Git Bash. GPU lock: result/logs/vac_gpu.lock.
set -u
PY=/d/repositories/20260910_reproduce_20260629_levy/.venv312/Scripts/python.exe
QW3="../20260710_Design_of_Large_Corpus/LLM/Qwen3-4B"
cd /d/repositories/shared
LOG=code/vac_t2_qwen3_progress.log
LOCK=result/logs/vac_gpu.lock
mkdir -p result/logs

# --- GPU lock (abort if another holder is alive) ---
if [ -f "$LOCK" ]; then
  OLD=$(head -1 "$LOCK")
  if tasklist //FI "PID eq $OLD" 2>/dev/null | grep -q "$OLD"; then
    echo "ABORT: GPU lock held by PID $OLD ($(cat "$LOCK" | tail -1))" | tee -a "$LOG"
    exit 1
  fi
  echo "stale lock (PID $OLD gone), taking over" | tee -a "$LOG"
fi
echo "$$" > "$LOCK"
echo "qwen3-4b chain" >> "$LOCK"
trap 'rm -f "$LOCK"' EXIT

run() {  # run <nbest.tsv> <out.tsv> <ref.tsv>
  if [ -s "$2" ]; then echo "skip $2 (exists)" | tee -a "$LOG"; return; fi
  echo "=== START $1 $(date '+%F %T')" | tee -a "$LOG"
  "$PY" code/opt_rescore.py "$1" "$2" --model "$QW3" --dtype bfloat16 \
    --batch 96 --acs 0.325 --search-acscale 0.325 --ref-tsv "$3" \
    --sweep-alpha 0.3,0.4,0.5,0.55,0.65,0.75 >> "$LOG" 2>&1
  echo "=== EXIT $? $2 $(date '+%F %T')" | tee -a "$LOG"
}

J=data/20260428_jude/run_25k
W=data/20230616_willett_BtoT24/run_60k
C=data/20241009_card_BtoT25/run_120k
run "$J/off_nbest_full.tsv" "$J/off_5g_opt_qwen3base.tsv" "$J/off_final_full.tsv"
run "$W/wl_nbest_full.tsv" "$W/wl_5g_opt_qwen3base.tsv" "$W/wl_final_full.tsv"
run "$C/nbest_full.tsv" "$C/word_decode_5g_opt_qwen3base.tsv" "$C/final_full.tsv"
echo "=== T2 ALL DONE $(date '+%F %T')" | tee -a "$LOG"
