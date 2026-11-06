#!/bin/bash
# T4: OPT-6.7B rescoring of best-param Kunz n-best lists.
# Best cells per arm (from T1 sweep + T16's existing six-cell sweep):
#   T12 bp1/a0.325, T15 bp1/a0.5, T16 bp1/a0.325.
# Official OPT protocol: float16, batch 48, acs 0.325, alpha sweep 0.3-0.75.
# Run from repo root in Git Bash. GPU lock: result/logs/vac_gpu.lock.
set -u
PY=/d/repositories/20260910_reproduce_20260629_levy/.venv312/Scripts/python.exe
OPT="../20260710_Design_of_Large_Corpus/LLM/opt-6.7b"
cd /d/repositories/shared
LOG=code/vac_t4_progress.log
LOCK=result/logs/vac_gpu.lock
K=data/20250821_kunz/vac
mkdir -p result/logs "$K"

if [ -f "$LOCK" ]; then
  OLD=$(head -1 "$LOCK")
  if tasklist //FI "PID eq $OLD" 2>/dev/null | grep -q "$OLD"; then
    echo "ABORT: GPU lock held by PID $OLD ($(tail -1 "$LOCK"))" | tee -a "$LOG"
    exit 1
  fi
  echo "stale lock (PID $OLD gone), taking over" | tee -a "$LOG"
fi
echo "$$" > "$LOCK"; echo "kunz opt chain" >> "$LOCK"
trap 'rm -f "$LOCK"' EXIT

# fetch the two best-cell outputs that predate this vacation run (WSL).
# NOTE: no shell vars inside the wsl.exe single-quoted string — Git Bash's
# MSYS layer strips them ($R arrives empty in WSL). Use explicit paths.
V=../20250821_kunz/data/20250821_kunz/vac
wsl.exe bash -lc "cp ~/tlv_kunz_t12_mixed/sw_bp1_a0.325/nbest_full.tsv $V/t12_bp1_a0.325_nbest.tsv; cp ~/tlv_kunz_t12_mixed/sw_bp1_a0.325/final_full.tsv $V/t12_bp1_a0.325_final.tsv; cp ~/tlv_kunz_t16_mixed/sw_bp1_a0.325/nbest_full.tsv $V/t16_bp1_a0.325_nbest.tsv; cp ~/tlv_kunz_t16_mixed/sw_bp1_a0.325/final_full.tsv $V/t16_bp1_a0.325_final.tsv"

run() {  # run <nbest> <out> <ref> <search_acs>
  [ -s "$2" ] && { echo "skip $2 (exists)" | tee -a "$LOG"; return; }
  echo "=== START $1 $(date '+%F %T')" | tee -a "$LOG"
  "$PY" code/opt_rescore.py "$1" "$2" --model "$OPT" --dtype float16 --batch 48 \
    --acs 0.325 --search-acscale "$4" --ref-tsv "$3" \
    --sweep-alpha 0.3,0.4,0.5,0.55,0.65,0.75 >> "$LOG" 2>&1
  echo "=== EXIT $? $2 $(date '+%F %T')" | tee -a "$LOG"
}

run "$K/t12_bp1_a0.325_nbest.tsv" "$K/t12_best_opt.tsv" "$K/t12_bp1_a0.325_final.tsv" 0.325
run "$K/t15_bp1_a0.5_nbest.tsv"   "$K/t15_best_opt.tsv" "$K/t15_bp1_a0.5_final.tsv"   0.5
run "$K/t16_bp1_a0.325_nbest.tsv" "$K/t16_best_opt.tsv" "$K/t16_bp1_a0.325_final.tsv" 0.325
echo "=== T4 ALL DONE $(date '+%F %T')" | tee -a "$LOG"
