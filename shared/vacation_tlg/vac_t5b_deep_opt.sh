#!/bin/bash
# T5b evening GPU chain (2026-10-02): protocol-drift baselines + Kunz OPT
# completions + deep-pool (NB=400) OPT + Qwen3-1.7B scale control.
# Context: 09-30 opt_rescore upgrade (both-variant augmentation, fp32 logprob)
# matches the OFFICIAL upstream augment_nbest; all doc "+OPT" numbers were made
# with the old protocol. This chain re-runs OPT baselines with the CURRENT
# script so vacation numbers have a same-protocol reference.
set -u
PY=/d/repositories/20260910_reproduce_20260629_levy/.venv312/Scripts/python.exe
OPTM="../20260710_Design_of_Large_Corpus/LLM/opt-6.7b"
QW17="../20260710_Design_of_Large_Corpus/LLM/Qwen3-1.7B"
cd /d/repositories/shared
LOG=code/vac_t5b_progress.log
LOCK=result/logs/vac_gpu.lock
mkdir -p result/logs

if [ -f "$LOCK" ]; then
  OLD=$(head -1 "$LOCK")
  if tasklist //FI "PID eq $OLD" 2>/dev/null | grep -q "$OLD"; then
    echo "ABORT: GPU lock held by PID $OLD ($(tail -1 "$LOCK"))" | tee -a "$LOG"
    exit 1
  fi
fi
echo "$$" > "$LOCK"; echo "t5b evening chain" >> "$LOCK"
trap 'rm -f "$LOCK"' EXIT

run() {  # run <model> <nbest> <out> <ref> <sacs> <extra args...>
  local MODEL=$1 NB=$2 OUT=$3 REF=$4 SACS=$5; shift 5
  [ -s "$OUT" ] && { echo "skip $OUT (exists)" | tee -a "$LOG"; return; }
  echo "=== START $OUT $(date '+%F %T')" | tee -a "$LOG"
  "$PY" code/opt_rescore.py "$NB" "$OUT" --model "$MODEL" --dtype float16 --batch 48 \
    --acs 0.325 --search-acscale "$SACS" --ref-tsv "$REF" "$@" >> "$LOG" 2>&1
  echo "=== EXIT $? $OUT $(date '+%F %T')" | tee -a "$LOG"
}

W=data/20230616_willett_BtoT24; C=data/20241009_card_BtoT25
J=data/20260428_jude; K=data/20250821_kunz/vac

# --- 1) OPT baselines under CURRENT script, official fixed alpha=0.55 ---
run "$OPTM" "$W/run_60k/wl_nbest_full.tsv"        "$W/vac/opt_rerun_willett.tsv" "$W/run_60k/wl_final_full.tsv"        0.325 --alpha 0.55
run "$OPTM" "$J/run_25k/off_nbest_full.tsv"       "$J/vac/opt_rerun_jude.tsv"    "$J/run_25k/off_final_full.tsv"       0.325 --alpha 0.55
run "$OPTM" "$C/run_120k/nbest_full.tsv"          "$C/vac/opt_rerun_card.tsv"    "$C/run_120k/final_full.tsv"          0.325 --alpha 0.55

# --- 2) Kunz T4 completions (t12/t16, best cells from T1) ---
run "$OPTM" "$K/t12_bp1_a0.325_nbest.tsv" "$K/t12_best_opt.tsv" "$K/t12_bp1_a0.325_final.tsv" 0.325 --sweep-alpha 0.3,0.4,0.5,0.55,0.65,0.75
run "$OPTM" "$K/t16_bp1_a0.325_nbest.tsv" "$K/t16_best_opt.tsv" "$K/t16_bp1_a0.325_final.tsv" 0.325 --sweep-alpha 0.3,0.4,0.5,0.55,0.65,0.75

# --- 3) deep-pool (NB=400) OPT, official params ---
run "$OPTM" "$W/vac/nbest400.tsv" "$W/vac/nbest400_opt.tsv" "$W/vac/nbest400_final.tsv" 0.325 --sweep-alpha 0.3,0.4,0.5,0.55,0.65,0.75
run "$OPTM" "$C/vac/nbest400.tsv" "$C/vac/nbest400_opt.tsv" "$C/vac/nbest400_final.tsv" 0.325 --sweep-alpha 0.3,0.4,0.5,0.55,0.65,0.75
run "$OPTM" "$J/vac/nbest400.tsv" "$J/vac/nbest400_opt.tsv" "$J/vac/nbest400_final.tsv" 0.325 --sweep-alpha 0.3,0.4,0.5,0.55,0.65,0.75

# --- 4) Qwen3-1.7B scale control on Jude (standard pool) ---
run "$QW17" "$J/run_25k/off_nbest_full.tsv" "$J/run_25k/off_5g_opt_qwen17.tsv" "$J/run_25k/off_final_full.tsv" 0.325 --dtype bfloat16 --sweep-alpha 0.3,0.4,0.5,0.55,0.65,0.75

echo "=== T5B ALL DONE $(date '+%F %T')" | tee -a "$LOG"
