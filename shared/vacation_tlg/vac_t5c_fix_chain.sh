#!/bin/bash
# T5c correction chain (2026-10-03): rerun ALL rescoring invalidated by the
# 09-30 opt_scores off-by-one (scoring token t with position-t logits) using
# the FIXED code/opt_rescore.py (validated: corr=1.000000 vs old scorer,
# Jude e2e 45.54 vs old-script 45.49). Defaults now: --case lower (official).
# Everything writes *_fx.tsv files; the broken 10-02 outputs are kept as-is
# for the audit trail.
set -u
PY=/d/repositories/20260910_reproduce_20260629_levy/.venv312/Scripts/python.exe
OPTM="../20260710_Design_of_Large_Corpus/LLM/opt-6.7b"
QW35="../20260710_Design_of_Large_Corpus/LLM/Qwen3.5-4B"
QW35B="../20260710_Design_of_Large_Corpus/LLM/Qwen3.5-4B-Base"
QW3="../20260710_Design_of_Large_Corpus/LLM/Qwen3-4B"
QW17="../20260710_Design_of_Large_Corpus/LLM/Qwen3-1.7B"
SW="0.3,0.4,0.5,0.55,0.65,0.75"
cd /d/repositories/shared
LOG=code/vac_t5c_progress.log
LOCK=result/logs/vac_gpu.lock
mkdir -p result/logs
if [ -f "$LOCK" ]; then
  OLD=$(head -1 "$LOCK")
  if tasklist //FI "PID eq $OLD" 2>/dev/null | grep -q "$OLD"; then
    echo "ABORT: GPU lock held by PID $OLD" | tee -a "$LOG"; exit 1
  fi
fi
echo "$$" > "$LOCK"; echo "t5c fix chain" >> "$LOCK"
trap 'rm -f "$LOCK"' EXIT

run() {  # run <model> <dtype> <nbest> <out> <ref> <sacs> <extra...>
  local M=$1 DT=$2 NB=$3 OUT=$4 REF=$5 SACS=$6; shift 6
  [ -s "$OUT" ] && { echo "skip $OUT" | tee -a "$LOG"; return; }
  echo "=== START $OUT $(date '+%F %T')" | tee -a "$LOG"
  "$PY" code/opt_rescore.py "$NB" "$OUT" --model "$M" --dtype "$DT" --batch 48 \
    --acs 0.325 --search-acscale "$SACS" --ref-tsv "$REF" "$@" >> "$LOG" 2>&1
  echo "=== EXIT $? $OUT $(date '+%F %T')" | tee -a "$LOG"
}

W=data/20230616_willett_BtoT24; C=data/20241009_card_BtoT25
J=data/20260428_jude; K=data/20250821_kunz/vac

# --- OPT-6.7B official protocol (alpha=0.55 fixed) ---
run "$OPTM" float16 "$W/run_60k/wl_nbest_full.tsv" "$W/vac/opt_fx_wl.tsv"     "$W/run_60k/wl_final_full.tsv"  0.325 --alpha 0.55
run "$OPTM" float16 "$C/run_120k/nbest_full.tsv"    "$C/vac/opt_fx_card.tsv"  "$C/run_120k/final_full.tsv"    0.325 --alpha 0.55
# jude done as opt_fixed_jude.tsv (45.54)

# --- Kunz OPT (alpha sweep), both t15 pools ---
run "$OPTM" float16 "$K/t12_bp1_a0.325_nbest.tsv" "$K/t12_opt_fx.tsv"     "$K/t12_bp1_a0.325_final.tsv" 0.325 --sweep-alpha "$SW"
run "$OPTM" float16 "$K/t15_bp1_a0.5_nbest.tsv"   "$K/t15_opt_fx.tsv"     "$K/t15_bp1_a0.5_final.tsv"   0.5   --sweep-alpha "$SW"
run "$OPTM" float16 "$K/t15_bp1_a0.325_nbest.tsv" "$K/t15a0325_opt_fx.tsv" "$K/t15_bp1_a0.325_final.tsv" 0.325 --sweep-alpha "$SW"
run "$OPTM" float16 "$K/t16_bp1_a0.325_nbest.tsv" "$K/t16_opt_fx.tsv"     "$K/t16_bp1_a0.325_final.tsv" 0.325 --sweep-alpha "$SW"

# --- deep-pool (NB=400) OPT ---
run "$OPTM" float16 "$W/vac/nbest400.tsv" "$W/vac/nbest400_opt_fx_wl.tsv"   "$W/vac/nbest400_final.tsv"   0.325 --sweep-alpha "$SW"
run "$OPTM" float16 "$C/vac/nbest400.tsv" "$C/vac/nbest400_opt_fx_card.tsv" "$C/vac/nbest400_final.tsv"   0.325 --sweep-alpha "$SW"
run "$OPTM" float16 "$J/vac/nbest400.tsv" "$J/vac/nbest400_opt_fx_jude.tsv" "$J/vac/nbest400_final.tsv"   0.325 --sweep-alpha "$SW"

# --- Qwen3.5-4B (instruct) re-validation: does the "family fails" conclusion survive? ---
run "$QW35" bfloat16 "$J/run_25k/off_nbest_full.tsv" "$J/run_25k/off_5g_qwen35_fx.tsv" "$J/run_25k/off_final_full.tsv" 0.325 --sweep-alpha "$SW"
run "$QW35" bfloat16 "$W/run_60k/wl_nbest_full.tsv"  "$W/run_60k/wl_5g_qwen35_fx.tsv"  "$W/run_60k/wl_final_full.tsv"  0.325 --sweep-alpha "$SW"
run "$QW35" bfloat16 "$C/run_120k/nbest_full.tsv"    "$C/run_120k/word_decode_5g_qwen35_fx.tsv" "$C/run_120k/final_full.tsv" 0.325 --sweep-alpha "$SW"

# --- Qwen3.5-4B-Base (jude) ---
run "$QW35B" bfloat16 "$J/run_25k/off_nbest_full.tsv" "$J/run_25k/off_5g_qwen35base_fx.tsv" "$J/run_25k/off_final_full.tsv" 0.325 --sweep-alpha "$SW"

# --- Qwen3-4B (pure-text dense base) redo ---
run "$QW3" bfloat16 "$J/run_25k/off_nbest_full.tsv" "$J/run_25k/off_5g_opt_qwen3base_fx.tsv" "$J/run_25k/off_final_full.tsv" 0.325 --sweep-alpha "$SW"
run "$QW3" bfloat16 "$W/run_60k/wl_nbest_full.tsv"  "$W/run_60k/wl_5g_opt_qwen3base_fx.tsv"  "$W/run_60k/wl_final_full.tsv"  0.325 --sweep-alpha "$SW"
run "$QW3" bfloat16 "$C/run_120k/nbest_full.tsv"    "$C/run_120k/word_decode_5g_opt_qwen3base_fx.tsv" "$C/run_120k/final_full.tsv" 0.325 --sweep-alpha "$SW"

# --- Qwen3-1.7B scale control (jude) ---
run "$QW17" bfloat16 "$J/run_25k/off_nbest_full.tsv" "$J/run_25k/off_5g_opt_qwen17_fx.tsv" "$J/run_25k/off_final_full.tsv" 0.325 --sweep-alpha "$SW"

echo "=== T5C ALL DONE $(date '+%F %T')" | tee -a "$LOG"
