#!/bin/bash
# Qwen3.5-4B rescoring sweep over the four papers' n-best lists (2026-09-30).
# Same decoding params as the OPT-6.7B official runs: acs=0.325,
# search-acscale=0.325, length-penalty=0, augment top20, uppercase input;
# alpha swept (official 0.55 included) because Qwen's summed token log-prob
# has a different length bias than OPT's.
set -u
PY=/d/repositories/20260910_reproduce_20260629_levy/.venv312/Scripts/python.exe
QW="../20260710_Design_of_Large_Corpus/LLM/Qwen3.5-4B"
cd /d/repositories/shared
ALPHAS=0.3,0.4,0.5,0.55,0.65,0.75

run() {  # run <nbest.tsv> <out.tsv> <ref.tsv>
  echo "=== START $1 $(date '+%F %T')" | tee -a code/qwen_rescore_progress.log
  "$PY" code/opt_rescore.py "$1" "$2" --model "$QW" --dtype bfloat16 \
    --batch 96 --acs 0.325 --search-acscale 0.325 --ref-tsv "$3" \
    --sweep-alpha "$ALPHAS"
  echo "=== EXIT $? $2 $(date '+%F %T')" | tee -a code/qwen_rescore_progress.log
}

K=data/20250821_kunz
run $K/run_t16_mixed/off_bp1_nbest.tsv $K/run_t16_mixed/off_bp1_opt_qwen.tsv $K/run_t16_mixed/off_bp1_final.tsv
run $K/run_t12_mixed/off_bp1_nbest.tsv $K/run_t12_mixed/off_bp1_opt_qwen.tsv $K/run_t12_mixed/off_bp1_final.tsv
run $K/run_t15_mixed/off_bp1_nbest.tsv $K/run_t15_mixed/off_bp1_opt_qwen.tsv $K/run_t15_mixed/off_bp1_final.tsv
run data/20260428_jude/run_25k/off_nbest_full.tsv data/20260428_jude/run_25k/off_5g_opt_qwen.tsv data/20260428_jude/run_25k/off_final_full.tsv
run data/20230616_willett_BtoT24/run_60k/wl_nbest_full.tsv data/20230616_willett_BtoT24/run_60k/wl_5g_opt_qwen.tsv data/20230616_willett_BtoT24/run_60k/wl_final_full.tsv
run data/20241009_card_BtoT25/run_120k/nbest_full.tsv data/20241009_card_BtoT25/run_120k/word_decode_5g_opt_qwen.tsv data/20241009_card_BtoT25/run_120k/final_full.tsv
echo "=== ALL DONE $(date '+%F %T')" | tee -a code/qwen_rescore_progress.log
