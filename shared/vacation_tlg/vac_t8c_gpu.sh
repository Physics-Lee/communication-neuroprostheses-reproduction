#!/bin/bash
# T8c GPU tail: after the WSL part, OPT-rescore the Jude wip=+0.2 pool.
set -u
PY=/d/repositories/20260910_reproduce_20260629_levy/.venv312/Scripts/python.exe
OPTM="../20260710_Design_of_Large_Corpus/LLM/opt-6.7b"
cd /d/repositories/shared
J=data/20260428_jude/vac
LOG=code/vac_t8c_progress.log
for i in $(seq 1 40); do [ -s "$J/jwip_p02_nbest.tsv" ] && break; sleep 20; done
[ -s "$J/jwip_p02_nbest.tsv" ] || { echo "pool missing" | tee -a "$LOG"; exit 1; }
echo "$$" > result/logs/vac_gpu.lock
"$PY" code/opt_rescore.py "$J/jwip_p02_nbest.tsv" "$J/jwip_p02_opt_fx.tsv" \
  --model "$OPTM" --dtype float16 --batch 48 --acs 0.325 --search-acscale 0.325 \
  --ref-tsv "$J/jwip_p02_5g_final.tsv" --sweep-alpha 0.3,0.4,0.5,0.55,0.65,0.75 >> "$LOG" 2>&1
rm -f result/logs/vac_gpu.lock
echo "T8C GPU DONE $(date '+%F %T')" | tee -a "$LOG"
