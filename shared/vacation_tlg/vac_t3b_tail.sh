#!/bin/bash
# waits for the WSL NB400@a0.4 decode, then OPT-rescores the deep pool
set -u
PY=/d/repositories/20260910_reproduce_20260629_levy/.venv312/Scripts/python.exe
OPTM="../20260710_Design_of_Large_Corpus/LLM/opt-6.7b"
cd /d/repositories/shared
W=data/20230616_willett_BtoT24/vac
for i in $(seq 1 90); do [ -s "$W/nbest400_a04.tsv" ] && break; sleep 30; done
if [ ! -s "$W/nbest400_a04.tsv" ]; then echo "deep pool still missing after 45min" ; exit 1; fi
echo "$$" > result/logs/vac_gpu.lock
"$PY" code/opt_rescore.py "$W/nbest400_a04.tsv" "$W/nbest400_a04_opt_fx.tsv" \
  --model "$OPTM" --dtype float16 --batch 48 --acs 0.325 --search-acscale 0.4 \
  --ref-tsv "$W/nbest400_a04_final.tsv" --sweep-alpha 0.3,0.4,0.5,0.55,0.65,0.75
rm -f result/logs/vac_gpu.lock
echo "T3B TAIL DONE $(date '+%F %T')"
