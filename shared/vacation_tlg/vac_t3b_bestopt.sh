#!/bin/bash
# T3b: Willett best-decode-param (bp90/acs0.4 from the T3 grid) + fixed OPT.
# 1) copy the grid cell's NB=100 nbest from WSL, rescore (alpha sweep);
# 2) decode NB=400 at bp90/acs0.4, rescore the deep pool too.
set -u
PY=/d/repositories/20260910_reproduce_20260629_levy/.venv312/Scripts/python.exe
OPTM="../20260710_Design_of_Large_Corpus/LLM/opt-6.7b"
SW="0.3,0.4,0.5,0.55,0.65,0.75"
cd /d/repositories/shared
LOG=code/vac_t3b_progress.log
LOCK=result/logs/vac_gpu.lock
W=data/20230616_willett_BtoT24/vac
mkdir -p result/logs "$W"
wsl.exe bash -lc 'cp ~/vac_grid/willett/grid_bp90_a0.4/nbest_full.tsv ../20230831_willett/data/20230616_willett_BtoT24/vac/grid_bp90_a04_nbest.tsv; cp ~/vac_grid/willett/grid_bp90_a0.4/final_full.tsv ../20230831_willett/data/20230616_willett_BtoT24/vac/grid_bp90_a04_final.tsv'
# decode NB=400 at the best cell (WSL, ~8 min) — runs while GPU rescoring
wsl.exe bash -lc 'nohup bash -c "ACS=0.4 BP=90 NB=400 ~/src_lattice/wenet_sweep.sh ~/tg5/TLG.fst ../20230831_willett/data/20230616_willett_BtoT24/languageModel_5gram_work_words.txt ~/tlv/trials ~/tlv/meta.tsv ~/vac_grid/nb400_wl_a04 full 8 >> ~/vac_logs/t3b.log 2>&1; cp ~/vac_grid/nb400_wl_a04/nbest_full.tsv ../20230831_willett/data/20230616_willett_BtoT24/vac/nbest400_a04.tsv; cp ~/vac_grid/nb400_wl_a04/final_full.tsv ../20230831_willett/data/20230616_willett_BtoT24/vac/nbest400_a04_final.tsv" > ~/vac_logs/t3b_console.log 2>&1 & disown; echo wl-nb400-a04-launched'

if [ -f "$LOCK" ]; then
  OLD=$(head -1 "$LOCK")
  if tasklist //FI "PID eq $OLD" 2>/dev/null | grep -q "$OLD"; then
    echo "ABORT: GPU lock held" | tee -a "$LOG"; exit 1
  fi
fi
echo "$$" > "$LOCK"; echo "t3b bestopt" >> "$LOCK"
trap 'rm -f "$LOCK"' EXIT
"$PY" code/opt_rescore.py "$W/grid_bp90_a04_nbest.tsv" "$W/grid_bp90_a04_opt_fx.tsv" \
  --model "$OPTM" --dtype float16 --batch 48 --acs 0.325 --search-acscale 0.4 \
  --ref-tsv "$W/grid_bp90_a04_final.tsv" --sweep-alpha "$SW" >> "$LOG" 2>&1
echo "=== grid cell OPT done $(date '+%F %T')" | tee -a "$LOG"
# wait for the WSL NB=400 decode, then rescore it
for i in $(seq 1 60); do [ -s "$W/nbest400_a04.tsv" ] && break; sleep 30; done
if [ -s "$W/nbest400_a04.tsv" ]; then
  "$PY" code/opt_rescore.py "$W/nbest400_a04.tsv" "$W/nbest400_a04_opt_fx.tsv" \
    --model "$OPTM" --dtype float16 --batch 48 --acs 0.325 --search-acscale 0.4 \
    --ref-tsv "$W/nbest400_a04_final.tsv" --sweep-alpha "$SW" >> "$LOG" 2>&1
  echo "=== deep-pool OPT done $(date '+%F %T')" | tee -a "$LOG"
else
  echo "deep pool not ready in 30min, skipped" | tee -a "$LOG"
fi
echo "=== T3B ALL DONE $(date '+%F %T')" | tee -a "$LOG"
