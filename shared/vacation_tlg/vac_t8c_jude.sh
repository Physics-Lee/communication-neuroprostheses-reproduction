#!/bin/bash
# T8c: Jude WIP completion — wip=+0.5 decode, then OPT fx on the best (wip=+0.2) pool.
# WSL part first; the Windows GPU part is chained from vac_t8c_gpu.sh.
set -u
WORDS=../20230831_willett/data/20230616_willett_BtoT24/languageModel_5gram_work_words.txt
JVAC=../20260428_jude/data/20260428_jude/vac
mkdir -p ~/vac_logs
OUT=~/vac_grid/wip/jd_wp05
if [ ! -s "$OUT/final_full.tsv" ]; then
  echo "=== jd wip=+0.5 $(date '+%F %T')"
  WIP=0.5 ~/src_lattice/wenet_sweep_wip.sh ~/tg5/TLG.fst "$WORDS" \
    ~/tlv_t17/trials ~/tlv_t17/meta.tsv "$OUT" full 8 >> ~/vac_logs/wip_sweep.log 2>&1
  cp "$OUT/final_full.tsv" "$JVAC/jwip_p05_final.tsv"
fi
# expose the wip=+0.2 pool for Windows-side OPT rescoring
cp ~/vac_grid/wip/jd_wp02/nbest_full.tsv "$JVAC/jwip_p02_nbest.tsv"
cp ~/vac_grid/wip/jd_wp02/final_full.tsv "$JVAC/jwip_p02_5g_final.tsv"
echo "T8C WSL DONE $(date '+%F %T')"
