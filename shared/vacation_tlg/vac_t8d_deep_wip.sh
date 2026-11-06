#!/bin/bash
# T8d WSL: Jude deep pool (NB=400) at wip=+0.2 — the "everything combined" config.
set -u
WORDS=../20230831_willett/data/20230616_willett_BtoT24/languageModel_5gram_work_words.txt
JVAC=../20260428_jude/data/20260428_jude/vac
mkdir -p ~/vac_logs ~/vac_grid/wip
OUT=~/vac_grid/wip/jd_deep_wp02
if [ ! -s "$OUT/nbest_full.tsv" ]; then
  echo "=== jd deep NB=400 wip=+0.2 $(date '+%F %T')"
  WIP=0.2 NB=400 ~/src_lattice/wenet_sweep_wip.sh ~/tg5/TLG.fst "$WORDS" \
    ~/tlv_t17/trials ~/tlv_t17/meta.tsv "$OUT" full 8 >> ~/vac_logs/wip_sweep.log 2>&1
  cp "$OUT/nbest_full.tsv" "$JVAC/jdeep_wp02_nbest400.tsv"
  cp "$OUT/final_full.tsv" "$JVAC/jdeep_wp02_nbest400_final.tsv"
fi
echo "T8D WSL DONE $(date '+%F %T')"
