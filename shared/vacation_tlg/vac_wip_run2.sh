#!/bin/bash
# T8 take 2: positive-wip configs with collision-free names (wm=bonus, wp=cost).
# Negative side already measured (vac/wip_0.2=-0.2, wip_0.5=-0.5 by name collision).
set -u
WORDS=../20230831_willett/data/20230616_willett_BtoT24/languageModel_5gram_work_words.txt
WVAC=../20230831_willett/data/20230616_willett_BtoT24/vac
JVAC=../20260428_jude/data/20260428_jude/vac
mkdir -p ~/vac_logs ~/vac_grid/wip
run() {  # run <wip> <tag> <trials> <meta> <copy>
  local W=$1 TAG=$2 TR=$3 ME=$4 CP=$5
  OUT=~/vac_grid/wip/$TAG
  [ -s "$OUT/final_full.tsv" ] && { echo "skip $TAG"; return; }
  echo "=== $TAG wip=$W $(date '+%F %T')"
  WIP=$W ~/src_lattice/wenet_sweep_wip.sh ~/tg5/TLG.fst "$WORDS" \
    "$TR" "$ME" "$OUT" full 8 >> ~/vac_logs/wip_sweep.log 2>&1
  cp "$OUT/final_full.tsv" "$CP"
}
run  0.2 wl_wp02 ~/tlv/trials    ~/tlv/meta.tsv    "$WVAC/wip_p02_final.tsv"
run  0.5 wl_wp05 ~/tlv/trials    ~/tlv/meta.tsv    "$WVAC/wip_p05_final.tsv"
run -0.2 jd_wm02 ~/tlv_t17/trials ~/tlv_t17/meta.tsv "$JVAC/jwip_m02_final.tsv"
run  0.2 jd_wp02 ~/tlv_t17/trials ~/tlv_t17/meta.tsv "$JVAC/jwip_p02_final.tsv"
echo "WIP2 DONE $(date '+%F %T')"
