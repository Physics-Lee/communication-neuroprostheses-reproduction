#!/bin/bash
# T8 sweep runner (WSL): WIP x {willett full, jude full} at official params.
set -u
WORDS=../20230831_willett/data/20230616_willett_BtoT24/languageModel_5gram_work_words.txt
WVAC=../20230831_willett/data/20230616_willett_BtoT24/vac
JVAC=../20260428_jude/data/20260428_jude/vac
mkdir -p ~/vac_logs ~/vac_grid/wip "$WVAC" "$JVAC"
for W in -0.5 -0.2 0.2 0.5; do
  T=${W#-}
  OUT=~/vac_grid/wip/wl_w$T
  if [ ! -s "$OUT/final_full.tsv" ]; then
    echo "=== willett wip=$W $(date '+%F %T')"
    WIP=$W ~/src_lattice/wenet_sweep_wip.sh ~/tg5/TLG.fst "$WORDS" \
      ~/tlv/trials ~/tlv/meta.tsv "$OUT" full 8 >> ~/vac_logs/wip_sweep.log 2>&1
    cp "$OUT/final_full.tsv" "$WVAC/wip_$T"_final.tsv
  fi
done
for W in -0.5 0.5; do
  T=${W#-}
  OUT=~/vac_grid/wip/jude_w$T
  if [ ! -s "$OUT/final_full.tsv" ]; then
    echo "=== jude wip=$W $(date '+%F %T')"
    WIP=$W ~/src_lattice/wenet_sweep_wip.sh ~/tg5/TLG.fst "$WORDS" \
      ~/tlv_t17/trials ~/tlv_t17/meta.tsv "$OUT" full 8 >> ~/vac_logs/wip_sweep.log 2>&1
    cp "$OUT/final_full.tsv" "$JVAC/jwip_$T"_final.tsv
  fi
done
echo "WIP SWEEP DONE $(date '+%F %T')"
