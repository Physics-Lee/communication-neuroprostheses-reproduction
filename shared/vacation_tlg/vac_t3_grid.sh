#!/bin/bash
# T3: official-decoder bp x acs fine grid on Willett (880) and Card (1426).
# bp in {9,20,40,90,150} x acs in {0.25,0.325,0.4}, full trials, NB=100.
# Run inside WSL: bash ../shared/code/vac_t3_grid.sh
# Skips existing cells. Copies final_full.tsv into the repo vac dirs.
set -u
REPO=../shared
WORDS=../20230831_willett/data/20230616_willett_BtoT24/languageModel_5gram_work_words.txt
WVAC=../20230831_willett/data/20230616_willett_BtoT24/vac
CVAC=../20240815_card/data/20241009_card_BtoT25/vac
mkdir -p ~/vac_logs "$WVAC" "$CVAC"

grid() { # <trials> <meta> <outdirbase> <copydir> <prefix>
  for BP in 9 20 40 90 150; do
    for ACS in 0.25 0.325 0.4; do
      TAG=bp${BP}_a${ACS}
      OUT=$3/grid_$TAG
      if [ -s "$OUT/final_full.tsv" ]; then
        echo "skip $5 $TAG (exists) $(date '+%F %T')"
        continue
      fi
      echo "=== $5 $TAG start $(date '+%F %T')"
      BP=$BP ACS=$ACS NB=100 ~/src_lattice/wenet_sweep.sh ~/tg5/TLG.fst "$WORDS" \
        "$1" "$2" "$OUT" full 8 >> ~/vac_logs/t3_grid.log 2>&1
      cp "$OUT/final_full.tsv" "$4/grid_${TAG}_final.tsv"
      echo "=== $5 $TAG done $(date '+%F %T')"
    done
  done
}

mkdir -p ~/vac_grid/willett ~/vac_grid/card
grid ~/tlv/trials ~/tlv/meta.tsv ~/vac_grid/willett "$WVAC"/ willett
grid ~/tlv_t15/trials ~/tlv_t15/meta.tsv ~/vac_grid/card "$CVAC"/ card
echo "T3 ALL DONE $(date '+%F %T')"
