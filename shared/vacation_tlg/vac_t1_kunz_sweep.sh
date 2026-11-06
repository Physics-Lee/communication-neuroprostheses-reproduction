#!/bin/bash
# T1: Kunz T12/T15 official-decoder bp x acs sweep (mirrors the T16 grid).
# Run inside WSL: bash ../shared/code/vac_t1_kunz_sweep.sh
# Skips cells whose final_full.tsv already exists. Copies results to the repo
# (data/20250821_kunz/vac/) for Windows-side WER evaluation.
set -u
REPO=../shared
WORDS=../20230831_willett/data/20230616_willett_BtoT24/languageModel_5gram_work_words.txt
VAC=../20250821_kunz/data/20250821_kunz/vac
mkdir -p ~/vac_logs "$VAC"
for ARM in t12 t15; do
  D=~/tlv_kunz_${ARM}_mixed
  for BP in 1 9 90; do
    for ACS in 0.325 0.5; do
      TAG=bp${BP}_a${ACS}
      OUT=$D/sw_$TAG
      if [ -s "$OUT/final_full.tsv" ]; then
        echo "skip $ARM $TAG (exists) $(date '+%F %T')"
        continue
      fi
      echo "=== $ARM $TAG start $(date '+%F %T')"
      BP=$BP ACS=$ACS NB=100 ~/src_lattice/wenet_sweep.sh ~/tg5/TLG.fst "$WORDS" \
        "$D/trials" "$D/meta.tsv" "$OUT" full 4 >> ~/vac_logs/t1_kunz.log 2>&1
      cp "$OUT/final_full.tsv" "$VAC/${ARM}_${TAG}_final.tsv"
      cp "$OUT/nbest_full.tsv" "$VAC/${ARM}_${TAG}_nbest.tsv"
      echo "=== $ARM $TAG done $(date '+%F %T')"
    done
  done
done
echo "T1 ALL DONE $(date '+%F %T')"
