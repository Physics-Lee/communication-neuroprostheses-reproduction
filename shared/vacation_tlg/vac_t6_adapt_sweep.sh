#!/bin/bash
# T6: day-adaptation lambda sweep — decode corrected trials dirs (official
# params bp90/acs0.325, NB=0 for 1-best speed) and copy finals to the repo.
# Runs inside WSL after code/vac_day_adapt.py has written trials_adapt_l*.
set -u
REPO=../shared
WORDS=../20230831_willett/data/20230616_willett_BtoT24/languageModel_5gram_work_words.txt
CVAC=../20240815_card/data/20241009_card_BtoT25/vac
mkdir -p ~/vac_logs "$CVAC"
for LAM in 0.25 0.5 0.75 1.0; do
  D=~/tlv_t15/trials_adapt_l$LAM
  OUT=~/tlv_t15/adapt_l$LAM
  if [ -s "$OUT/final_full.tsv" ]; then echo "skip l$LAM"; continue; fi
  echo "=== l$LAM start $(date '+%F %T')"
  ACS=0.325 BP=90 NB=0 ~/src_lattice/wenet_sweep.sh ~/tg5/TLG.fst "$WORDS" \
    "$D" ~/tlv_t15/meta.tsv "$OUT" full 8 >> ~/vac_logs/t6_adapt.log 2>&1 \
    || { echo "l$LAM decode FAILED (see ~/vac_logs/t6_adapt.log)"; continue; }
  cp "$OUT/final_full.tsv" "$CVAC/dayadapt_l${LAM}_final.tsv"
  echo "=== l$LAM done $(date '+%F %T')"
done
echo "T6 SWEEP DONE $(date '+%F %T')"
