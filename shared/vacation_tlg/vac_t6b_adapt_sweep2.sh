#!/bin/bash
# T6b: day-adaptation lambda sweep, TAKE 2. Take 1 used NB=0 which makes
# wenet_lattice emit an EMPTY hyp column (score -0.000) — invalid. Take 2
# decodes with NB=100 into fresh adapt2_l* dirs / dayadapt2_* repo copies.
set -u
REPO=../shared
WORDS=../20230831_willett/data/20230616_willett_BtoT24/languageModel_5gram_work_words.txt
CVAC=../20240815_card/data/20241009_card_BtoT25/vac
mkdir -p ~/vac_logs
for LAM in 0.25 0.5 0.75 1.0; do
  D=~/tlv_t15/trials_adapt_l$LAM
  OUT=~/tlv_t15/adapt2_l$LAM
  if [ -s "$OUT/final_full.tsv" ]; then echo "skip l$LAM"; continue; fi
  echo "=== l$LAM start $(date '+%F %T')"
  ACS=0.325 BP=90 NB=100 ~/src_lattice/wenet_sweep.sh ~/tg5/TLG.fst "$WORDS" \
    "$D" ~/tlv_t15/meta.tsv "$OUT" full 8 >> ~/vac_logs/t6b_adapt.log 2>&1 \
    || { echo "l$LAM decode FAILED"; continue; }
  cp "$OUT/final_full.tsv" "$CVAC/dayadapt2_l${LAM}_final.tsv"
  cp "$OUT/nbest_full.tsv" "$CVAC/dayadapt2_l${LAM}_nbest.tsv"
  echo "=== l$LAM done $(date '+%F %T')"
done
echo "T6B SWEEP DONE $(date '+%F %T')"
