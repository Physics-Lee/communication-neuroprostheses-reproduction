#!/bin/bash
# P2 decode reproduction (2026-10-13): wenet_lattice official params on the
# regenerated ~/t15r/trials (byte-verified against archive ~/tlv_t15/trials).
# Outputs -> ~/t15r/decode_bp90_a0325, then copied into the repo with repro_
# prefix (archive originals in run_120k/ are never overwritten).
set -eu
WORDS=../20230831_willett/data/20230616_willett_BtoT24/languageModel_5gram_work_words.txt
OUT=~/t15r/decode_bp90_a0325
REPO=../20240815_card
BP=90 ACS=0.325 NB=100 ~/src_lattice/wenet_sweep.sh ~/tg5/TLG.fst "$WORDS" \
  ~/t15r/trials ~/t15r/meta.tsv "$OUT" full 8
mkdir -p "$REPO/data/20241009_card_BtoT25/run_120k"
cp "$OUT/final_full.tsv" "$REPO/data/20241009_card_BtoT25/run_120k/repro_20261013_final_full.tsv"
cp "$OUT/nbest_full.tsv" "$REPO/data/20241009_card_BtoT25/run_120k/repro_20261013_nbest_full.tsv"
wc -l "$REPO"/data/20241009_card_BtoT25/run_120k/repro_20261013_{final_full,nbest_full}.tsv
echo "P2 DECODE DONE"
