#!/bin/bash
# P4 endpoint decode: run_p4_20261013 logits -> npy -> wenet_lattice official
# params -> repo copy with p4_ prefix. Fires after dump_p4_logits.py completes.
set -eu
REPO=../20240815_card
P4=~/t15p4
WORDS=../20230831_willett/data/20230616_willett_BtoT24/languageModel_5gram_work_words.txt

python3 "$REPO/code/tools/export_npy_t15.py" \
  ../20240815_card/data/20241009_card_BtoT25/run_p4_20261013/t15_logits.hdf5 \
  "$P4/trials" "$P4/meta.tsv"

OUT="$P4/decode_bp90_a0325"
BP=90 ACS=0.325 NB=100 ~/src_lattice/wenet_sweep.sh ~/tg5/TLG.fst "$WORDS" \
  "$P4/trials" "$P4/meta.tsv" "$OUT" full 8

DEST=../20240815_card/data/20241009_card_BtoT25/run_p4_20261013
cp "$OUT/final_full.tsv" "$DEST/p4_final_full.tsv"
cp "$OUT/nbest_full.tsv" "$DEST/p4_nbest_full.tsv"
wc -l "$DEST"/p4_{final_full,nbest_full}.tsv
echo "P4 DECODE DONE"
