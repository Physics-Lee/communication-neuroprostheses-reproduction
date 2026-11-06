#!/bin/bash
# Decode the CORRECTED-index logits (t15_logits_fixed.hdf5) with official
# params. Mirrors p4_decode_20261013.sh; outputs land back in run_p4 as
# p4_fixed_{final,nbest}_full.tsv.
set -eu
REPO=../20240815_card
P4=~/t15p4fix
WORDS=../20230831_willett/data/20230616_willett_BtoT24/languageModel_5gram_work_words.txt

mkdir -p "$P4"
python3 "$REPO/code/tools/export_npy_t15.py" \
  "$REPO/data/20241009_card_BtoT25/run_p4_20261013/t15_logits_fixed.hdf5" \
  "$P4/trials" "$P4/meta.tsv"

OUT="$P4/decode_bp90_a0325"
rm -rf "$OUT"
BP=90 ACS=0.325 NB=100 ~/src_lattice/wenet_sweep.sh ~/tg5/TLG.fst "$WORDS" \
  "$P4/trials" "$P4/meta.tsv" "$OUT" full 8

DEST=../20240815_card/data/20241009_card_BtoT25/run_p4_20261013
cp "$OUT/final_full.tsv" "$DEST/p4_fixed_final_full.tsv"
cp "$OUT/nbest_full.tsv" "$DEST/p4_fixed_nbest_full.tsv"
wc -l "$DEST"/p4_fixed_{final,nbest}_full.tsv
echo "FIXED-INDEX DECODE DONE"
