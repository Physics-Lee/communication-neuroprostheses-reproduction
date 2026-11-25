#!/bin/bash
# Decode biGRU-48k corrected-index logits with official params.
set -eu
REPO=../20240815_card
W=~/t15bigru
WORDS=../20230831_willett/data/20230616_willett_BtoT24/languageModel_5gram_work_words.txt
mkdir -p "$W"
python3 "$REPO/code/tools/export_npy_t15.py" \
  "$REPO/data/20241009_card_BtoT25/run_bigru48k_20261017/t15_logits_fixed.hdf5" \
  "$W/trials" "$W/meta.tsv"
OUT="$W/decode_bp90_a0325"
rm -rf "$OUT"
BP=90 ACS=0.325 NB=100 ~/src_lattice/wenet_sweep.sh ~/tg5/TLG.fst "$WORDS" \
  "$W/trials" "$W/meta.tsv" "$OUT" full 8
DEST=../20240815_card/data/20241009_card_BtoT25/run_bigru48k_20261017
cp "$OUT/final_full.tsv" "$DEST/bigru_final_full.tsv"
cp "$OUT/nbest_full.tsv" "$DEST/bigru_nbest_full.tsv"
wc -l "$DEST"/bigru_{final,nbest}_full.tsv
echo "BIGRU DECODE DONE"
