#!/bin/bash
# Decode the ensemble (P4 x T2a averaged log-probs) with official params.
set -eu
REPO=../20240815_card
ENS=~/t15ens
WORDS=../20230831_willett/data/20230616_willett_BtoT24/languageModel_5gram_work_words.txt

mkdir -p "$ENS"
python3 "$REPO/code/tools/export_npy_t15.py" \
  "$REPO/data/20241009_card_BtoT25/run_ens_p4_t2a/t15_logits_fixed.hdf5" \
  "$ENS/trials" "$ENS/meta.tsv"

OUT="$ENS/decode_bp90_a0325"
rm -rf "$OUT"
BP=90 ACS=0.325 NB=100 ~/src_lattice/wenet_sweep.sh ~/tg5/TLG.fst "$WORDS" \
  "$ENS/trials" "$ENS/meta.tsv" "$OUT" full 8

DEST=../20240815_card/data/20241009_card_BtoT25/run_ens_p4_t2a
cp "$OUT/final_full.tsv" "$DEST/ens_final_full.tsv"
cp "$OUT/nbest_full.tsv" "$DEST/ens_nbest_full.tsv"
wc -l "$DEST"/ens_{final,nbest}_full.tsv
echo "ENSEMBLE DECODE DONE"
