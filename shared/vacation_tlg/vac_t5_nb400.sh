#!/bin/bash
# T5: NB=400 deep n-best decodes (official params bp90/acs0.325) for
# willett/card/jude + oracle-depth curves on old (NB=100) and new pools.
# Run inside WSL: bash ../shared/code/vac_t5_nb400.sh
set -u
REPO=../shared
WORDS=../20230831_willett/data/20230616_willett_BtoT24/languageModel_5gram_work_words.txt
mkdir -p ~/vac_logs ~/vac_grid/nb400 \
  "../20240815_card/data/20241009_card_BtoT25/vac" "../20260428_jude/data/20260428_jude/vac"

dec() {  # <trials> <meta> <tag> <copydir>
  OUT=~/vac_grid/nb400/$3
  if [ -s "$OUT/nbest_full.tsv" ]; then echo "skip $3 (exists)"; return; fi
  echo "=== $3 start $(date '+%F %T')"
  ACS=0.325 BP=90 NB=400 ~/src_lattice/wenet_sweep.sh ~/tg5/TLG.fst "$WORDS" \
    "$1" "$2" "$OUT" full 8 >> ~/vac_logs/t5_nb400.log 2>&1
  cp "$OUT/final_full.tsv" "$4/nbest400_final.tsv"
  cp "$OUT/nbest_full.tsv" "$4/nbest400.tsv"
  echo "=== $3 done $(date '+%F %T')"
}

dec ~/tlv/trials    ~/tlv/meta.tsv    willett "../20230831_willett/data/20230616_willett_BtoT24/vac"
dec ~/tlv_t15/trials ~/tlv_t15/meta.tsv card   "../20240815_card/data/20241009_card_BtoT25/vac"
dec ~/tlv_t17/trials ~/tlv_t17/meta.tsv jude   "../20260428_jude/data/20260428_jude/vac"

echo "--- oracle depth curves (NB=400) ---"
W=../20230831_willett/data/20230616_willett_BtoT24/vac; C=../20240815_card/data/20241009_card_BtoT25/vac; J=../20260428_jude/data/20260428_jude/vac
python3 $REPO/code/vac_oracle_depth.py "$W/nbest400.tsv" "$W/nbest400_final.tsv" 1,5,10,25,50,100,200,400
python3 $REPO/code/vac_oracle_depth.py "$C/nbest400.tsv" "$C/nbest400_final.tsv" 1,5,10,25,50,100,200,400
python3 $REPO/code/vac_oracle_depth.py "$J/nbest400.tsv" "$J/nbest400_final.tsv" 1,5,10,25,50,100,200,400
echo "--- oracle depth curves (existing NB=100 baseline pools) ---"
python3 $REPO/code/vac_oracle_depth.py "../20230831_willett/data/20230616_willett_BtoT24/run_60k/wl_nbest_full.tsv" "../20230831_willett/data/20230616_willett_BtoT24/run_60k/wl_final_full.tsv" 1,10,25,50,100
python3 $REPO/code/vac_oracle_depth.py "../20240815_card/data/20241009_card_BtoT25/run_120k/nbest_full.tsv" "../20240815_card/data/20241009_card_BtoT25/run_120k/final_full.tsv" 1,10,25,50,100
python3 $REPO/code/vac_oracle_depth.py "../20260428_jude/data/20260428_jude/run_25k/off_nbest_full.tsv" "../20260428_jude/data/20260428_jude/run_25k/off_final_full.tsv" 1,10,25,50,100
echo "T5 ALL DONE $(date '+%F %T')"
