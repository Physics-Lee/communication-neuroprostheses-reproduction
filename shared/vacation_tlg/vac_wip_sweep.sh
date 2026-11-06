#!/bin/bash
# WIP sweep wrapper: same as ~/src_lattice/wenet_sweep.sh but calls
# wenet_lattice_wip and passes $WIP as argv[16] (after build=0 skip=1.0).
# usage: WIP=<v> [ACS BP NB] wenet_sweep_wip.sh <TLG> <words> <trials> <meta> <outdir> <tag> <nworkers>
set -u
TLG=$1; WORDS=$2; TRIALS=$3; META=$4; OUTDIR=$5; TAG=$6; N=${7:-8}
ACS=${ACS:-0.325}; BP=${BP:-90}; BEAM=${BEAM:-17}; LATBEAM=${LATBEAM:-8}
MAXACT=${MAXACT:-7000}; MINACT=${MINACT:-200}; NB=${NB:-100}; WIP=${WIP:-0}
mkdir -p "$OUTDIR"
TOT=$(wc -l < "$META")
STEP=$(( (TOT + N - 1) / N ))
for ((i=0; i<N; i++)); do
  S=$((i*STEP+1)); E=$(((i+1)*STEP))
  printf -v P02 "%02d" $i
  sed -n "${S},${E}p" "$META" > "$OUTDIR/meta_${TAG}_${P02}.tsv"
  if [ -s "$OUTDIR/meta_${TAG}_${P02}.tsv" ]; then
    ~/bin/wenet_lattice_wip "$TLG" "$WORDS" "$TRIALS" "$OUTDIR/meta_${TAG}_${P02}.tsv" \
      "$OUTDIR/out_${TAG}_${P02}.tsv" "$ACS" "$BP" "$BEAM" "$LATBEAM" "$MAXACT" "$MINACT" \
      "$NB" "$OUTDIR/nb_${TAG}_${P02}" 0 1.0 "$WIP" > "$OUTDIR/log_${TAG}_${P02}.txt" 2>&1 &
  fi
done
wait
grep -h -v "^key" "$OUTDIR"/out_"${TAG}"_*.tsv > "$OUTDIR/body.tsv"
{ printf "key\tref\thyp\tscore\n"; cat "$OUTDIR/body.tsv"; } > "$OUTDIR/final_${TAG}.tsv"
cat "$OUTDIR"/nb_"${TAG}"_*.tsv > "$OUTDIR/nbest_${TAG}.tsv"
rm -f "$OUTDIR"/out_"${TAG}"_*.tsv "$OUTDIR"/meta_"${TAG}"_*.tsv "$OUTDIR"/body.tsv
echo "done: $TOT trials wip=$WIP -> $OUTDIR/final_${TAG}.tsv"
