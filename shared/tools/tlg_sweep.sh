#!/bin/bash
# Parallel TLG decode sweep: splits a meta file across N workers (shared page
# cache makes workers nearly free) and concatenates outputs.
# usage: tlg_sweep.sh <TLG.fst> <words.txt> <trials_dir> <meta> <outdir> <nworkers> <beam> <acscale...>
set -u
TLG=$1; WORDS=$2; TRIALS=$3; META=$4; OUTDIR=$5; N=$6; BEAM=$7; shift 7
mkdir -p "$OUTDIR"
split -n l/$N -d "$META" "$OUTDIR/chunk_"
for ACS in "$@"; do
  pids=()
  for i in $(seq 0 $((N-1))); do
    i=$(printf '%02d' $i)
    c="$OUTDIR/chunk_$i"
    [ -f "$c" ] || continue
    NB=${NBEST:-0}
    NBP=${NBEST_PATH:-}
    BP=${BLANK_PENALTY:-1.0}
    if [ "$NB" -gt 0 ] && [ -n "$NBP" ]; then
      ~/bin/tlg_viterbi "$TLG" "$WORDS" "$TRIALS" "$c" "$OUTDIR/part_${ACS}_$i.tsv" "$BEAM" "$ACS" 100000 1 0 "$NB" "${NBP}_$i" "$BP" &
    else
      ~/bin/tlg_viterbi "$TLG" "$WORDS" "$TRIALS" "$c" "$OUTDIR/part_${ACS}_$i.tsv" "$BEAM" "$ACS" 100000 1 0 "$BP" &
    fi
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait $p; done
  cat "$OUTDIR"/part_${ACS}_*.tsv | grep -v '^key' > "$OUTDIR/all_${ACS}.tsv.tmp"
  (head -1 "$OUTDIR"/part_${ACS}_0.tsv; cat "$OUTDIR"/all_${ACS}.tsv.tmp) > "$OUTDIR/all_${ACS}.tsv"
  if [ "$NB" -gt 0 ] && [ -n "$NBP" ]; then
    cat "$OUTDIR"/nbest_${ACS}_*.tsv > "${NBP}"
  fi
  echo "=== acscale=$ACS done: $(wc -l < "$OUTDIR/all_${ACS}.tsv") lines"
done
