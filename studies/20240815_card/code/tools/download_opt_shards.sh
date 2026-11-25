#!/bin/bash
# Parallel chunked download of facebook/opt-6.7b pytorch shards via hf-mirror.
# 2026-10-13: single-connection speed ~0.6 MB/s; 16 parallel range chunks
# give ~10x. Chunks are resumable (skipped when already the right size).
# Run from Git Bash: bash download_opt_shards.sh
set -u
DEST="../20260710_Design_of_Large_Corpus/LLM/opt-6.7b"
BASE="https://hf-mirror.com/facebook/opt-6.7b/resolve/main"
CHUNK=$((500 * 1024 * 1024))   # 500 MiB
MAXJOBS=16

download_shard() { # <fname> <total_size>
  local FNAME=$1 TOTAL=$2
  local NCHUNKS=$(( (TOTAL + CHUNK - 1) / CHUNK ))
  echo "== $FNAME: $TOTAL bytes in $NCHUNKS chunks"
  local pids=()
  for ((c = 0; c < NCHUNKS; c++)); do
    local S=$((c * CHUNK))
    local E=$((S + CHUNK - 1))
    [ $E -ge $TOTAL ] && E=$((TOTAL - 1))
    local WANT=$((E - S + 1))
    local PART="$DEST/.$FNAME.part$(printf %04d "$c")"
    (
      for try in 1 2 3 4 5; do
        local HAVE=0
        [ -f "$PART" ] && HAVE=$(stat -c %s "$PART")
        if [ "$HAVE" -eq "$WANT" ]; then exit 0; fi
        if [ "$HAVE" -gt 0 ] && [ "$HAVE" -lt "$WANT" ]; then
          # resume partial chunk: append the missing byte range
          curl -sL --fail -o - -r $((S + HAVE))-$E "$BASE/$FNAME" >> "$PART" && break
        else
          curl -sL --fail -o "$PART" -r "$S-$E" "$BASE/$FNAME" && break
        fi
        sleep 3
      done
      HAVE=0; [ -f "$PART" ] && HAVE=$(stat -c %s "$PART")
      [ "$HAVE" -eq "$WANT" ] || { echo "CHUNK FAIL $FNAME #$c"; exit 1; }
    ) &
    pids+=($!)
    while [ "$(jobs -rp | wc -l)" -ge "$MAXJOBS" ]; do sleep 2; done
  done
  local fail=0
  for p in "${pids[@]}"; do wait "$p" || fail=1; done
  [ $fail -eq 0 ] || { echo "ABORT $FNAME (chunk failures)"; return 1; }
  cat "$DEST/.$FNAME".part* > "$DEST/$FNAME"
  local GOT=$(stat -c %s "$DEST/$FNAME")
  if [ "$GOT" -eq "$TOTAL" ]; then
    rm -f "$DEST/.$FNAME".part*
    echo "== $FNAME OK ($GOT bytes)"
  else
    echo "== $FNAME SIZE MISMATCH: got $GOT want $TOTAL"; return 1
  fi
}

WHICH=${1:-all}
[ "$WHICH" = all ] || [ "$WHICH" = 1 ] && download_shard pytorch_model-00001-of-00002.bin 9960750957
[ "$WHICH" = all ] || [ "$WHICH" = 2 ] && download_shard pytorch_model-00002-of-00002.bin 3356360185
echo "ALL SHARDS DONE ($WHICH)"
