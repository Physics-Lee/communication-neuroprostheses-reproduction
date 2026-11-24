#!/bin/bash
# P1 overnight runner (20261014): S2.03 sentence-by-sentence fine-tune sim.
# Order: 3 disaster days @ LR 0.004 (main arm) -> same 3 @ 0.04 (text-value
# contrast) -> remaining 3 disaster-ish days @ 0.004 -> 2 good-day controls
# @ 0.004. One GPU, sequential; single-run failures do not stop the queue.
PY="../20260910_reproduce_20260629_levy/.venv312/Scripts/python.exe"
CODE="../20240815_card/code/calib_online_sim.py"
LOGS="../20240815_card/result/logs"

run () {  # run <session> <lr> [extra args...]
  local sess=$1 lr=$2; shift 2
  local tag="$(echo "$sess_lr_$lr$*" | tr -cd 'a-zA-Z0-9._-' | tail -c 40)"
  echo "=== [$(date +%H:%M:%S)] $sess lr=$lr $* ==="
  "$PY" "$CODE" "$sess" "$lr" "$@" \
    > "$LOGS/calib_20261014_${sess##*.}_lr${lr}_$(echo "$*" | tr -cd 'a-z0-9').log" 2>&1 \
    || echo "!!! FAILED: $sess lr=$lr $* (see log)"
}

# main arm: three disaster days
run t15.2025.01.10 0.004
run t15.2025.03.14 0.004
run t15.2025.04.13 0.004
# contrast arm: Table-S5-vs-text LR ambiguity
run t15.2025.01.10 0.04
run t15.2025.03.14 0.04
run t15.2025.04.13 0.04
# remaining 2025 days, main arm
run t15.2025.01.12 0.004
run t15.2025.03.16 0.004
run t15.2025.03.30 0.004
# good-day controls, main arm
run t15.2023.08.18 0.004
run t15.2023.11.04 0.004
echo "=== [$(date +%H:%M:%S)] ALL RUNS DONE ==="
