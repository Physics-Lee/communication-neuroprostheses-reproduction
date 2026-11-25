#!/bin/bash
# P2-lite runner (20261014 night): day-layer init ablation on top of the P4
# checkpoint. keep = P4's own trained day layer (the fair "layer 3 increment"
# test); eye = official offline-code identity init. LR 0.004, main config.
PY="../20260910_reproduce_20260629_levy/.venv312/Scripts/python.exe"
CODE="../20240815_card/code/calib_online_sim.py"
LOGS="../20240815_card/result/logs"

run () {
  local sess=$1 lr=$2; shift 2
  echo "=== [$(date +%H:%M:%S)] $sess lr=$lr $* ==="
  "$PY" "$CODE" "$sess" "$lr" "$@" \
    > "$LOGS/calib_20261014_${sess##*.}_lr${lr}_$(echo "$*" | tr -cd 'a-z0-9').log" 2>&1 \
    || echo "!!! FAILED: $sess lr=$lr $* (see log)"
}

# keep arm: layer-3 increment over the static model, all six 2025 days
run t15.2025.01.10 0.004 --arm keep
run t15.2025.03.14 0.004 --arm keep
run t15.2025.04.13 0.004 --arm keep
run t15.2025.01.12 0.004 --arm keep
run t15.2025.03.16 0.004 --arm keep
run t15.2025.03.30 0.004 --arm keep
# eye arm: official offline-code new-day mechanism, disaster days
run t15.2025.01.10 0.004 --arm eye
run t15.2025.03.14 0.004 --arm eye
run t15.2025.04.13 0.004 --arm eye
echo "=== [$(date +%H:%M:%S)] P2-LITE DONE ==="
