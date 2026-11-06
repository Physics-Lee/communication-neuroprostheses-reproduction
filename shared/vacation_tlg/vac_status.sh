#!/bin/bash
# Vacation experiment status: running jobs, GPU, recent outputs, disk, logs.
# Run from repo root in Git Bash: bash code/vac_status.sh
set -u
cd "$(dirname "$0")/.."
echo "===== now: $(date '+%F %T') ====="

echo "--- WSL decode processes ---"
wsl.exe bash -lc 'ps aux | grep -E "wenet_lattice|tlg_viterbi" | grep -v grep' 2>/dev/null || echo "(none / wsl down)"

echo "--- WSL latest vac/sweep outputs ---"
wsl.exe bash -lc 'ls -lt ~/vac_grid/*/*/final_full.tsv ~/*/sw_*/final_full.tsv 2>/dev/null | head -8' 2>/dev/null

echo "--- GPU (Windows nvidia-smi) ---"
NSMI="/c/Windows/System32/nvidia-smi.exe"
if [ -f "$NSMI" ]; then
  "$NSMI" --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null
  "$NSMI" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null
else
  wsl.exe bash -lc 'nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader' 2>/dev/null
fi

echo "--- python.exe processes ---"
tasklist 2>/dev/null | grep -i "python" | head -5 || echo "(none)"

echo "--- repo vac outputs (Windows side) ---"
ls -lt data/*/vac/*.tsv 2>/dev/null | head -12 || echo "(none yet)"

echo "--- progress logs (last 4 lines each) ---"
for f in code/vac_t2_qwen3_progress.log result/logs/vac_*.log; do
  [ -f "$f" ] && echo "[$f]" && tail -4 "$f"
done

echo "--- WSL logs (last 3 lines t1/t3) ---"
wsl.exe bash -lc 'for f in ~/vac_logs/*.log; do [ -f "$f" ] && echo "[$f]" && tail -3 "$f"; done' 2>/dev/null

echo "--- disk ---"
wsl.exe bash -lc 'df -h / /mnt/d | tail -2' 2>/dev/null
echo "D: $(df -h /d | tail -1 | awk '{print $4}') free (Git Bash view)"
