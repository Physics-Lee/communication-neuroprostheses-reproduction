#!/bin/bash
# T17 修复链：04.25（block 6 数据中不存在→改 [1,3,4,5]）+ 07.22（耐心 3000 重训）+ 事后 LM 解码（2026-10-12 14:40）
set -x
cd /d/repositories/20260316_jude
TFPY='/c/Users/user\.conda\envs\reproduce_20260316_jude\python.exe'
LEVY='..\20260910_reproduce_20260629_levy\.venv312\Scripts\python.exe'

echo "===== FIX 04.25 START $(date) ====="
"$TFPY" code/window_W4_convert.py t17.2024.04.25 1,3,4,5 20 "" || { echo "CONVERT_FAIL_0425"; exit 1; }
"$TFPY" code/pytorch_port/export_tfrecord_to_npz.py t17.2024.04.25 || { echo "EXPORT_FAIL_0425"; exit 1; }
NPZ1='..\20260316_jude\data\20260316_jude\Data\t17\t17.2024.04.25\Typing\m1_torch_data.npz'
RUN1='..\20260316_jude\result\runs\m3_t1720240425_torch'
"$LEVY" code/pytorch_port/train.py --batches 10000 --npz "$NPZ1" --out "$RUN1" || echo "TRAIN_FAIL_0425"
"$LEVY" code/pytorch_port/dump_logits.py "$RUN1" "$NPZ1" || echo "DUMP_FAIL_0425"
echo "===== FIX 04.25 DONE $(date) ====="

echo "===== RETRAIN 07.22 (patience 3000) START $(date) ====="
NPZ2='..\20260316_jude\data\20260316_jude\Data\t17\t17.2024.07.22\Typing\m1_torch_data.npz'
RUN2='..\20260316_jude\result\runs\m3_t1720240722_torch'
"$LEVY" code/pytorch_port/train.py --batches 10000 --patience 3000 --npz "$NPZ2" --out "$RUN2" || echo "TRAIN_FAIL_0722"
"$LEVY" code/pytorch_port/dump_logits.py "$RUN2" "$NPZ2" || echo "DUMP_FAIL_0722"
echo "===== RETRAIN 07.22 DONE $(date) ====="

echo "===== LM DECODE (5 会话) START $(date) ====="
cat > code/pytorch_port/s9_runs.json <<'EOF'
{
 "t17.2024.04.25": "../20260316_jude/result/runs/m3_t1720240425_torch",
 "t17.2024.06.05": "../20260316_jude/result/runs/m3_t1720240605_torch",
 "t17.2024.06.13": "../20260316_jude/result/runs/m3_t1720240613_torch",
 "t17.2024.07.10": "../20260316_jude/result/runs/m3_t1720240710_torch",
 "t17.2024.07.22": "../20260316_jude/result/runs/m3_t1720240722_torch"
}
EOF
wsl -u root bash -c "tr -d '\r' < ../20260316_jude/code/pytorch_port/s9a_lm_decode_multi.py > /tmp/s9n.py && printf '#!/bin/bash\n/root/miniconda3/envs/py310lm/bin/python /tmp/s9n.py\n' > /tmp/s9n.sh"
wsl -u root bash -c "bash /tmp/s9n.sh"
echo "===== FIX_CHAIN_ALL_DONE $(date) ====="
