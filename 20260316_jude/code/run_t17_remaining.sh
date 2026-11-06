#!/bin/bash
# T17 剩余 5 会话补全链：转换→npz→训练→dump logits（逐会话串行），最后一次性 5-gram 解码（2026-10-12 09:15）
# 官方 SessionArgs 原样照抄（block/trials_to_remove）；num_test 按 trial 量适配（>50 用 20，少则 ~25%）。
set -x
cd /d/repositories/20260316_jude
TFPY='/c/Users/user\.conda\envs\reproduce_20260316_jude\python.exe'
LEVY='..\20260910_reproduce_20260629_levy\.venv312\Scripts\python.exe'

declare -A BLOCKS TEST RM OUT
# 官方：04.25=[1,3,4,5,6]{}; 06.05=[1,2,6]{}; 06.13=[7,8]{7:[0,1,2]}; 07.10=[2]{}; 07.22=[1,2,3]{1:[18]}
SESSIONS="t17.2024.04.25 t17.2024.06.05 t17.2024.06.13 t17.2024.07.10 t17.2024.07.22"
BLOCKS[t17.2024.04.25]="1,3,4,5,6";  RM[t17.2024.04.25]="";            TEST[t17.2024.04.25]=20
BLOCKS[t17.2024.06.05]="1,2,6";      RM[t17.2024.06.05]="";            TEST[t17.2024.06.05]=20
BLOCKS[t17.2024.06.13]="7,8";        RM[t17.2024.06.13]="7:0,1,2";     TEST[t17.2024.06.13]=10
BLOCKS[t17.2024.07.10]="2";          RM[t17.2024.07.10]="";            TEST[t17.2024.07.10]=4
BLOCKS[t17.2024.07.22]="1,2,3";      RM[t17.2024.07.22]="1:18";        TEST[t17.2024.07.22]=20

for S in $SESSIONS; do
  TAG=$(echo "$S" | tr -d .)
  echo "===== SESSION $S START $(date) ====="
  "$TFPY" code/window_W4_convert.py "$S" "${BLOCKS[$S]}" "${TEST[$S]}" "${RM[$S]}" || { echo "CONVERT_FAIL_$S"; continue; }
  "$TFPY" code/pytorch_port/export_tfrecord_to_npz.py "$S" || { echo "EXPORT_FAIL_$S"; continue; }
  NPZ="D:\\repositories\\20260316_jude\\data\\20260316_jude\\Data\\t17\\$S\\Typing\\m1_torch_data.npz"
  RUN="D:\\repositories\\20260316_jude\\result\\runs\\m3_${TAG}_torch"
  "$LEVY" code/pytorch_port/train.py --batches 10000 --npz "$NPZ" --out "$RUN" || { echo "TRAIN_FAIL_$S"; continue; }
  "$LEVY" code/pytorch_port/dump_logits.py "$RUN" "$NPZ" || echo "DUMP_FAIL_$S"
  echo "===== SESSION $S DONE $(date) ====="
done

# 最后：一次性 5-gram 解码 5 个新会话
echo "===== LM DECODE START $(date) ====="
cat > code/pytorch_port/s9_runs.json <<'EOF'
{
 "t17.2024.04.25": "../20260316_jude/result/runs/m3_t1720240425_torch",
 "t17.2024.06.05": "../20260316_jude/result/runs/m3_t1720240605_torch",
 "t17.2024.06.13": "../20260316_jude/result/runs/m3_t1720240613_torch",
 "t17.2024.07.10": "../20260316_jude/result/runs/m3_t1720240710_torch",
 "t17.2024.07.22": "../20260316_jude/result/runs/m3_t1720240722_torch"
}
EOF
wsl -u root bash -c "tr -d '\r' < ../20260316_jude/code/pytorch_port/s9a_lm_decode_multi.py > /tmp/s9n.py && printf '#!/bin/bash\n/root/miniconda3/envs/py310lm/bin/python /tmp/s9n.py\n' > /tmp/s9n.sh && /root/miniconda3/envs/py310lm/bin/python -c 'import py_compile; py_compile.compile(\"/tmp/s9n.py\", doraise=True); print(\"COMPILE_OK\")'"
wsl -u root bash -c "bash /tmp/s9n.sh"
echo "===== CHAIN_ALL_DONE $(date) ====="
