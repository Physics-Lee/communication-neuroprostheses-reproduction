#!/bin/bash
# S6 夜间流水线：WSL2 内准备并编译 lm_decoder（2026-10-09）
# 来源：fwillett/speechBCI LanguageModelDecoder 第三步；依赖 libtorch 1.13.1、gcc>=10、cmake>=3.14
# 本步最易失败：任何一步失败都记录并继续到最终标记，错误全文留人工，不阻塞主线。
# 注意：本脚本在 WSL 内以 root 运行（wsl -u root bash /mnt/d/.../code/nightly_S6_build_lm_decoder.sh）
set -x
date

echo "=== [1/5] WSL 基础信息 ==="
uname -a || { echo "S6_FAILED_WSL_UNAVAILABLE"; exit 1; }
cat /etc/os-release | head -2
gcc --version 2>/dev/null | head -1 || echo "gcc 未装（下一步装）"

echo "=== [2/5] 安装编译工具链（root 直装免 sudo 交互） ==="
apt-get update -qq 2>&1 | tail -2
apt-get install -y -qq build-essential cmake git python3-pip 2>&1 | tail -3
gcc --version | head -1
cmake --version | head -1

echo "=== [3/5] 克隆 speechBCI ==="
mkdir -p ../20260316_jude/open_source_code_for_reference
cd ../20260316_jude/open_source_code_for_reference
if [ ! -d speechBCI/.git ]; then
  git clone --depth 1 https://github.com/fwillett/speechBCI 2>&1 | tail -2
else
  echo "speechBCI 已存在，跳过克隆"
fi

echo "=== [4/5] 安装 libtorch 1.13.1 CPU（用户级） ==="
python3 --version
pip3 install --user 'torch==1.13.1+cpu' --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -3
python3 -c "import torch; print('torch', torch.__version__)" || echo "S6_FAILED_TORCH"

echo "=== [5/5] 编译 lm_decoder（耗时长步骤） ==="
cd speechBCI/LanguageModelDecoder/runtime/server/x86 || { echo "S6_FAILED_NO_DIR"; exit 1; }
python3 setup.py install 2>&1 | tail -40
python3 -c "import lm_decoder; print('lm_decoder import OK')" && echo "S6_DONE" || echo "S6_FAILED_BUILD"

date
