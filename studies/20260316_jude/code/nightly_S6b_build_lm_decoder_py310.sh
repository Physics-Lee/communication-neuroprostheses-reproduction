#!/bin/bash
# S6b 夜间流水线：WSL 内用 Miniconda py3.10 环境重试编译 lm_decoder（2026-10-09）
# 背景：S6 失败——WSL 系统 Python 是 3.14，torch 1.13.1 无 cp314 轮子 → torch 装不上 → CMake 失败。
# 对策：WSL 内装 Miniconda（root home，不碰仓库盘），建 python=3.10 env 编译（官方平台即 py3.10 系）。
set -x
date

echo "=== [1/4] WSL 内装 Miniconda ==="
if [ ! -x /root/miniconda3/bin/conda ]; then
  curl -sL -o /tmp/mc.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh || { echo "S6B_FAILED_DOWNLOAD"; exit 1; }
  bash /tmp/mc.sh -b -p /root/miniconda3 || { echo "S6B_FAILED_INSTALL"; exit 1; }
fi
/root/miniconda3/bin/conda --version

echo "=== [2/4] 建 py310lm env ==="
/root/miniconda3/bin/conda create -n py310lm python=3.10 -y || { echo "S6B_FAILED_CONDA_CREATE"; exit 1; }
P=/root/miniconda3/envs/py310lm/bin/python
$P --version

echo "=== [3/4] 装 libtorch 1.13.1 CPU ==="
$P -m pip install 'torch==1.13.1+cpu' --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -3
$P -c "import torch; print('torch', torch.__version__)" || { echo "S6B_FAILED_TORCH"; exit 1; }

echo "=== [4/4] 编译 lm_decoder ==="
cd ../20260316_jude/open_source_code_for_reference/speechBCI/LanguageModelDecoder/runtime/server/x86 || { echo "S6B_FAILED_NO_DIR"; exit 1; }
$P setup.py install 2>&1 | tail -40
$P -c "import lm_decoder; print('lm_decoder import OK')" && echo "S6B_DONE" || echo "S6B_FAILED_BUILD"
date
