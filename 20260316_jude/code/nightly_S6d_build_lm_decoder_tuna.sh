#!/bin/bash
# S6d 夜间流水线：换清华 TUNA 镜像重试 lm_decoder 编译（2026-10-09，编译线最后一次自动尝试）
# 背景：S6c 的 conda create 卡在 repo.anaconda.com 元数据 52 分钟（CPU 0:00，网络性卡死；github/apt 均正常）。
# 修复：conda 用 TUNA 镜像、pip 用 TUNA pypi；torch 用 PyPI 版 1.13.1（linux 轮子自带依赖，CPU 使用即可编译）。
set -x
date

echo "=== [1/4] 配置 conda TUNA 镜像 ==="
/root/miniconda3/bin/conda config --remove-key channels 2>/dev/null
/root/miniconda3/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
/root/miniconda3/bin/conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
/root/miniconda3/bin/conda config --set show_channel_urls yes
/root/miniconda3/bin/conda config --set remote_connect_timeout_secs 30
/root/miniconda3/bin/conda config --set remote_read_timeout_secs 60
timeout 600 /root/miniconda3/bin/conda create -n py310lm python=3.10 -y || { echo "S6D_FAILED_CONDA_CREATE"; exit 1; }
P=/root/miniconda3/envs/py310lm/bin/python
$P --version || { echo "S6D_FAILED_NO_PY"; exit 1; }

echo "=== [2/4] TUNA pypi 装 torch 1.13.1 + numpy ==="
timeout 1800 $P -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 60 'torch==1.13.1' 'numpy' 2>&1 | tail -5
$P -c "import torch; print('torch', torch.__version__)" || { echo "S6D_FAILED_TORCH"; exit 1; }

echo "=== [3/4] 编译 lm_decoder ==="
cd ../20260316_jude/open_source_code_for_reference/speechBCI/LanguageModelDecoder/runtime/server/x86 || { echo "S6D_FAILED_NO_DIR"; exit 1; }
$P setup.py install 2>&1 | tail -40

echo "=== [4/4] 验证 ==="
$P -c "import lm_decoder; print('lm_decoder import OK')" && echo "S6D_DONE" || echo "S6D_FAILED_BUILD"
date
