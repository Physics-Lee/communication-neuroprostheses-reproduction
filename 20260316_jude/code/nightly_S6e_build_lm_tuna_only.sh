#!/bin/bash
# S6e: --override-channels 强制仅 TUNA（绕开 defaults 卡死源），编译线今夜最后一次干预
set -x
date
timeout 420 /root/miniconda3/bin/conda create -n py310lm python=3.10 -y --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ || { echo "S6E_FAILED_CONDA_CREATE"; exit 1; }
P=/root/miniconda3/envs/py310lm/bin/python
$P --version || { echo "S6E_FAILED_NO_PY"; exit 1; }
timeout 1800 $P -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 60 "torch==1.13.1" "numpy" 2>&1 | tail -5
$P -c "import torch; print(\"torch OK\", torch.__version__)" || { echo "S6E_FAILED_TORCH"; exit 1; }
cd ../20260316_jude/open_source_code_for_reference/speechBCI/LanguageModelDecoder/runtime/server/x86 || { echo "S6E_FAILED_NO_DIR"; exit 1; }
$P setup.py install 2>&1 | tail -40
$P -c "import lm_decoder; print(\"lm_decoder OK\")" && echo "S6E_DONE" || echo "S6E_FAILED_BUILD"
date
