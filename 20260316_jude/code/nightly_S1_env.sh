#!/bin/bash
# S1 夜间流水线：conda env + pip 依赖（2026-10-09 S1）
# 配方见仓库根 pyproject.toml [tool.reproduce.environment]
set -xe
CONDA='D:\apps\miniconda3\Scripts\conda.exe'
PY='/c/Users/user\.conda\envs\reproduce_20260316_jude\python.exe'
REQ='..\20260316_jude\open_source_code_for_reference\20260316_jude_typingBCI\requirements.txt'

"$CONDA" create -n reproduce_20260316_jude python=3.11 -y
"$PY" --version
"$PY" -m pip install -r "$REQ"
# 冒烟：关键包可导入
"$PY" -c "import tensorflow as tf; print('TF', tf.__version__)"
"$PY" -c "import scipy, numpy, sklearn, numba, pandas; print('scipy', scipy.__version__, '| numpy', numpy.__version__, '| sklearn', sklearn.__version__, '| numba', numba.__version__, '| pandas', pandas.__version__)"
echo "S1_DONE"
