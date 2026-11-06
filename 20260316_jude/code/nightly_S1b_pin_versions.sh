#!/bin/bash
# S1b 夜间流水线：把 requirements.txt 未锁定的包对齐到官方 README 测试版本（2026-10-09）
# 依据：官方 README "tested with the following python package versions"；requirements.txt 只锁 numpy/tensorflow，
# pip 默认装了 2026 最新版（pandas 3.0.6/scipy 1.16.3/numba 0.68 等），与官方 2024 时代代码有破坏性差异风险。
set -xe
PY='/c/Users/user\.conda\envs\reproduce_20260316_jude\python.exe'
"$PY" -m pip install \
  'scikit-learn==1.2.2' \
  'scipy==1.11.3' \
  'pandas==2.2.2' \
  'numba==0.58.1' \
  'matplotlib==3.8.4' \
  'omegaconf==2.3.0' \
  'pyyaml==6.0.1' \
  'redis==5.0.1'
"$PY" -c "import tensorflow as tf; import scipy, numpy, sklearn, numba, pandas, matplotlib, omegaconf, yaml, redis; print('TF', tf.__version__, '| scipy', scipy.__version__, '| numpy', numpy.__version__, '| sklearn', sklearn.__version__, '| numba', numba.__version__, '| pandas', pandas.__version__, '| mpl', matplotlib.__version__, '| omegaconf', omegaconf.__version__, '| pyyaml', yaml.__version__, '| redis', redis.__version__)"
echo "S1B_DONE"
