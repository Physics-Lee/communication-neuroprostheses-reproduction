#!/bin/bash
# S2 夜间流水线：解压 Data.zip（排除 *.png 截图，省约 10GB；分析只读 .mat）（2026-10-09 S2）
set -xe
cd '..\20260316_jude'
mkdir -p data/20260316_jude
unzip -q -o 'data/raw/20260316_jude/Data.zip' -x '*.png' -d 'data/20260316_jude/'
# 对账：解出的 .mat 数
echo "mat files: $(find data/20260316_jude/Data -name '*.mat' | wc -l)"
echo "S2_DONE"
