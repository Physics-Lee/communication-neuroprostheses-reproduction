#!/bin/bash
# S3 夜间流水线：选择性解压 LanguageModels.zip（2026-10-09）
# 只取解码必需文件；G_no_prune.fst（24.5GB）绝对不解压——官方全部分析 rescore=False 用不到，
# 内存与磁盘预算都依赖跳过它（见 docs/20261009_复现计划.md §3/§6）。
set -xe
cd '..\20260316_jude'
unzip -q -o 'data/raw/20260316_jude/LanguageModels.zip' \
  'LanguageModels/Typing5gramPunc/TLG.fst' \
  'LanguageModels/Typing5gramPunc/G.fst' \
  'LanguageModels/Typing5gramPunc/words.txt' \
  'LanguageModels/Typing5gramPunc/lexicon_numbers.txt' \
  'LanguageModels/Typing5gramPunc/units.txt' \
  'LanguageModels/Typing5gramPunc/tokens.txt' \
  -d 'data/20260316_jude/'
ls -la 'data/20260316_jude/LanguageModels/Typing5gramPunc/'
echo "S3_DONE"
