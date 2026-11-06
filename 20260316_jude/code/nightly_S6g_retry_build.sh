#!/bin/bash
# S6g 夜间流水线：S6f 网络僵死后重试 lm_decoder 编译（2026-10-09 22:05）
# gflags/glog 源码已在 fc_base/ 缓存（FetchContent 会跳过下载直接编译）。
# 教训固化：不用 tail 截断日志；退出码无条件写日志。
set -x
date
export CMAKE_POLICY_VERSION_MINIMUM=3.5
cd ../20260316_jude/open_source_code_for_reference/speechBCI/LanguageModelDecoder/runtime/server/x86 || { echo S6G_FAILED_CD; exit 1; }
P=/root/miniconda3/envs/py310lm/bin/python
"$P" setup.py install
ec=$?
echo "setup_exit_$ec"
if [ "$ec" -eq 0 ]; then
  "$P" -c 'import lm_decoder; print("lm_decoder OK")' && echo S6G_DONE || echo S6G_FAILED_IMPORT
else
  echo S6G_FAILED_BUILD
fi
date
