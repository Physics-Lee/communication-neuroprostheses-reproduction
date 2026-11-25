#!/bin/bash
# W2-1：openfst autoreconf 自试（2026-10-10 16:45，授权方案 A，3 轮上限的第 1 轮）
# 根因（夜间流水线触发 #9）：openfst-src 的 aclocal.m4 由 Automake 1.14 生成，系统 1.18 拒绝版本不匹配的重生成。
# autoreconf -fi 用系统 1.18 全量重生成 aclocal.m4/configure/Makefile.in，消解版本僵局。
set -x
date
cd ../20260316_jude/open_source_code_for_reference/speechBCI/LanguageModelDecoder/runtime/server/x86/fc_base/openfst-src || { echo W2_FAILED_CD; exit 1; }

autoreconf -fi 2>&1 | tail -20
ec=$?
echo "autoreconf_exit_$ec"
if [ "$ec" -ne 0 ]; then echo W2_FAILED_AUTORECONF; exit 1; fi

# autoreconf 过了就接着完成整个构建（依赖全缓存，仅 openfst 环节会真正干活）
cd ../20260316_jude/open_source_code_for_reference/speechBCI/LanguageModelDecoder/runtime/server/x86 || exit 1
export CMAKE_POLICY_VERSION_MINIMUM=3.5
P=/root/miniconda3/envs/py310lm/bin/python
"$P" setup.py install
ec2=$?
echo "setup_exit_$ec2"
"$P" -c 'import lm_decoder; print("lm_decoder OK")' && echo W2_DONE || echo W2_FAILED_BUILD
date
