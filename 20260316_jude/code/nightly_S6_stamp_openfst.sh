#!/bin/bash
# 补 openfst 的 ExternalProject stamp（跳过被网络掐死的 2MB 下载；源码已种子）
set -x
ST=../20260316_jude/open_source_code_for_reference/speechBCI/LanguageModelDecoder/runtime/server/x86/build/temp.linux-x86_64-cpython-310/openfst-prefix/src/openfst-stamp
mkdir -p "$ST"
for s in mkdir download update patch; do touch "$ST/openfst-$s"; done
ls -la "$ST"
echo STAMP_DONE
