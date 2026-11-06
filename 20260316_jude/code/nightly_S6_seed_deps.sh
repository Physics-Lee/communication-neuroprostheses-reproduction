#!/bin/bash
# S6 种子脚本：预置全部剩余 FetchContent/ExternalProject 依赖，使构建零联网（2026-10-10 00:10）
# 背景：github codeload / pytorch.org 从本网络时断时续（SSL reset），今晚构建已 4 次死于依赖下载。
# 原理：FetchContent/ExternalProject 只检查 stamp 文件决定是否下载；src 目录 + stamp 齐备即跳过。
set -x
FC=../20260316_jude/open_source_code_for_reference/speechBCI/LanguageModelDecoder/runtime/server/x86/fc_base
T=/root/miniconda3/envs/py310lm/lib/python3.10/site-packages/torch

# --- 克隆 cnpy / openfst（git 智能协议今晚通畅；codeload zip 才是被掐的） ---
cd /tmp
rm -rf cnpy openfst_seed
git clone -q --depth 1 https://github.com/rogersce/cnpy || { echo "SEED_FAILED_CNPY_CLONE"; exit 1; }
git clone -q --depth 1 -b 1.6.5 https://github.com/mjansche/openfst openfst_seed || { echo "SEED_FAILED_OPENFST_CLONE"; exit 1; }

# --- cnpy：源码做 src（CMakeLists 未设 URL_HASH，内容等价即可） ---
rm -rf "$FC/cnpy-src"
cp -r /tmp/cnpy "$FC/cnpy-src"
mkdir -p "$FC/cnpy-subbuild/cnpy-populate-prefix/src/cnpy-populate-stamp"
cd "$FC/cnpy-subbuild/cnpy-populate-prefix/src/cnpy-populate-stamp"
for s in mkdir download update patch configure build install test done; do touch "cnpy-populate-$s"; done

# --- libtorch：pip torch 符号链接农场（跳过下载即跳过 URL_HASH 校验；cmake 配置在 share/cmake/Torch） ---
rm -rf "$FC/libtorch-src"
mkdir -p "$FC/libtorch-src/lib"
for f in "$T"/lib/*; do ln -sf "$f" "$FC/libtorch-src/lib/$(basename "$f")"; done
ln -sfn "$T/share/cmake" "$FC/libtorch-src/lib/cmake"
ln -sfn "$T/include" "$FC/libtorch-src/include"
ln -sfn "$T/share" "$FC/libtorch-src/share"
mkdir -p "$FC/libtorch-subbuild/libtorch-populate-prefix/src/libtorch-populate-stamp"
cd "$FC/libtorch-subbuild/libtorch-populate-prefix/src/libtorch-populate-stamp"
for s in mkdir download update patch configure build install test done; do touch "libtorch-populate-$s"; done

# --- openfst：git clone 1.6.5 做 src（configure 是否在仓库里，探测） ---
rm -rf "$FC/openfst-src"
cp -r /tmp/openfst_seed "$FC/openfst-src"
ls "$FC/openfst-src/configure" >/dev/null 2>&1 || echo "OPENFST_NEEDS_AUTORECONF"

echo "SEED_DONE"
