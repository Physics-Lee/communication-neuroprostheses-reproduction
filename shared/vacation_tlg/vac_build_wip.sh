#!/bin/bash
# Build ~/bin/wenet_lattice_wip from the WIP-patched source.
# Reconstructed 2026-10-04 from the binary's embedded source list.
# Run inside WSL: bash ~/src_lattice/build_wip.sh
set -e
K=../shared/nejm-brain-to-text/language_model/runtime/core/kaldi
CORE=../shared/nejm-brain-to-text/language_model/runtime/core
cd ~/src_lattice
g++ -O2 -std=c++17 -fpermissive \
  -I. -I/usr/local/include -I"$K" -I"$CORE" \
  -include ~/src_lattice/glog_shim.h \
  wenet_lattice_wip.cc \
  "$K/base/kaldi-math.cc" \
  "$K/decoder/lattice-faster-decoder.cc" \
  "$K/decoder/lattice-faster-online-decoder.cc" \
  "$K/lat/determinize-lattice-pruned.cc" \
  "$K/lat/lattice-functions.cc" \
  -L/usr/local/lib -lfst -ldl -pthread \
  -o ~/bin/wenet_lattice_wip
echo "BUILD OK: ~/bin/wenet_lattice_wip"
