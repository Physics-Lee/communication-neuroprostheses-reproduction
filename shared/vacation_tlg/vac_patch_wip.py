# -*- coding: utf-8 -*-
"""Patch ~/src_lattice/wenet_lattice.cc -> wenet_lattice_wip.cc: add a word
insertion penalty (wip) knob. Applied on the WSL side because the repo copy
of wenet_lattice.cc kept getting corrupted by something on the Windows side
(observed twice on 2026-10-04; all other files unaffected)."""
import sys

p = "/home/yixuanli/src_lattice/wenet_lattice.cc"
out = "/home/yixuanli/src_lattice/wenet_lattice_wip.cc"
s = open(p, encoding="utf-8").read()
orig = s

# 1) usage comment
s = s.replace(
    "//        [min_active=200] [nbest=100] [nbest_prefix=-] [build=0] [skip=1.0]",
    "//        [min_active=200] [nbest=100] [nbest_prefix=-] [build=0] [skip=1.0]\n"
    "//        [wip=0.0]  (word insertion penalty in nats; >0 discourages words)")

# 2) global + hook in MappedArcIterator::Value()
s = s.replace(
    """class MappedArcIterator : public fst::ArcIteratorBase<fst::StdArc> {""",
    """// word insertion penalty (nats, added to every word-emitting arc; negative
// = word bonus). Idempotent: Value() re-copies a_ from the mmap each call.
static float g_wip = 0.0f;
class MappedArcIterator : public fst::ArcIteratorBase<fst::StdArc> {""")

s = s.replace(
    """  const fst::StdArc &Value() const override {
    m_->arc(s_, (int32_t)i_, &a_);
    a_.ilabel = remap_ilabel(a_.ilabel);
    return a_;
  }""",
    """  const fst::StdArc &Value() const override {
    m_->arc(s_, (int32_t)i_, &a_);
    a_.ilabel = remap_ilabel(a_.ilabel);
    if (a_.olabel != 0) a_.weight = a_.weight.Value() + g_wip;
    return a_;
  }""")

# 3) CLI arg + log
s = s.replace(
    """  const float skip_thresh = (argc > 15) ? (float)atof(argv[15]) : 1.0f;""",
    """  const float skip_thresh = (argc > 15) ? (float)atof(argv[15]) : 1.0f;
  g_wip = (argc > 16) ? (float)atof(argv[16]) : 0.0f;""")

s = s.replace(
    """min_act=%d nbest=%d skip=%.2f\\n",""",
    """min_act=%d nbest=%d skip=%.2f wip=%.2f\\n",""")
s = s.replace(
    """            min_active, nbest, skip_thresh);""",
    """            min_active, nbest, skip_thresh, g_wip);""")

n_changed = sum(1 for tag in ["[wip=0.0]", "static float g_wip", "g_wip;",
                              "g_wip = (argc > 16", "wip=%.2f", "skip_thresh, g_wip)"])
if s == orig or n_changed < 6:
    print("PATCH FAILED — patterns not found; aborting", file=sys.stderr)
    sys.exit(1)
open(out, "w", encoding="utf-8").write(s)
print(f"patched -> {out} ({len(s)} bytes, {n_changed}/6 markers)")
