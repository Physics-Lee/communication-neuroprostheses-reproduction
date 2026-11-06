# -*- coding: utf-8 -*-
"""Two-stage word decoding for BtoT24 test partition (run in WSL).

Stage 1: pyctcdecode acoustic-only beam -> n-best phoneme strings (joint CTC
         logp, collapse handled). Model class 40 (' | ') is dropped.
Stage 2: word-segmentation DP over each phoneme n-best with the Willett
         lexicon (exact pronunciations; fuzzy ed<=1 fallback) scored by the
         recovered 3-gram (G3.trie.bin) via kenlm BaseFullScore.

Node = (score, lm_state, ctx2, parent_node, word). Backtrack walks parent
references, so pruned ancestors stay alive via object refs.

Usage: decode_words_btot24.py [--alpha 0.5] [--edpen 3.0] [--nbeams 20]
       [--beamk 20] [--limit N] [--out path.csv]
"""
import argparse
import csv
import math
import re
import time
from collections import defaultdict

import editdistance
import h5py
import kenlm
import numpy as np
from pyctcdecode import build_ctcdecoder as _b

LM_DIR = ("./20230831_willett/data/raw/"
          "20230616_willett_BtoT24/languageModel_work/languageModel")
H5_PATH = "./20230831_willett/data/20230616_willett_BtoT24/run_60k/test_logits.hdf5"

PHONES = ["AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER",
          "EY", "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG",
          "OW", "OY", "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W",
          "Y", "Z", "ZH"]  # model classes 1..39
CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLM"  # one char per phoneme
PH2CH = dict(zip(PHONES, CHARS))
CH2PH = dict(zip(CHARS, PHONES))
LN10 = math.log(10)
MAXWORD = 13  # max phonemes per word span


def load_lexicon():
    id2ph = {}
    for line in open(LM_DIR + "/tokens.txt", encoding="utf-8"):
        p = line.split()
        if len(p) == 2:
            id2ph[int(p[1])] = p[0]
    pron2words = defaultdict(list)
    by_first = defaultdict(list)  # (len, first_phone) -> [(pron, word)]
    uni = {}
    for line in open(LM_DIR + "/lexicon_numbers.txt", encoding="utf-8"):
        p = line.split()
        if len(p) < 2:
            continue
        word = p[0]
        pron = tuple(id2ph[int(i)] for i in p[1:]
                     if int(i) <= 41 and id2ph[int(i)] not in ("SIL", "#0"))
        if not pron:
            continue
        pron2words[pron].append(word)
        by_first[(len(pron), pron[0])].append((pron, word))
    for line in open(LM_DIR + "/G3.part1", encoding="utf-8"):
        p = line.rstrip("\n").split("\t")
        if len(p) >= 2:
            try:
                uni[p[1]] = float(p[0])
            except ValueError:
                pass
    return pron2words, by_first, uni


def norm_ref(sent):
    return re.sub(r"[^A-Za-z'< ]", " ", sent).upper().split()


class WordDecoder:
    def __init__(self, lm_path, pron2words, by_first, uni,
                 alpha, edpen, wordpen, beamk):
        self.lm = kenlm.LanguageModel(lm_path)
        self.pron2words = pron2words
        self.by_first = by_first
        self.uni = uni
        self.alpha = alpha
        self.edpen = edpen
        self.wordpen = wordpen
        self.beamk = beamk

    def _span_candidates(self, rem):
        """{span_len: [(word, ed), ...]} for spans starting at position 0 of rem.

        Exact matches always win; fuzzy tolerance grows with span length
        (ed <= len//3 + 1, capped 3) and is searched in (len, first-phone)
        buckets, capped and ordered by (ed, unigram prior).
        """
        out = {}
        for k in range(1, min(MAXWORD, len(rem)) + 1):
            span = tuple(rem[:k])
            exact = self.pron2words.get(span)
            if exact:
                out[k] = [(w, 0) for w in exact]
                continue
            edmax = min(3, k // 3 + 1) if k > 1 else 0
            if edmax == 0:
                continue
            cands = []
            for L in range(max(1, k - edmax), k + edmax + 1):
                for pron, w in self.by_first.get((L, span[0]), ()):
                    ed = editdistance.eval(span, pron)
                    if ed <= edmax:
                        cands.append((w, ed, self.uni.get(w, -20.0)))
            cands.sort(key=lambda x: (x[1], -x[2]))
            out[k] = [(w, ed) for w, ed, _ in cands[:6]]
        return out

    def decode(self, phoneme_chars, acoustic_logp):
        seq = [CH2PH[c] for c in phoneme_chars]
        L = len(seq)
        if L == 0:
            return None, None, None
        init_state = kenlm.State()
        self.lm.BeginSentenceWrite(init_state)
        root = (0.0, init_state, ("", "<s>"), None, None)
        pos_nodes = {0: [root]}

        for i in range(L):
            nodes = pos_nodes.pop(i, [])
            if not nodes:
                continue
            rem = seq[i:]
            cands = self._span_candidates(rem)
            new_by_pos = defaultdict(list)
            for node in nodes:
                score, state, ctx, _, _ = node
                for k, words in cands.items():
                    if k > len(rem):
                        continue
                    for w, ed in words:
                        out_state = kenlm.State()
                        r = self.lm.BaseFullScore(state, w, out_state)
                        ns = score + self.alpha * r.log_prob - self.edpen * ed + self.wordpen
                        new_by_pos[i + k].append(
                            (ns, out_state, (ctx[1], w), node, w))
            for pos, lst in new_by_pos.items():
                merged = pos_nodes.get(pos, []) + lst
                merged.sort(key=lambda t: -t[0])
                seen = set()
                out = []
                for n in merged:
                    if n[2] in seen:
                        continue
                    seen.add(n[2])
                    out.append(n)
                    if len(out) >= self.beamk:
                        break
                pos_nodes[pos] = out

        best = None
        for node in pos_nodes.get(L, ()):
            score, state, ctx, _, _ = node
            out_state = kenlm.State()
            r = self.lm.BaseFullScore(state, "</s>", out_state)
            tot = score + self.alpha * r.log_prob
            if best is None or tot > best[0]:
                best = (tot, node)
        if best is None:
            return None, None, None
        words = []
        node = best[1]
        while node[4] is not None:
            words.append(node[4])
            node = node[3]
        words.reverse()
        return acoustic_logp / LN10 + best[0], words, best[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--edpen", type=float, default=2.5)
    ap.add_argument("--wordpen", type=float, default=-1.5)
    ap.add_argument("--nbeams", type=int, default=20)
    ap.add_argument("--beamk", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="/root/lm/word_decode_v1.csv")
    args = ap.parse_args()

    pron2words, by_first, uni = load_lexicon()
    print(f"lexicon prons: {len(pron2words)}", flush=True)
    decoder = _b(labels=list(CHARS))
    wd = WordDecoder(LM_DIR + "/G3.trie.bin", pron2words, by_first, uni,
                     args.alpha, args.edpen, args.wordpen, args.beamk)

    rows = []
    n_done = 0
    t0 = time.time()
    with h5py.File(H5_PATH, "r") as f:
        trials = [(sess, key) for sess in sorted(f.keys())
                  for key in sorted(f[sess].keys())]
        if args.limit:
            trials = trials[: args.limit]
        for sess, key in trials:
            g = f[sess][key]
            logp = g["log_probs"][:, :40]  # drop model class 40 (' | ')
            beams = decoder.decode_beams(logp, beam_width=100,
                                         beam_prune_logp=-10.0)
            beams = beams[: args.nbeams]  # sorted by plogp desc
            best = None
            for text, _, _, _, plogp in beams:
                tot, words, _ = wd.decode(text, plogp)
                if tot is not None and (best is None or tot > best[0]):
                    best = (tot, words)
            ref = norm_ref(g.attrs["sentence_label"])
            hyp = best[1] if best else []
            ed = editdistance.eval(ref, hyp)
            rows.append([sess, g.attrs["sentence_label"], " ".join(ref),
                         " ".join(hyp), ed, len(ref)])
            n_done += 1
            if n_done % 20 == 0:
                wer = sum(r[4] for r in rows) / sum(r[5] for r in rows)
                print(f"[{time.strftime('%H:%M:%S')}] {n_done}/{len(trials)} "
                      f"running WER={100*wer:.2f}% ({time.time()-t0:.0f}s)",
                      flush=True)
    tot_ed = sum(r[4] for r in rows)
    tot_ref = sum(r[5] for r in rows)
    print(f"\n=== word WER (alpha={args.alpha} edpen={args.edpen}): "
          f"{tot_ed}/{tot_ref} = {100*tot_ed/max(1,tot_ref):.2f}% "
          f"over {len(rows)} sentences ===")
    with open(args.out, "w", encoding="utf-8", newline="") as fo:
        w = csv.writer(fo)
        w.writerow(["session", "sentence_label", "ref_words", "hyp_words",
                    "edit_distance", "ref_len"])
        w.writerows(rows)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
