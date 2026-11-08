# -*- coding: utf-8 -*-
"""Joint lexicon-constrained CTC beam search with word-level 3-gram KenLM
for the BtoT24 test partition (run in WSL: python3 decode_words_btot24_v2.py).

NOTE: G3.arpa (fst2arpa dump of the official G.fst) stores probabilities in
NATS, not log10 -- verified via reference perplexity (ppl ~245 after
nats->log10 correction). kenlm returns those values unchanged, so its
.log_prob is already a natural-log score: do NOT multiply by ln(10).

Replaces the earlier two-stage pipeline (decode_words_btot24.py) whose
lexicon lookup read lexicon_numbers.txt with tokens.txt ids -- the lexicon
ids are 1-based alphabetical over units.txt (AABERG = "1 7 12 15" =
AA B ER G), so all prior pronunciations (and the 31.5% WER) were wrong.

Decoding follows the official TLG idea without kaldi:
  - states live in a pronunciation trie built from lexicon_numbers.txt
    (ids via units.txt), i.e. only vocabulary words are reachable;
  - CTC prefix semantics per Graves: each state tracks (p_nb, p_b);
    class 0 = blank, classes 1..39 = phones, class 40 = ' | ' separator;
  - '|' closes the current word and scores it with kenlm
    (alpha * logP + beta per word); a phone with no trie child, or any phone
    at a word-end node, also allows a free boundary close (the optional
    silence between words in L.fst) so the next word can start with p;
  - pruning on log(p_nb + p_b) + alpha*logP_LM + beta*n_words.

Usage: decode_words_btot24_v2.py [--alpha A] [--beta B] [--beam N]
       [--limit N] [--workers K] [--out path.csv] [--grid A:B,A:B,...]
"""
import argparse
import csv
import math
import re
import time

import editdistance
import h5py
import kenlm
import numpy as np

import os as _os
LM_DIR = _os.environ.get(
    "W3_LM_DIR",
    "./20230831_willett/"
    "data/raw/20230616_willett_BtoT24/languageModel_work/languageModel")
H5_PATH = _os.environ.get(
    "W3_H5",
    "./20230831_willett/"
    "data/20230616_willett_BtoT24/run_60k/test_logits.hdf5")

BLANK, SEP, NCLS = 0, 40, 41
LN10 = math.log(10)
TOPK = 39             # phones kept per frame besides blank/sep (39 = all)
LM_BEAM_PENALTY = 20.0  # drop states worse than best minus this (nats)
NEG_INF = -math.inf
ROOT_ID = 0            # id(root) placeholder; real id()s are never 0


def _ladd(a, b):
    """log-add-exp for python floats.

    Numerically stable log(exp(a) + exp(b)); -inf operands are handled so
    that combining with an impossible path returns the other operand.

    Parameters
    ----------
    a : float
        First log-probability (may be -inf).
    b : float
        Second log-probability (may be -inf).

    Returns
    -------
    float
        log(exp(a) + exp(b)), or -inf when both inputs are -inf.
    """
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a)) if b > NEG_INF else a


def norm_ref(sent):
    """Normalize a reference sentence for WER scoring.

    Keeps only letters, apostrophes and spaces, upper-cases the result and
    splits it into words, matching the vocabulary convention of the
    lexicon and language model.

    Parameters
    ----------
    sent : str
        Raw reference sentence, e.g. an h5 ``sentence_label`` attribute.

    Returns
    -------
    list of str
        Uppercased reference words.
    """
    return re.sub(r"[^A-Za-z'< ]", " ", sent).upper().split()


# ---------------------------------------------------------------- trie / lexicon
class Node:
    """Trie node of the pronunciation lexicon.

    Children are keyed by phone class id (1..39), ``words`` holds the
    homophones whose pronunciations end exactly at this node (None when no
    word ends here), and ``last_ph`` records the phone leading into this
    node (None at the root).
    """
    __slots__ = ("children", "words", "last_ph")

    def __init__(self, last_ph=None):
        """Create an empty trie node.

        Parameters
        ----------
        last_ph : int or None, optional
            Phone class id of the edge leading into this node; None for the
            trie root (default None).
        """
        self.children = {}
        self.words = None      # homophones ending here
        self.last_ph = last_ph  # phone leading into this node (None for root)


def build_trie():
    """Build the pronunciation trie from lexicon_numbers.txt.

    Reads <LM_DIR>/lexicon_numbers.txt where each line is a vocabulary word
    followed by its 1-based units.txt phone ids. Those ids coincide with the
    model's phone classes 1..39, so each pronunciation becomes a path in the
    trie and only vocabulary words are reachable during decoding.

    Returns
    -------
    tuple(Node, int)
        (root, n_lines): the trie root node and the number of lexicon
        lines inserted.
    """
    root = Node()
    n_lines = 0
    with open(LM_DIR + "/lexicon_numbers.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            word = parts[0]
            # units.txt ids are 1-based and coincide with model classes 1..39
            pron = tuple(int(i) for i in parts[1:] if 1 <= int(i) <= 39)
            if not pron:
                continue
            node = root
            for p in pron:
                nxt = node.children.get(p)
                if nxt is None:
                    nxt = node.children[p] = Node(p)
                node = nxt
            if node.words is None:
                node.words = []
            if word not in node.words:
                node.words.append(word)
            n_lines += 1
    return root, n_lines


# ---------------------------------------------------------------- decoder
class State:
    """One hypothesis in the joint CTC/LM beam search.

    Combines Graves-style CTC prefix probabilities (p_nb for prefixes
    ending with a real token, p_b for prefixes ending with blank) with a
    lexicon-trie position, an accumulated KenLM score and a backpointer
    chain recording each closed word for the final traceback.
    """
    __slots__ = ("p_nb", "p_b", "node", "lm_state", "lm_score", "nwords",
                 "parent", "word", "ctx1", "ctx2")

    def __init__(self, p_nb, p_b, node, lm_state, lm_score, nwords,
                 parent, word, ctx1, ctx2):
        """Create a beam-search hypothesis state.

        Parameters
        ----------
        p_nb : float
            Log-probability of the CTC prefix ending with a real token.
        p_b : float
            Log-probability of the CTC prefix ending with blank.
        node : Node
            Trie node of the partially spelled word.
        lm_state : kenlm.State
            KenLM n-gram state after the last closed word.
        lm_score : float
            Accumulated ``alpha*logP + beta*nwords`` language-model score.
        nwords : int
            Number of closed words so far.
        parent : State or None
            Backpointer to the state at the last word close.
        word : str or None
            Word closed relative to ``parent`` (None = word still ongoing).
        ctx1 : str
            Last closed word (or "<s>").
        ctx2 : str
            Second-to-last closed word (or "<s>").
        """
        self.p_nb = p_nb        # log prob, prefix ends with a real token
        self.p_b = p_b          # log prob, prefix ends with blank
        self.node = node        # trie node of the partial word
        self.lm_state = lm_state
        self.lm_score = lm_score  # alpha*logP + beta*nwords accumulated
        self.nwords = nwords
        self.parent = parent    # backpointer at last word close
        self.word = word        # word closed vs parent (None = ongoing word)
        self.ctx1 = ctx1        # last closed word (or "<s>")
        self.ctx2 = ctx2        # second-to-last closed word (or "<s>")


class JointDecoder:
    """Joint lexicon-constrained CTC beam search decoder with a 3-gram LM.

    Pure-Python replacement for the paper's WFST decoder: at each frame,
    CTC prefix probabilities are advanced through a pronunciation trie (so
    only vocabulary words are reachable), closed words are scored by KenLM
    (``alpha * logP + beta`` per word), and the beam is pruned on
    ``log(p_nb + p_b) + lm_score`` with a hard penalty margin
    (LM_BEAM_PENALTY nats) and a size cap.
    """

    def __init__(self, root, lm, alpha, beta, beam, boundary="free"):
        """Create a decoder instance.

        Parameters
        ----------
        root : Node
            Pronunciation trie root from build_trie().
        lm : kenlm.LanguageModel
            3-gram KenLM model (G3.trie.bin; natural-log scores).
        alpha : float
            Language-model weight applied to each word's log-probability.
        beta : float
            Per-word insertion bonus added at every word close.
        beam : int
            Maximum number of states kept after each frame.
        boundary : str, optional
            Word-boundary mode: "free" (any phone can close a word at a
            word-end node), "blank" (only blank-separated mass bridges word
            boundaries) or "strict" (words close only on the '|'
            separator) (default "free").
        """
        self.root = root
        self.lm = lm
        self.alpha = alpha
        self.beta = beta
        self.beam = beam
        self.boundary = boundary  # free | blank | strict (SEP-only)

    def _close_words(self, st):
        """[(word, kenlm_state, lm_delta)] for every homophone of st.node.

        Scores each word ending at the hypothesis's current trie node with
        the 3-gram KenLM and applies the alpha/beta weighting.

        Parameters
        ----------
        st : State
            Hypothesis whose ``node`` holds the homophone list.

        Returns
        -------
        list of tuple
            One (word, kenlm_state, lm_delta) per homophone, where
            lm_delta is ``alpha * log_prob + beta`` to add to
            ``st.lm_score`` when closing that word.
        """
        out = []
        for w in st.node.words:
            nst = kenlm.State()
            r = self.lm.BaseFullScore(st.lm_state, w, nst)
            out.append((w, nst, self.alpha * r.log_prob + self.beta))
        return out

    def decode(self, logp):
        """Decode one trial's frame log-probs into the best word sequence.

        Keeps only blank, separator and the TOPK best phones per frame,
        advances every beam state under CTC prefix semantics (blank, '|'
        separator word closes, phone emissions with repeat handling and
        optional free-boundary word closes), prunes the beam each frame,
        then force-closes any open word (scoring "</s>") and follows the
        parent backpointers to recover the word sequence.

        Parameters
        ----------
        logp : numpy.ndarray
            (T, 41) float32 frame log-probabilities over CTC classes
            (0 = blank, 1..39 = phones, 40 = ' | ' separator).

        Returns
        -------
        list of str
            Decoded word sequence (empty when no hypothesis survives).
        """
        boundary = self.boundary
        T = logp.shape[0]
        keep = np.full((T, NCLS), NEG_INF, dtype=np.float32)
        keep[:, BLANK] = logp[:, BLANK]
        keep[:, SEP] = logp[:, SEP]
        if SEP - 1 > TOPK:
            top = np.argpartition(logp[:, 1:SEP], -TOPK, axis=1)[:, -TOPK:] + 1
            rows = np.arange(T)[:, None]
            keep[rows, top] = logp[rows, top]
        else:
            keep[:, 1:SEP] = logp[:, 1:SEP]

        beg = kenlm.State()
        self.lm.BeginSentenceWrite(beg)
        beams = {0: State(NEG_INF, 0.0, self.root, beg, 0.0, 0, None, None,
                          "<s>", "<s>")}
        root = self.root

        for t in range(T):
            frame = keep[t]
            new = {}

            def push(key, st):
                """Insert a new state under its dedup key or merge duplicates.

                States sharing the same (ctx2, ctx1, trie node) key are
                merged by log-add-exp of their CTC masses; the surviving
                entry adopts the LM bookkeeping of the better-scoring state.
                """
                cur = new.get(key)
                if cur is None:
                    new[key] = st
                    return
                if _ladd(st.p_nb, st.p_b) > _ladd(cur.p_nb, cur.p_b):
                    cur.p_nb, st.p_nb = st.p_nb, cur.p_nb
                    cur.p_b, st.p_b = st.p_b, cur.p_b
                    cur.lm_state = st.lm_state
                    cur.lm_score = st.lm_score
                    cur.nwords = st.nwords
                    cur.parent = st.parent
                    cur.word = st.word
                cur.p_nb = _ladd(cur.p_nb, st.p_nb)
                cur.p_b = _ladd(cur.p_b, st.p_b)

            for st in beams.values():
                tot = _ladd(st.p_nb, st.p_b)
                node = st.node
                nid = id(node) if node is not root else ROOT_ID
                base = (st.ctx2, st.ctx1, nid)

                # blank: same linguistic state, mass moves to p_b
                push(base, State(NEG_INF, tot + frame[BLANK], node,
                                 st.lm_state, st.lm_score, st.nwords,
                                 st.parent, st.word, st.ctx1, st.ctx2))

                # separator '|': close the current word
                lp = frame[SEP]
                if node.words:
                    for w, nst, delta in self._close_words(st):
                        push((st.ctx1, w, ROOT_ID),
                             State(tot + lp, NEG_INF, root, nst,
                                   st.lm_score + delta, st.nwords + 1,
                                   st, w, w, st.ctx1))
                elif node is root:
                    # repeated separator: no-op on the same prefix
                    push(base, State(st.p_nb + lp, NEG_INF, node,
                                     st.lm_state, st.lm_score, st.nwords,
                                     st.parent, st.word, st.ctx1, st.ctx2))

                # phones
                for p in range(1, SEP):
                    lp = frame[p]
                    if lp == NEG_INF:
                        continue
                    child = node.children.get(p)
                    if p == node.last_ph:
                        # same phone again: p_nb mass is a CTC repeat
                        push(base, State(st.p_nb + lp, NEG_INF, node,
                                         st.lm_state, st.lm_score, st.nwords,
                                         st.parent, st.word,
                                         st.ctx1, st.ctx2))
                        if child is not None:
                            # blank boundary, then a fresh emission extends
                            push((st.ctx2, st.ctx1, id(child)),
                                 State(st.p_b + lp, NEG_INF, child,
                                       st.lm_state, st.lm_score, st.nwords,
                                       st.parent, st.word,
                                       st.ctx1, st.ctx2))
                    elif child is not None:
                        push((st.ctx2, st.ctx1, id(child)),
                             State(tot + lp, NEG_INF, child, st.lm_state,
                                   st.lm_score, st.nwords, st.parent,
                                   st.word, st.ctx1, st.ctx2))
                    # free boundary at a word-end: close word, start new one
                    if node.words and node is not root:
                        if boundary == "strict":
                            pass
                        elif p == node.last_ph:
                            # only blank-separated mass can bridge repeats
                            if boundary == "free" or boundary == "blank":
                                ch2 = root.children.get(p)
                                if ch2 is not None:
                                    for w, nst, delta in self._close_words(st):
                                        push((st.ctx1, w, id(ch2)),
                                             State(st.p_b + lp, NEG_INF, ch2,
                                                   nst, st.lm_score + delta,
                                                   st.nwords + 1, st, w,
                                                   w, st.ctx1))
                        else:
                            mass = st.p_b if boundary == "blank" else tot
                            ch2 = root.children.get(p)
                            if ch2 is not None:
                                for w, nst, delta in self._close_words(st):
                                    push((st.ctx1, w, id(ch2)),
                                         State(mass + lp, NEG_INF, ch2, nst,
                                               st.lm_score + delta,
                                               st.nwords + 1, st, w,
                                               w, st.ctx1))
            if not new:
                break
            scored = sorted(((-_ladd(s.p_nb, s.p_b) - s.lm_score, i, s)
                             for i, s in enumerate(new.values())))
            best = -scored[0][0]
            beams = {}
            for neg, _, s in scored:
                if -neg < best - LM_BEAM_PENALTY or len(beams) >= self.beam:
                    break
                beams[id(s)] = s

        # finalize: force-close any open word
        best_tot, best_st = NEG_INF, None
        for st in beams.values():
            tot = _ladd(st.p_nb, st.p_b)
            if st.node is self.root:
                cands = [(None, st.lm_state, 0.0, tot)]
            elif st.node.words:
                cands = [(w, nst, delta, tot)
                         for w, nst, delta in self._close_words(st)]
            else:
                continue
            for w, nst, delta, mass in cands:
                end = kenlm.State()
                r = self.lm.BaseFullScore(nst, "</s>", end)
                tot_final = (mass + st.lm_score + delta
                             + self.alpha * r.log_prob)
                if tot_final > best_tot:
                    best_tot = tot_final
                    best_st = State(mass, NEG_INF, self.root, nst,
                                    st.lm_score + delta, st.nwords + 1,
                                    st, w, w, st.ctx1)
        if best_st is None:
            return []
        words = []
        node = best_st
        while node is not None:
            if node.word:
                words.append(node.word)
            node = node.parent
        words.reverse()
        return words


TRIALS = []
DEC = None


def decode_one(tno):
    """Decode a single trial (worker function for the multiprocessing pool).

    Reads the trial from the module-global TRIALS and decodes it with the
    module-global DEC JointDecoder, so fork-shared trie/LM data is reused
    without pickling.

    Parameters
    ----------
    tno : int
        Index into TRIALS.

    Returns
    -------
    tuple(int, list of str)
        (tno, words): the trial index and its decoded word sequence.
    """
    sess, key, logp, sent = TRIALS[tno]
    return tno, DEC.decode(logp)


def main():
    """Parse arguments, load the trie/LM/logits and run the beam search.

    Builds the lexicon trie, loads the KenLM 3-gram and reads per-trial
    log-probs from H5_PATH, then either sweeps an --alpha:beta grid on the
    --limit subset or decodes the full test set once, printing WER and
    optionally writing a per-trial csv via --out.
    """
    global TRIALS, DEC
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--beta", type=float, default=0.3)
    ap.add_argument("--beam", type=int, default=300)
    ap.add_argument("--boundary", default="free", choices=["free", "blank", "strict"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--grid", default="",
                    help="alpha:beta,alpha:beta,... sweep on --limit subset")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    print("building trie...", flush=True)
    root, n_lines = build_trie()
    print(f"trie built from {n_lines} lexicon lines", flush=True)
    lm = kenlm.LanguageModel(LM_DIR + "/G3.trie.bin")
    print("kenlm loaded", flush=True)

    TRIALS = []
    with h5py.File(H5_PATH, "r") as f:
        for sess in sorted(f.keys()):
            for key in sorted(f[sess].keys()):
                g = f[sess][key]
                TRIALS.append((sess, key,
                               g["log_probs"][:].astype(np.float32),
                               g.attrs["sentence_label"]))
    if args.limit:
        TRIALS = TRIALS[: args.limit]
    print(f"{len(TRIALS)} trials", flush=True)

    def run_grid(alpha, beta):
        """Decode all trials at one (alpha, beta) operating point.

        Builds a JointDecoder, decodes every trial (in a fork-based
        multiprocessing pool when --workers > 1) and accumulates
        corpus-level WER as total edit distance over total reference words.

        Parameters
        ----------
        alpha : float
            Language-model weight for this run.
        beta : float
            Per-word insertion bonus for this run.

        Returns
        -------
        tuple(float, dict)
            (wer, hyps): word error rate as a fraction and a dict mapping
            trial index to its decoded word list.
        """
        DEC = JointDecoder(root, lm, alpha, beta, args.beam, args.boundary)
        globals()["DEC"] = DEC
        if args.workers > 1:
            from multiprocessing import get_context
            # python 3.14 switched the default to forkserver, which would
            # lose the fork-shared TRIALS/trie/kenlm -- force fork
            ctx = get_context("fork")
            with ctx.Pool(args.workers) as pool:
                res = pool.map(decode_one, range(len(TRIALS)), chunksize=4)
        else:
            res = [decode_one(i) for i in range(len(TRIALS))]
        tot_ed = tot_ref = 0
        hyps = {}
        for tno, words in res:
            ref = norm_ref(TRIALS[tno][3])
            hyps[tno] = words
            tot_ed += editdistance.eval(ref, words)
            tot_ref += len(ref)
        return tot_ed / max(1, tot_ref), hyps

    if args.grid:
        for spec in args.grid.split(","):
            a, b = map(float, spec.split(":"))
            t0 = time.time()
            wer, _ = run_grid(a, b)
            print(f"alpha={a:<5} beta={b:<4} WER={100*wer:6.2f}%  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        return

    t0 = time.time()
    wer, hyps = run_grid(args.alpha, args.beta)
    print(f"\n=== BtoT24 test WER (joint lexicon beam, alpha={args.alpha} "
          f"beta={args.beta} beam={args.beam}): {100*wer:.2f}% over "
          f"{len(TRIALS)} sentences ({time.time()-t0:.0f}s) ===", flush=True)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="") as fo:
            w = csv.writer(fo)
            w.writerow(["session", "sentence_label", "ref_words", "hyp_words",
                        "edit_distance", "ref_len"])
            for tno, (sess, key, _, sent) in enumerate(TRIALS):
                ref = norm_ref(sent)
                hyp = hyps[tno]
                w.writerow([sess, sent, " ".join(ref), " ".join(hyp),
                            editdistance.eval(ref, hyp), len(ref)])
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
