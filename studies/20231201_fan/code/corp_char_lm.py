"""Character 5-gram and CTC prefix beam search from the CORP v9 replay."""

from collections import defaultdict

import numpy as np

from tier3_corp_sim_official_labels import gather_trials, text_to_ids


class CharNGram:
    """Char n-gram with stupid backoff over 1-based ids (1..31)."""

    def __init__(self, order=5, backoff=0.4):
        self.order = order
        self.backoff_log = np.log10(backoff)
        self.counts = [defaultdict(lambda: defaultdict(int))
                       for _ in range(order + 1)]
        self.totals = [defaultdict(int) for _ in range(order + 1)]
        self.n_uni = 0

    def train(self, sentences_ids):
        """Fit n-gram counts from seed-session character sequences."""
        for ids in sentences_ids:
            padded = (0,) * (self.order - 1) + tuple(ids)  # 0 = BOS pad
            for t in range(self.order - 1, len(padded)):
                for n in range(1, self.order + 1):
                    ctx = padded[t - n + 1:t]
                    if len(ctx) < n - 1:
                        continue
                    self.counts[n][ctx][padded[t]] += 1
                    self.totals[n][ctx] += 1
        self.n_uni = self.totals[1][()]
        return self

    def logprob(self, ctx_ids, nxt):
        """log10 P(nxt | ctx), stupid backoff; ctx 0-padded to order-1."""
        ctx = tuple(ctx_ids)
        if len(ctx) > self.order - 1:
            ctx = ctx[-(self.order - 1):]
        for n in range(len(ctx) + 1, 0, -1):
            c = ctx[len(ctx) - (n - 1):]
            tab = self.counts[n][c]
            tot = self.totals[n][c]
            if tot and nxt in tab:
                return np.log10(tab[nxt] / tot)
        return self.backoff_log + np.log10(
            max(1, 1) / 32.0)


def train_lm_from_seed_sessions(nsd):
    """Fit the decoder LM from the eleven seed sessions."""
    sents_ids = []
    for si in range(11):
        _, _, sents = gather_trials(nsd, si)
        for s in sents:
            ids = text_to_ids(s)
            if len(ids) >= 3:
                sents_ids.append(ids)
    lm = CharNGram().train(sents_ids)
    return lm, len(sents_ids)


def ctc_beam_lm(logits, lm, alpha=0.3, beam=12, topk=9):
    """CTC prefix beam search; LM enters the pruning score.

    logits: (T, 32) raw; index 31 = blank; label id = class index + 1.
    Returns 1-based label id list.
    """
    T = logits.shape[0]
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    logp = np.log(np.maximum(p, 1e-12))

    NEG = -1e30
    # prefix -> [logp_blank, logp_nonblank, lm_log10]
    beams = {(): (0.0, NEG, 0.0)}
    for t in range(T):
        row = logp[t]
        top_cls = np.argpartition(row[:31], -topk)[-topk:]
        nb_next = defaultdict(lambda: NEG)
        pb_next = defaultdict(lambda: NEG)
        lm_next = {}
        # rank current beams by combined score
        ranked = sorted(
            beams.items(),
            key=lambda kv: kv[1][0] if kv[1][1] <= NEG / 2
            else np.logaddexp(kv[1][0], kv[1][1]) + alpha * kv[1][2],
            reverse=True)[:beam]
        for prefix, (pb, pnb, lms) in ranked:
            tot = pb if pnb <= NEG / 2 else np.logaddexp(pb, pnb)
            if tot <= NEG / 2:
                continue
            # blank
            nb = row[31] + tot
            pb_next[prefix] = (nb if pb_next[prefix] <= NEG / 2
                               else np.logaddexp(pb_next[prefix], nb))
            lm_next.setdefault(prefix, lms)
            # char extensions
            for c in top_cls:
                cid = int(c) + 1
                if prefix and prefix[-1] == cid:
                    # same char, no blank between: CTC collapse keeps
                    # the prefix unchanged (smoke-test bugfix 2026-11-07:
                    # this missing term let LM-rewarded duplicates of
                    # '>' and '~' win over the true prefix)
                    v_stay = row[c] + pnb
                    if v_stay > NEG / 2:
                        nb_next[prefix] = (
                            v_stay if nb_next[prefix] <= NEG / 2
                            else np.logaddexp(nb_next[prefix], v_stay))
                    # blank-separated double emit -> prefix + cid via pb
                    base = pb
                else:
                    base = tot
                if base <= NEG / 2:
                    continue
                cand = prefix + (cid,)
                v = row[c] + base
                if cand not in lm_next:
                    lm_next[cand] = lms + lm.logprob(prefix, cid)
                nb_next[cand] = (v if nb_next[cand] <= NEG / 2
                                 else np.logaddexp(nb_next[cand], v))
        beams = {}
        for cand, v in nb_next.items():
            beams[cand] = (NEG, v, lm_next[cand])
        for cand, v in pb_next.items():
            pb, pnb, lms = beams.get(cand, (NEG, NEG, lm_next[cand]))
            beams[cand] = (v if pb <= NEG / 2 else np.logaddexp(pb, v),
                           pnb, lms)
        if not beams:
            break
    best = max(
        beams.items(),
        key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]) + alpha * kv[1][2])
    return list(best[0])
