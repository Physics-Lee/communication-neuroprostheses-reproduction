# -*- coding: utf-8 -*-
"""OPT-6.7B rescoring of tlg_viterbi n-best lists (Card 2024 pipeline port).

Faithful port of nejm-brain-to-text/language_model/language-model-standalone.py:
  - candidate augmentation by word-swaps between top candidates
  - OPT scoring = sum of token log-probs - n_tokens * length_penalty
  - total = acs * ac_logp + (1 - alpha) * graph_logp + alpha * opt_logp
    (all nats; higher better; acs = post-hoc acoustic weight)

Input: tlg_viterbi nbest tsv (key, rank, ac_logp, graph_logp, sentence)
Usage:
  opt_rescore.py <nbest.tsv> <out.tsv> [--limit N] [--alpha A] [--acs A]
                 [--length-penalty P] [--augment 20] [--batch 48]
"""
import argparse
import collections
import sys

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = r".\20260710_Design_of_Large_Corpus\LLM\opt-6.7b"
QWEN_PATH = r".\20260710_Design_of_Large_Corpus\LLM\Qwen3.5-4B"


def read_nbest(path, limit):
    groups = collections.OrderedDict()
    with open(path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 5:
                continue
            key = p[0]
            ac, g = float(p[2]), float(p[3])
            sent = p[4]
            groups.setdefault(key, []).append((sent, ac, g))
            if limit and len(groups) >= limit:
                break
    return groups


def word_diff_ops(a_words, b_words):
    """ops[i] in {I,D,R}: how to turn a_words into b_words (word level)."""
    n, m = len(a_words), len(b_words)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1,
                           dp[i - 1][j - 1] + (a_words[i - 1] != b_words[j - 1]))
    ops, i, j = [], n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i][j] == dp[i - 1][j - 1] + (a_words[i - 1] != b_words[j - 1]):
            ops.append("R" if a_words[i - 1] != b_words[j - 1] else "M")
            i, j = i - 1, j - 1
        elif i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append("D"); i -= 1
        else:
            ops.append("I"); j -= 1
    return ops[::-1]


def augment_nbest(cands, top_k=20, acs=0.3, penalty=0.01):
    """Port of augment_nbest: swap replaced words between top candidates."""
    cands = sorted(cands, key=lambda c: -(c[2] + acs * c[1]))
    sents = [c[0].strip() for c in cands]
    acs_l = [c[1] for c in cands]
    g_l = [c[2] for c in cands]
    new = []
    for i1 in range(min(len(sents) - 1, top_k)):
        w1 = sents[i1].split()
        for i2 in range(i1 + 1, min(len(sents), top_k)):
            w2 = sents[i2].split()
            if len(w1) != len(w2):
                continue
            ops1 = word_diff_ops(w1, w2)
            ops2 = word_diff_ops(w2, w1)
            r1 = [i for i, o in enumerate(ops1) if o == "R"]
            r2 = [i for i, o in enumerate(ops2) if o == "R"]
            if len(r1) != len(r2) or not r1:
                continue
            for a, b in zip(r1, r2):
                if a >= len(w1) or b >= len(w2):
                    continue
                n1, n2 = w1.copy(), w2.copy()
                n1[a], n2[b] = w2[b], w1[a]
                base_ac = (acs_l[i1] + acs_l[i2]) / 2
                base_g = (g_l[i1] + g_l[i2]) / 2
                # official augment_nbest emits BOTH swapped variants
                for nw in (n1, n2):
                    ns = " ".join(nw)
                    if ns in sents or any(ns == x[0] for x in new):
                        continue
                    nac = base_ac - penalty * abs(base_ac)
                    ng = base_g - penalty * abs(base_g)
                    new.append((ns, nac, ng))
    out = list(cands) + [(s, a, g) for s, a, g in new]
    out.sort(key=lambda c: -(c[2] + acs * c[1]))
    return out


@torch.inference_mode()
def opt_scores(model, tok, device, sentences, length_penalty, batch=48,
               token_budget=1536):
    """Length-aware batching: the 248k-vocab Qwen logits blow up VRAM when a
    batch pads to a long outlier (Kunz garbage candidates), so pre-tokenize,
    sort by length and cap each batch by a total-token budget."""
    scores = [None] * len(sentences)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    enc = tok(sentences, padding=False, truncation=True, max_length=192)
    order = sorted(range(len(sentences)),
                   key=lambda i: -len(enc["input_ids"][i]))
    done = 0
    i = 0
    while i < len(order):
        maxlen = len(enc["input_ids"][order[i]])
        n = max(1, min(batch, token_budget // max(1, maxlen)))
        idx = order[i:i + n]
        i += n
        chunk = [sentences[j] for j in idx]
        while True:
            try:
                inputs = tok(chunk, return_tensors="pt", padding=True,
                             truncation=True, max_length=192)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = model(**inputs).logits
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if len(chunk) == 1:
                    raise
                print(f"  OOM: halving batch {len(chunk)} -> {len(chunk)//2}",
                      flush=True)
                i -= len(chunk) - len(chunk) // 2
                idx, chunk = idx[:len(chunk) // 2], chunk[:len(chunk) // 2]
        # memory-lean logprob: to score token t we need logits at position
        # t-1 (causal LM: position t-1 predicts x_t). The 09-30 refactor
        # gathered at position t — scoring each token with logits computed
        # AFTER seeing it — which made scores uncorrelated with the correct
        # ones (validated 2026-10-03: corr ~= 0 old-vs-new on Jude). Fixed by
        # shifting: rows lg[:, :-1] score tokens ids[:, 1:].
        lg = logits.float()
        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        tgt = lg[:, :-1].gather(2, ids[:, 1:].unsqueeze(-1)).squeeze(-1)  # [B, L-1]
        lse = torch.logsumexp(lg[:, :-1], dim=-1)                # [B, L-1]
        tok_lp = tgt - lse
        del lg, logits
        for b, j in enumerate(idx):
            n_tok = int(mask[b].sum())
            scores[j] = float(tok_lp[b, :n_tok - 1].sum().item()) \
                - n_tok * length_penalty
        done += len(idx)
        if done % 2400 < n:
            print(f"  scored {done}/{len(sentences)}", flush=True)
    print(f"  scored {done}/{len(sentences)}", flush=True)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("nbest_tsv")
    ap.add_argument("out_tsv")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--acs", type=float, default=0.3)
    ap.add_argument("--search-acscale", type=float, default=0.4,
                    help="acscale used at search time (for candidate ranking)")
    ap.add_argument("--length-penalty", type=float, default=0.0)
    ap.add_argument("--augment", type=int, default=20)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--ref-tsv", required=True,
                    help="1-best tsv (key TAB ref TAB hyp TAB score) for refs")
    ap.add_argument("--sweep-alpha", default="",
                    help="comma alphas: report WER for each instead of writing out")
    ap.add_argument("--case", choices=["upper", "lower"], default="lower",
                    help="case fed to the LM (official remove_punctuation() "
                         "lowercases before scoring; the 09-30 'official feeds "
                         "uppercase' claim was wrong, corrected 2026-10-03)")
    ap.add_argument("--model", default=MODEL_PATH,
                    help="rescoring LLM path (opt-6.7b or Qwen3.5-4B)")
    ap.add_argument("--dtype", default="float16", choices=["float16", "bfloat16"])
    args = ap.parse_args()

    groups = read_nbest(args.nbest_tsv, args.limit)
    print(f"{len(groups)} trials", flush=True)

    # search-time ranking + augmentation
    aug = {}
    for key, cands in groups.items():
        aug[key] = augment_nbest(cands, top_k=args.augment,
                                 acs=args.search_acscale)
    keys = list(aug.keys())

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.bfloat16 if args.dtype == "bfloat16" else torch.float16
    print(f"loading rescoring LM from {args.model} ({args.dtype})...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = None
    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=dt, device_map="auto")
    except Exception as e:
        print(f"AutoModelForCausalLM failed ({str(e)[:120]}); "
              "trying AutoModelForImageTextToText", flush=True)
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, torch_dtype=dt, device_map="auto")
    model.eval()
    print("model ready", flush=True)

    # score ALL augmented candidates (same set for every alpha)
    flat = []
    for key in keys:
        for sent, ac, g in aug[key]:
            s = sent.replace(">", "").replace("  ", " ").replace(" ,", ",") \
                    .replace(" .", ".").replace(" ?", "?")
            if args.case == "lower":
                s = s.lower()
            flat.append((key, s, ac, g))
    print(f"scoring {len(flat)} candidates with OPT...", flush=True)
    opt = opt_scores(model, tok, device, [f[1] for f in flat],
                     args.length_penalty, args.batch)
    del model
    if device != "cpu":
        torch.cuda.empty_cache()

    by_key = collections.defaultdict(list)
    for (key, sent, ac, g), o in zip(flat, opt):
        by_key[key].append((sent, ac, g, o))

    def decode_with(alpha, acs):
        tot_ed = tot_ref = n = 0
        out_rows = []
        for key in keys:
            best, best_sc = None, -np.inf
            for sent, ac, g, o in by_key[key]:
                sc = acs * ac + (1 - alpha) * g + alpha * o
                if sc > best_sc:
                    best_sc, best = sc, sent
            ref = key_ref.get(key)
            if ref is None:
                continue
            r = re.sub(r"[^A-Za-z'< ]", " ", ref).upper().split()
            h = best.upper().split()
            ed = editdistance.eval(r, h)
            tot_ed += ed
            tot_ref += len(r)
            n += 1
            out_rows.append((key, ref, best, best_sc, ed, len(r)))
        return tot_ed, tot_ref, n, out_rows

    import re
    import editdistance
    key_ref = {}
    with open(args.ref_tsv, encoding="utf-8") as rf:
        for line in rf:
            pp = line.rstrip("\n").split("\t")
            if len(pp) >= 2 and pp[0] != "key":
                key_ref[pp[0]] = pp[1]


    if args.sweep_alpha:
        best = None
        for a in map(float, args.sweep_alpha.split(",")):
            tot_ed, tot_ref, n, out_rows = decode_with(a, args.acs)
            print(f"alpha={a}: WER = {tot_ed}/{tot_ref} = "
                  f"{100*tot_ed/max(1,tot_ref):.2f}% over {n}")
            if best is None or tot_ed < best[0]:
                best = (tot_ed, a, out_rows)
        tot_ed, a, out_rows = best
        with open(args.out_tsv, "w", encoding="utf-8", newline="") as f:
            f.write("key\tref\thyp\tscore\ted\tref_len\n")
            for row in out_rows:
                f.write("\t".join(map(str, row)) + "\n")
        print(f"best alpha={a} saved -> {args.out_tsv}")
        return

    tot_ed, tot_ref, n, out_rows = decode_with(args.alpha, args.acs)
    print(f"alpha={args.alpha}: WER = {tot_ed}/{tot_ref} = "
          f"{100*tot_ed/max(1,tot_ref):.2f}% over {n}")
    with open(args.out_tsv, "w", encoding="utf-8", newline="") as f:
        f.write("key\tref\thyp\tscore\ted\tref_len\n")
        for row in out_rows:
            f.write("\t".join(map(str, row)) + "\n")
    print("saved", args.out_tsv)


if __name__ == "__main__":
    main()
