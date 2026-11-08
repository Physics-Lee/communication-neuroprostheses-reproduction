# -*- coding: utf-8 -*-
"""Diagnose WHY Qwen3.5-4B rescoring hurts: does its raw sentence score rank
the correct candidate highly within each n-best pool?

For a sample of trials: rank the near-reference candidate (min word-ED to ref)
under (a) pure Qwen logprob, (b) search-time score acs*ac + lm from the nbest
file. If Qwen ranks truth poorly -> model's sentence prior is the problem;
if it ranks truth fine -> the combination scale (alpha) is the problem.

Usage: qwen_rank_probe.py <nbest.tsv> <model_path> [sample_stride=14]
"""
import collections
import re
import sys

import editdistance
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def norm(s):
    """Normalize a transcript for edit-distance scoring.

    Replaces every character other than ASCII letters, apostrophes,
    ``<``/``>`` and spaces with a space, uppercases, and tokenizes on
    whitespace.

    Parameters
    ----------
    s : str
        Raw transcript string (reference or hypothesis).

    Returns
    -------
    list of str
        Uppercase word tokens.
    """
    return re.sub(r"[^A-Za-z'< ]", " ", s).upper().split()


def main():
    """Probe where the near-reference candidate ranks under Qwen vs search.

    For a strided sample of trials from an n-best TSV, scores every
    candidate with the Qwen LM, finds the near-reference candidate
    (minimum word edit distance), and reports its rank under both the
    pure Qwen log-probability and the search-time score, to separate
    LM-prior problems from interpolation-scale problems.

    Notes
    -----
    Arguments come from ``sys.argv``:
    ``qwen_rank_probe.py <nbest.tsv> <model_path> [sample_stride=14]``.
    """
    nbest_path, model_path = sys.argv[1], sys.argv[2]
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 14

    groups = collections.OrderedDict()
    with open(nbest_path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 5:
                groups.setdefault(p[0], []).append(
                    (p[4], float(p[2]), float(p[3])))
    keys = list(groups.keys())[::stride]
    print(f"{len(groups)} trials, probing {len(keys)}")

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    qwen_ranks, search_ranks, top1_qwen_is_truth = [], [], 0

    # refs from the companion final tsv
    final_path = nbest_path.replace("off_nbest_full", "off_final_full") \
        .replace("off_bp1_nbest", "off_bp1_final") \
        .replace("wl_nbest_full", "wl_final_full") \
        .replace("nbest_full", "final_full")
    refs = {}
    with open(final_path, encoding="utf-8") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if p[0] != "key":
                refs[p[0]] = norm(p[1])

    for k in keys:
        cands = groups[k]
        r = refs.get(k)
        if not r:
            continue
        sents = [c[0] for c in cands]
        # score with qwen (one trial at a time; batches of 24)
        scores = []
        for i in range(0, len(sents), 24):
            chunk = sents[i:i + 24]
            inputs = tok(chunk, return_tensors="pt", padding=True,
                         truncation=True, max_length=192)
            inputs = {a: b.to("cuda") for a, b in inputs.items()}
            with torch.inference_mode():
                lg = model(**inputs).logits.float()
            tgt = lg.gather(2, inputs["input_ids"].unsqueeze(-1)).squeeze(-1)
            lse = torch.logsumexp(lg, dim=-1)
            lp = (tgt - lse)
            for b in range(lp.shape[0]):
                n = int(inputs["attention_mask"][b].sum())
                scores.append(float(lp[b, 1:n].sum()))
        order_q = sorted(range(len(cands)), key=lambda i: -scores[i])
        rank_of = {c: i for i, c in enumerate(order_q)}
        search_sc = [0.325 * c[1] + c[2] for c in cands]
        order_s = sorted(range(len(cands)), key=lambda i: -search_sc[i])
        rank_s = {c: i for i, c in enumerate(order_s)}
        truth = min(range(len(cands)),
                    key=lambda i: editdistance.eval(r, sents[i].split()))
        qwen_ranks_t = rank_of[truth]
        search_ranks_t = rank_s[truth]
        qwen_ranks.append(qwen_ranks_t)
        search_ranks.append(search_ranks_t)
        if qwen_ranks_t == 0:
            top1_qwen_is_truth += 1
        print(f"{k}: pool={len(cands)} truth_ed="
              f"{editdistance.eval(r, sents[truth].split())} "
              f"qwen_rank={qwen_ranks_t} search_rank={search_ranks_t}",
              flush=True)

    n = len(qwen_ranks)
    print(f"\nmean rank of near-ref candidate: Qwen="
          f"{sum(qwen_ranks)/n:.1f}  search(ac+5g)={sum(search_ranks)/n:.1f}")
    print(f"Qwen argmax == near-ref candidate in {top1_qwen_is_truth}/{n}")


if __name__ == "__main__":
    main()
