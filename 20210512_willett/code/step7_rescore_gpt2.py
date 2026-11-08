# -*- coding: utf-8 -*-
"""M1c：GPT-2 1558M 重打分——对包内 Step6 KaldiOutput 的 n-best 重打分并重选（HF transformers 版）。

复刻官方 Step7_gpt2Rescore.ipynb 逻辑：
- 每个候选 y：text = content[y]（空串→' '），编码 content[0].upper()+content[1:]，
  前后加 <|endoftext|>(50256)
- lmRescore[y] = -sum_t logP(token_t | prefix)（即句子的负对数似然）
- newBest = argmin(acScore + 2.0*lmRescore)
- errCounts 口径同 Step6：真值 '>'→' '，'~'→'.'，'#'→''，CER/WER 保留空格

与官方的差异（如实记录）：
- 官方 TF1 + openai/gpt-2 1558M fp32 batch=1；此处 HF transformers gpt2-xl fp32，候选按句批处理。
  模型与分词器同源（OpenAI GPT-2），数值差异应在 1e-3 量级，边界句选择可能极少量翻转。
- 官方处理全部句（train+test）；此处只处理 val 句（选句逐句独立，聚合指标不受影响）。

对齐目标（包内 Step7 errCounts，M1a 已核）：Blocks 0.90/3.21，Trials 0.34/1.97。
"""
import os
import argparse
import numpy as np
import scipy.io
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ROOT = r'.\20210512_willett\data\20210512_willett\handwritingBCIData'
OUT_DIR = r'.\20210512_willett\result'

DATA_DIRS = ['t5.2019.05.08', 't5.2019.11.25', 't5.2019.12.09', 't5.2019.12.11', 't5.2019.12.18',
             't5.2019.12.20', 't5.2020.01.06', 't5.2020.01.08', 't5.2020.01.13', 't5.2020.01.15']

EOT = 50256


def wer(r, h):
    """Levenshtein edit distance between two token sequences.

    Parameters
    ----------
    r : list or str
        Reference tokens (ground truth).
    h : list or str
        Hypothesis tokens (decoded text).

    Returns
    -------
    int
        Minimum number of insertions/deletions/substitutions to turn ``h``
        into ``r``.
    """
    d = np.zeros((len(r) + 1, len(h) + 1), dtype=np.int32)
    for i in range(len(r) + 1):
        d[i, 0] = i
    for j in range(len(h) + 1):
        d[0, j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            if r[i - 1] == h[j - 1]:
                d[i, j] = d[i - 1, j - 1]
            else:
                d[i, j] = min(d[i - 1, j - 1] + 1, d[i - 1, j] + 1, d[i, j - 1] + 1)
    return int(d[len(r), len(h)])


def read_kaldi_lattice_file(file_name, file_type):
    """Read a Kaldi lattice transcript or score file into Python objects.

    Port of the official ``kaldiReadWrite.readKaldiLatticeFile``: each line
    carries a discarded index token followed by the payload.

    Parameters
    ----------
    file_name : str
        Path to the lattice file (``<i>_transcript.txt`` or
        ``<i>_best_*score.ark``).
    file_type : str
        ``'numeric'`` for per-line float scores, ``'string'`` for per-line
        transcripts (with ``<space>`` mapped to a space and the literal
        ``\\n`` token to the empty string).

    Returns
    -------
    numpy.ndarray or list of str
        Float64 array of scores for ``'numeric'``; list of decoded
        transcript strings for ``'string'``.

    Raises
    ------
    ValueError
        If ``file_type`` is neither ``'numeric'`` nor ``'string'``.
    """
    with open(file_name, 'r') as f:
        all_lines = f.readlines()
    content = []
    for x in range(len(all_lines)):
        split_str = all_lines[x].split(' ')
        if file_type == 'numeric':
            content.append(float(split_str[1]))
        elif file_type == 'string':
            del split_str[0]
            for k in range(len(split_str)):
                if split_str[k] == '<space>':
                    split_str[k] = ' '
                elif split_str[k] == '\n':
                    split_str[k] = ''
            content.append(''.join(split_str))
        else:
            raise ValueError('Wrong file type.')
    if file_type == 'numeric':
        content = np.array(content)
    return content


@torch.no_grad()
def gpt2_nll_batch(model, tokenizer, texts, device, chunk=32):
    """一批候选句子的负对数似然（每个 = [EOT] + tokens + [EOT] 的条件 NLL，从第 1 个 token 起）。
    分块前向（chunk 条/次），控制显存（与训练进程共存时尤其重要）。

    Parameters
    ----------
    model : transformers.GPT2LMHeadModel
        GPT-2 model in eval mode (no gradients recorded).
    tokenizer : transformers.GPT2TokenizerFast
        Tokenizer matching ``model``.
    texts : list of str
        Candidate sentence strings; each is lowercased-except-first-char
        (``t[0].upper() + t[1:]``), empty strings replaced by ``' '``, and
        bracketed by ``<|endoftext|>`` before scoring.
    device : str or torch.device
        Device the model lives on.
    chunk : int, optional
        Number of candidates per forward pass (default 32); larger chunks
        use more GPU memory.

    Returns
    -------
    numpy.ndarray
        Float array of shape ``(len(texts),)`` with each candidate's
        negative log-likelihood.
    """
    out = []
    for c0 in range(0, len(texts), chunk):
        sub = texts[c0:c0 + chunk]
        encs = []
        for t in sub:
            if t == '':
                t = ' '
            ids = tokenizer.encode(t[0].upper() + t[1:])
            encs.append([EOT] + ids + [EOT])
        maxlen = max(len(e) for e in encs)
        batch = torch.full((len(encs), maxlen), EOT, dtype=torch.long)
        for i, e in enumerate(encs):
            batch[i, :len(e)] = torch.tensor(e)
        batch = batch.to(device)
        logits = model(batch).logits.float()
        logprobs = torch.log_softmax(logits, dim=-1)
        for i, e in enumerate(encs):
            L = len(e)
            tgt = batch[i, 1:L]
            lp = logprobs[i, 0:L - 1, :].gather(1, tgt.unsqueeze(1)).sum()
            out.append(-float(lp))
    return np.array(out)


def main():
    """GPT-2-rescore the official Step6 n-best lists and verify against Step7.

    Loads GPT-2 (default gpt2-xl), then for each validation sentence of the
    chosen partition computes GPT-2 NLLs for every n-best candidate,
    reselects ``argmin(ac + 2.0*nll)``, computes char/word errors against
    the prompt (Step6 conventions), prints the rescored CER/WER and the
    overlap with the bigram selection, checks per-sentence errCounts
    against the packaged Step7 errCounts, and writes
    ``result/m1c_rescore_<cv_part>.csv``.

    Parameters
    ----------
    None
        Options come from the command line: ``--cv_part`` (default
        HeldOutBlocks), ``--model`` (default gpt2-xl) and ``--fp16``
        (half-precision inference; official run was fp32).

    Returns
    -------
    None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--cv_part', default='HeldOutBlocks', choices=['HeldOutBlocks', 'HeldOutTrials'])
    ap.add_argument('--model', default='gpt2-xl')
    ap.add_argument('--fp16', action='store_true', help='fp16 推理（与训练进程共显存时用；官方为 fp32，记录偏差）')
    args = ap.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'loading {args.model} on {device} ({"fp16" if args.fp16 else "fp32"})...')
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    dtype = torch.float16 if args.fp16 else torch.float32
    model = GPT2LMHeadModel.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    val_err = []
    n_sel_same, n_sent = 0, 0
    for data_dir in DATA_DIRS:
        kaldi_dir = os.path.join(ROOT, 'RNNTrainingSteps/Step6_ApplyBigramLM', args.cv_part, 'KaldiOutput', data_dir)
        if not os.path.isdir(kaldi_dir):
            continue
        part = scipy.io.loadmat(os.path.join(ROOT, 'RNNTrainingSteps', 'trainTestPartitions_' + args.cv_part + '.mat'))
        val_idx = part[data_dir + '_test']
        if len(val_idx) == 0:
            continue
        val_idx = val_idx[0, :]
        sentence_dat = scipy.io.loadmat(os.path.join(ROOT, 'Datasets', data_dir, 'sentences.mat'))

        for file_idx in val_idx:
            content = read_kaldi_lattice_file(os.path.join(kaldi_dir, str(int(file_idx)) + '_transcript.txt'), 'string')
            ac_score = read_kaldi_lattice_file(os.path.join(kaldi_dir, str(int(file_idx)) + '_best_acscore.ark'), 'numeric')
            lm_score = read_kaldi_lattice_file(os.path.join(kaldi_dir, str(int(file_idx)) + '_best_lmscore.ark'), 'numeric')

            lm_rescore = gpt2_nll_batch(model, tokenizer, list(content), device)
            new_best = int(np.argmin(ac_score + 2.0 * lm_rescore))
            bigram_best = int(np.argmin(ac_score + 2.0 * lm_score))
            n_sent += 1
            if new_best == bigram_best:
                n_sel_same += 1
            dec_sent = content[new_best]

            true_text = sentence_dat['sentencePrompt'][int(file_idx), 0][0]
            true_text = true_text.replace('>', ' ').replace('~', '.').replace('#', '')
            char_errs = wer(true_text, dec_sent)
            word_errs = wer(true_text.split(), dec_sent.split())
            val_err.append([char_errs, len(true_text), word_errs, len(true_text.split())])

    concat = np.array(val_err)
    cer = 100 * np.sum(concat[:, 0]) / np.sum(concat[:, 1])
    wer_rate = 100 * np.sum(concat[:, 2]) / np.sum(concat[:, 3])
    print(f'{args.cv_part}: GPT-2 重打分 CER={cer:.4f}% WER={wer_rate:.4f}% n={n_sent}')
    print(f'与 bigram 选句相同比例: {n_sel_same}/{n_sent}')

    ref_all = []
    for data_dir in DATA_DIRS:
        p = os.path.join(ROOT, 'RNNTrainingSteps/Step7_GPT2Rescore', args.cv_part, data_dir + '_errCounts.mat')
        if not os.path.isfile(p):
            continue
        dat = scipy.io.loadmat(p)
        part = scipy.io.loadmat(os.path.join(ROOT, 'RNNTrainingSteps', 'trainTestPartitions_' + args.cv_part + '.mat'))
        vi = part[data_dir + '_test']
        if len(vi) == 0:
            continue
        vi = vi[0, :]
        ref_all.append(np.stack([dat['charCounts'][0, vi], dat['charErrors'][0, vi],
                                 dat['wordCounts'][0, vi], dat['wordErrors'][0, vi]], axis=0).T)
    ref = np.squeeze(np.concatenate(ref_all, axis=0))
    mine = concat[:, [1, 0, 3, 2]]
    n_exact = int(np.sum(np.all(mine == ref, axis=1)))
    print(f'与包内 Step7 errCounts 逐句完全一致: {n_exact}/{n_sent}')
    np.savetxt(os.path.join(OUT_DIR, f'm1c_rescore_{args.cv_part}.csv'),
               np.column_stack([mine, ref]), delimiter=',',
               header='charCounts,charErrors,wordCounts,wordErrors,refCC,refCE,refWC,refWE')


if __name__ == '__main__':
    main()
