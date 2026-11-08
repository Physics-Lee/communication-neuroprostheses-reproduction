# -*- coding: utf-8 -*-
"""M1b-B 批量：用 WSL C++ 解码器（latgen_lazy2，惰性组合 HCL×G）自解码 val 句。

每句调用：wsl.exe ~/bin/latgen_lazy2 HCL.fst G.fst model kaldiMat_i.txt --graphw 2.0
选句等价：argmin(ac + 2.0*graph) 的 1-best（graphw 烘入搜索，官方 nbest 后选句的路径级等价）。
transcript：发射弧 phone 游程折叠 → 过滤 '.'/'<ctc>' → <space>→空格 → 去首尾。

产出：result/m1b_own_decode_<cvPart>.csv（逐句 decText/refText/charErrs/...）+ 终端汇总。
对照官方：HeldOutBlocks 1.69/6.10，HeldOutTrials 0.80/3.64。
"""
import argparse
import csv
import os
import subprocess
import sys

import numpy as np
import scipy.io

REPO = r'.\20210512_willett'
ROOT = os.path.join(REPO, 'data', '20210512_willett', 'handwritingBCIData')
LM = os.path.join(ROOT, 'BigramLM')

SESSIONS = ['t5.2019.05.08', 't5.2019.11.25', 't5.2019.12.09', 't5.2019.12.11', 't5.2019.12.18',
            't5.2019.12.20', 't5.2020.01.06', 't5.2020.01.08', 't5.2020.01.13', 't5.2020.01.15']

CHARS = {0: '', 1: '', 2: ' ', 3: "'", 4: ',', 5: '.', 6: '?', 33: ''}
for _i in range(7, 33):
    CHARS[_i] = chr(ord('a') + _i - 7)


def tid_to_phone(tid):
    """tid→phone：phone p(2..32) 的 tid 块为 5(p-2)+1..5(p-2)+5；phone 33 (<ctc>) 为 156,157。

    Parameters
    ----------
    tid : int
        Transition id emitted by the decoder (1-based).

    Returns
    -------
    int
        Phone id in the characterDefinitions indexing (2..32 map to
        characters, 33 is the ``<ctc>`` silence phone).
    """
    if tid >= 156:
        return 33
    return 2 + (tid - 1) // 5


def tid_is_state0(tid):
    """state0 的 3 个 tid（块内位置 1..3）；state1 tid（位置 4,5）读 pdf0 列=时长填充。
    字符实例 = 一个 state0 tid 游程；双字母（pp/ee）靠两次 state0 游程保留。

    Parameters
    ----------
    tid : int
        Transition id emitted by the decoder.

    Returns
    -------
    bool
        True if the tid belongs to a state0 slot (character-emitting);
        False for state1 slots and for the ``<ctc>`` silence tids.
    """
    if tid >= 156:
        return False  # <ctc> 视为静默
    return (tid - 1) % 5 < 3


def tids_to_text(tids):
    """新字符实例 = state0 tid 且 phone 相对当前 s0 游程发生变化。
    字符可经 s0→final 直接跳出（无 state1 间隔），相邻字符 s0 游程相接，
    因此按 phone 变化判定而不是按 s1→s0 转换；双字母靠中间的 state1/ctc 游程隔开。

    Parameters
    ----------
    tids : list of int
        Sequence of transition ids from the decoder's 1-best path.

    Returns
    -------
    str
        Decoded character string with blank/``<ctc>``/non-character phones
        dropped and surrounding spaces stripped.
    """
    out = []
    cur_phone = None
    for t in tids:
        if not tid_is_state0(t):
            cur_phone = None
            continue
        ph = tid_to_phone(t)
        if ph != cur_phone:
            if CHARS.get(ph, ''):
                out.append(CHARS[ph])
            cur_phone = ph
    return ''.join(out).strip(' ')


def wer(r, h):
    """Levenshtein edit distance between two token sequences.

    Works for both character lists and word lists, so it serves as both
    charErrs and wordErrs depending on the inputs.

    Parameters
    ----------
    r : list or str
        Reference tokens (ground truth).
    h : list or str
        Hypothesis tokens (decoder output).

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
            d[i, j] = d[i - 1, j - 1] if r[i - 1] == h[j - 1] else min(d[i - 1, j - 1] + 1, d[i - 1, j] + 1, d[i, j - 1] + 1)
    return int(d[len(r), len(h)])


def decode_one(mat_path):
    """返回 (ac, graph, text) 或 None。

    Runs the WSL ``latgen_lazy2`` decoder on a single kaldiMat text matrix
    with the graph weight 2.0 baked into the search, then folds the
    emitted tid sequence back into characters via ``tids_to_text``.

    Parameters
    ----------
    mat_path : str
        Windows path of a ``kaldiMat_<sidx>.txt`` matrix file; it is
        translated to the WSL ``/mnt/...`` form for the subprocess.

    Returns
    -------
    tuple of (float, float, str) or None
        ``(acoustic_score, graph_score, decoded_text)`` on success, or None
        when the decoder produced no output line.

    Raises
    ------
    subprocess.TimeoutExpired
        If the WSL decoder exceeds its 300 s timeout.
    """
    cmd = ['wsl.exe', '-e', 'bash', '-c',
           '~/bin/latgen_lazy2 %s/HCL.fst %s/G.fst %s/model %s --graphw 2.0 2>/dev/null'
           % (LM.replace('\\', '/').replace('D:', '/mnt/d'),
              LM.replace('\\', '/').replace('D:', '/mnt/d'),
              LM.replace('\\', '/').replace('D:', '/mnt/d'),
              mat_path.replace('\\', '/').replace('D:', '/mnt/d'))]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    line = r.stdout.strip()
    if not line:
        return None
    parts = line.split('\t')
    tids = [int(x) for x in parts[2].split()]
    return float(parts[0]), float(parts[1]), tids_to_text(tids)


def read_ref(kout, sidx):
    """Read the official Kaldi n-best selection for one sentence.

    Loads the official Step6 KaldiOutput transcript/acoustic/LM score files
    for sentence ``sidx`` and reproduces the official selection rule
    ``argmin(ac + 2.0*lm)``.

    Parameters
    ----------
    kout : str
        Directory ``KaldiOutput/<session>`` holding the official per-file
        ``<sidx>_transcript.txt``, ``<sidx>_best_acscore.ark`` and
        ``<sidx>_best_lmscore.ark``.
    sidx : int
        Sentence index within the session.

    Returns
    -------
    str or None
        The officially selected transcript string, or None when the needed
        files are missing (e.g. LM score file absent).
    """
    tr = os.path.join(kout, f'{sidx}_transcript.txt')
    acf = os.path.join(kout, f'{sidx}_best_acscore.ark')
    lmf = os.path.join(kout, f'{sidx}_best_lmscore.ark')
    if not (os.path.isfile(tr) and os.path.isfile(acf)):
        return None
    texts = []
    for l in open(tr, encoding='utf-8').read().strip().split('\n'):
        toks = l.split(' ')
        del toks[0]
        texts.append(''.join(' ' if t == '<space>' else '' if t == '\n' else t for t in toks))
    acs = [float(l.split()[1]) for l in open(acf).read().strip().split('\n')]
    lms = [float(l.split()[1]) for l in open(lmf).read().strip().split('\n')]
    return texts[int(np.argmin(np.array(acs) + 2.0 * np.array(lms)))]


def main():
    """Decode all validation sentences with the WSL decoder and report CER/WER.

    For every session in the chosen partition, iterates the validation
    kaldiMat files, decodes each with ``decode_one``, computes char/word
    errors against the sentence prompt, compares against the official
    selection via ``read_ref``, prints per-sentence progress and the
    corpus-level summary (CER/WER and exact-match count vs. official), and
    writes one row per sentence to ``result/m1b_own_decode_<cvPart>.csv``.

    Parameters
    ----------
    None
        ``--cvPart`` (HeldOutBlocks or HeldOutTrials) is required on the
        command line.

    Returns
    -------
    None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--cvPart', choices=['HeldOutBlocks', 'HeldOutTrials'], required=True)
    cli = vars(ap.parse_args())
    cv = cli['cvPart']

    rows = []
    val_err = []
    n_match, n_tot = 0, 0
    import time
    t0 = time.time()
    for sess in SESSIONS:
        mats_dir = os.path.join(ROOT, 'RNNTrainingSteps/Step6_ApplyBigramLM', cv, 'KaldiMatrices', sess)
        if not os.path.isdir(mats_dir):
            continue
        part = scipy.io.loadmat(os.path.join(ROOT, 'RNNTrainingSteps', 'trainTestPartitions_' + cv + '.mat'))
        vi = part[sess + '_test']
        if len(vi) == 0:
            continue
        val_idx = vi[0, :]
        sdat = scipy.io.loadmat(os.path.join(ROOT, 'Datasets', sess, 'sentences.mat'))
        kout = os.path.join(ROOT, 'RNNTrainingSteps/Step6_ApplyBigramLM', cv, 'KaldiOutput', sess)
        mats = sorted([f for f in os.listdir(mats_dir) if f.startswith('kaldiMat_') and int(f[9:-4]) in set(val_idx.tolist())],
                      key=lambda x: int(x[9:-4]))
        for mf in mats:
            sidx = int(mf[9:-4])
            dec = decode_one(os.path.join(mats_dir, mf))
            if dec is None:
                print(f'{sess} #{sidx}: DECODE_FAIL', flush=True)
                continue
            ac, graph, text = dec
            true_text = sdat['sentencePrompt'][sidx, 0][0].replace('>', ' ').replace('~', '.').replace('#', '')
            ce = wer(true_text, text)
            we = wer(true_text.split(), text.split())
            val_err.append([ce, len(true_text), we, len(true_text.split())])
            ref = read_ref(kout, sidx)
            n_tot += 1
            if ref is not None and text == ref:
                n_match += 1
            rows.append([sess, sidx, ac, graph, text, ref or '', ce, len(true_text), we, len(true_text.split())])
            print(f'{sess} #{sidx}: mine={text!r} match={text == ref}', flush=True)
    arr = np.array(val_err)
    cer = 100 * arr[:, 0].sum() / arr[:, 1].sum()
    werr = 100 * arr[:, 2].sum() / arr[:, 3].sum()
    dt = time.time() - t0
    print(f'\n{cv}: 自解码 bigram CER={cer:.4f}% WER={werr:.4f}% n={len(rows)} ({dt/60:.1f} min)')
    print(f'与包内官方选句完全一致: {n_match}/{n_tot}')
    print('官方对照: Blocks 1.69/6.10, Trials 0.80/3.64')
    out = os.path.join(REPO, 'result', f'm1b_own_decode_{cv}.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['session', 'sidx', 'ac', 'graph', 'decText', 'refText', 'charErrs', 'charCounts', 'wordErrs', 'wordCounts'])
        w.writerows(rows)
    print('saved ->', out)


if __name__ == '__main__':
    main()
