# -*- coding: utf-8 -*-
"""M3/M4 组件：用自研 C++ 解码器（latgen_lazy2）对**自训模型**的 kaldiMat 做 bigram 解码。

与 step6_decode_batch.py（官方 RNN 输出上的 M1b-B）同口径：argmin(ac+2*graph) 的 1-best、
transcript 由 state0 tid 游程折叠、CER/WER 保留空格（与官方 Step6 errCounts 口径一致）。
区别：读 runs/<run>/step6_kaldimat/<cvPart>/<sess>/，无官方 n-best 可对照，只出汇总。

用法：
    python step6_decode_ours.py --runName HeldOutTrials_synth --cvPart HeldOutTrials
产出：result/m3_own_decode_<cvPart>.csv + 终端汇总（对照官方 bigram 档 Trials 0.80/3.64）
"""
import argparse
import csv
import os
import time

import numpy as np
import scipy.io

from step6_decode_batch import SESSIONS, decode_one, wer  # noqa: E402  复用已验证的解码/求距函数

REPO = r'.\20210512_willett'
ROOT = os.path.join(REPO, 'data', '20210512_willett', 'handwritingBCIData')


def main():
    """Bigram-decode the self-trained model's kaldiMats and write the M3 CSV.

    For every session of the chosen partition that has kaldiMats under
    ``runs/<runName>/step6_kaldimat/<cvPart>/``, decodes each validation
    sentence with the WSL ``latgen_lazy2`` 1-best (reusing the verified
    helpers from ``step6_decode_batch``), computes char/word errors against
    the sentence prompt (spaces kept, matching the official Step6 errCounts
    convention), prints the corpus-level CER/WER summary against the
    official bigram numbers, and writes one row per sentence to
    ``result/m3_own_decode_<cvPart>.csv``.

    Parameters
    ----------
    None
        Required command-line options: ``--runName`` (runs directory name
        containing ``step6_kaldimat``) and ``--cvPart``.

    Returns
    -------
    None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--runName', required=True, help='runs 下的运行目录名（含 step6_kaldimat）')
    ap.add_argument('--cvPart', choices=['HeldOutBlocks', 'HeldOutTrials'], required=True)
    cli = vars(ap.parse_args())
    cv = cli['cvPart']

    mats_root = os.path.join(REPO, 'data', '20210512_willett', 'runs', cli['runName'], 'step6_kaldimat', cv)
    part = scipy.io.loadmat(os.path.join(ROOT, 'RNNTrainingSteps', 'trainTestPartitions_' + cv + '.mat'))

    rows = []
    t0 = time.time()
    for sess in SESSIONS:
        mats_dir = os.path.join(mats_root, sess)
        if not os.path.isdir(mats_dir):
            continue
        vi = part[sess + '_test']
        if len(vi) == 0:
            continue
        val_idx = vi[0, :]
        sdat = scipy.io.loadmat(os.path.join(ROOT, 'Datasets', sess, 'sentences.mat'))
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
            rows.append([sess, sidx, ac, graph, text, true_text, ce, len(true_text), we, len(true_text.split())])
            print(f'{sess} #{sidx}: mine={text!r}', flush=True)

    arr = np.array([[r[6], r[7], r[8], r[9]] for r in rows])
    cer = 100 * arr[:, 0].sum() / arr[:, 1].sum()
    werr = 100 * arr[:, 2].sum() / arr[:, 3].sum()
    print(f'\n{cv}（自训模型，bigram 1-best）: CER={cer:.4f}% WER={werr:.4f}% n={len(rows)} '
          f'({(time.time()-t0)/60:.1f} min)')
    print('官方对照（官方 RNN + 官方 Kaldi n-best 选句）: Blocks 1.69/6.10, Trials 0.80/3.64')
    out = os.path.join(REPO, 'result', f'm3_own_decode_{cv}.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['session', 'sidx', 'ac', 'graph', 'decText', 'trueText', 'charErrs', 'charCounts', 'wordErrs', 'wordCounts'])
        w.writerows(rows)
    print('saved ->', out)


if __name__ == '__main__':
    main()
