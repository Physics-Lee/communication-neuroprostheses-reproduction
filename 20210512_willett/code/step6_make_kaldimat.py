# -*- coding: utf-8 -*-
"""Step6 前置：把自训模型 step5_inference 的 RNN 输出转成 Kaldi 概率矩阵（kaldiMat_*.txt）。

逻辑 = 官方 Step6_bigramLanguageModel.ipynb cell 2（直接 import 官方 rnnEval.rnnOutputToKaldiMatrices，
内含 5 倍降采样 + 用 charStart 造 fake CTC blank + idxToKaldi 重排）。为省磁盘，先全量生成
（保持句号索引不变），再删掉非 val 句（解码驱动按 val 索引取文件）。

用法：
    python step6_make_kaldimat.py --inferDir <step5_inference 目录> --cvPart HeldOutTrials
输出：
    <runs 根>/step6_kaldimat/HeldOutTrials/<session>/kaldiMat_<i>.txt（仅 val 句）
"""
import argparse
import os
import sys

import numpy as np
import scipy.io

REPO = r'.\20210512_willett'
OFFICIAL_REPO = os.path.join(REPO, r'open_source_code_for_reference\20210512_willett_handwritingBCI')
sys.path.insert(0, OFFICIAL_REPO)

from characterDefinitions import getHandwritingCharacterDefinitions  # noqa: E402  官方模块，只读引用
from rnnEval import rnnOutputToKaldiMatrices  # noqa: E402  官方模块，只读引用

ROOT = os.path.join(REPO, 'data', '20210512_willett', 'handwritingBCIData')
SESSIONS = ['t5.2019.05.08', 't5.2019.11.25', 't5.2019.12.09', 't5.2019.12.11', 't5.2019.12.18',
            't5.2019.12.20', 't5.2020.01.06', 't5.2020.01.08', 't5.2020.01.13', 't5.2020.01.15']


def main():
    """Convert self-trained Step5 RNN outputs into Kaldi probability matrices.

    For each session with an ``<sess>_inferenceOutputs.mat`` in the given
    inference directory, calls the official ``rnnOutputToKaldiMatrices``
    (5x downsampling, fake CTC blank from charStart, idxToKaldi reordering)
    to write ``kaldiMat_<i>.txt`` files under
    ``step6_kaldimat/<cvPart>/<session>/``, then deletes the non-validation
    sentence files to save disk while keeping val indices stable.

    Parameters
    ----------
    None
        Required command-line options: ``--inferDir`` (the step5_inference
        directory produced by infer.py) and ``--cvPart``.

    Returns
    -------
    None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--inferDir', required=True, help='infer.py 输出的 step5_inference 目录')
    ap.add_argument('--cvPart', choices=['HeldOutBlocks', 'HeldOutTrials'], required=True)
    cli = vars(ap.parse_args())

    char_def = getHandwritingCharacterDefinitions()
    part = scipy.io.loadmat(os.path.join(ROOT, 'RNNTrainingSteps', 'trainTestPartitions_' + cli['cvPart'] + '.mat'))

    out_root = os.path.join(os.path.dirname(os.path.normpath(cli['inferDir'])), 'step6_kaldimat', cli['cvPart'])
    for sess in SESSIONS:
        infer_f = os.path.join(cli['inferDir'], sess + '_inferenceOutputs.mat')
        if not os.path.isfile(infer_f):
            continue
        kaldi_dir = os.path.join(out_root, sess)
        os.makedirs(kaldi_dir, exist_ok=True)

        outputs = scipy.io.loadmat(infer_f)['outputs']
        sentence_dat = scipy.io.loadmat(os.path.join(ROOT, 'Datasets', sess, 'sentences.mat'))

        rnnOutputToKaldiMatrices(outputs,
                                 sentence_dat['numTimeBinsPerSentence'] / 2 + 50,
                                 char_def,
                                 kaldi_dir + os.sep)

        vi = set(part[sess + '_test'][0, :].tolist()) if len(part[sess + '_test']) else set()
        n_keep = 0
        for f in os.listdir(kaldi_dir):
            if f.startswith('kaldiMat_') and int(f[9:-4]) not in vi:
                os.remove(os.path.join(kaldi_dir, f))
            elif f.startswith('kaldiMat_'):
                n_keep += 1
        print(f'{sess}: 生成 {outputs.shape[0]} 句，保留 val {n_keep} 句', flush=True)
    print('kaldiMat 完成 ->', out_root)


if __name__ == '__main__':
    main()
