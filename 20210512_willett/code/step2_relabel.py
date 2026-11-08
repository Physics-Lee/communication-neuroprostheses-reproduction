# -*- coding: utf-8 -*-
"""M2：Step2 HMM 强制对齐标注重跑 + 与包内官方标签的一致率对账。

官方 Step2_hmmDataLabeling.ipynb 流程（labelDataset + constructRNNTargets，纯 numpy/scipy/sklearn），
重跑产物写到 RNNTrainingSteps/Step2_HMMLabels_rerun/<cvPart>/（不覆盖包内官方标签）。

对账口径：
- letterStarts：逐句逐字符起点 |diff| 中位数与 ≤2 帧一致率
- charProbTarget argmax：与官方标签逐帧 argmax 一致率（限 ignoreErrorHere==0 区域）
- charStartTarget：逐帧 >0.5 一致率
- ignoreErrorHere：逐帧一致率

用法：
    python step2_relabel.py                     # 全部 10 session × 2 partition
    python step2_relabel.py --dataDir t5.2019.05.08 --cvPart HeldOutTrials
"""
import argparse
import os
import sys
import types

import numpy as np
import scipy.io

REPO = r'.\20210512_willett'
OFFICIAL_REPO = os.path.join(REPO, r'open_source_code_for_reference\20210512_willett_handwritingBCI')
sys.path.insert(0, OFFICIAL_REPO)
# code/ 必须排在官方仓库之前，官方 dataLabelingStep 才会 import 到本仓的
# forcedAlignmentHMM numpy2 垫片模块（spawn worker 同理）。
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from characterDefinitions import getHandwritingCharacterDefinitions  # noqa: E402
from dataLabelingStep import labelDataset, constructRNNTargets  # noqa: E402
# numpy 2.x 兼容：forcedAlignmentHMM 用 code/ 下的同名垫片模块（spawn worker 也能命中），
# 见 code/forcedAlignmentHMM.py 头注释。

ROOT = os.path.join(REPO, 'data', '20210512_willett', 'handwritingBCIData')
OUT_ROOT = os.path.join(ROOT, 'RNNTrainingSteps', 'Step2_HMMLabels_rerun')
REPORT = os.path.join(REPO, 'result', 'm2_step2_relabel_report.csv')

DATA_DIRS = ['t5.2019.05.08', 't5.2019.11.25', 't5.2019.12.09', 't5.2019.12.11', 't5.2019.12.18',
             't5.2019.12.20', 't5.2020.01.06', 't5.2020.01.08', 't5.2020.01.13', 't5.2020.01.15']
CV_PARTS = ['HeldOutBlocks', 'HeldOutTrials']


def compare(ref_path, my_path):
    """Compare rerun Step2 HMM labels against the official packaged labels.

    Parameters
    ----------
    ref_path : str
        Path to the official ``<dataDir>_timeSeriesLabels.mat`` under
        ``RNNTrainingSteps/Step2_HMMLabels``.
    my_path : str
        Path to the rerun labels mat written to ``Step2_HMMLabels_rerun``.

    Returns
    -------
    dict
        Agreement statistics with keys:
        ``'letterStarts_medDiff_bins'`` (median |diff| of per-character
        starts, in time bins, over characters positive in both files),
        ``'letterStarts_within2_rate'`` (fraction of those within 2 bins),
        ``'letterStarts_n'`` (number of compared characters),
        ``'charProb_argmax_agree'`` (frame-wise argmax agreement of
        charProbTarget), ``'charStart_agree'`` (frame-wise >0.5 agreement of
        charStartTarget) and ``'ignoreError_agree'`` (frame-wise >0.5
        agreement of ignoreErrorHere).
    """
    ref = scipy.io.loadmat(ref_path)
    mine = scipy.io.loadmat(my_path)
    out = {}

    ls_r, ls_m = ref['letterStarts'], mine['letterStarts']
    diff = np.abs(ls_r.astype(float) - ls_m.astype(float))
    valid = (ls_r > 0) & (ls_m > 0)
    out['letterStarts_medDiff_bins'] = float(np.median(diff[valid])) if valid.any() else np.nan
    out['letterStarts_within2_rate'] = float(np.mean(diff[valid] <= 2)) if valid.any() else np.nan
    out['letterStarts_n'] = int(valid.sum())

    both = ~np.isnan(ref['charProbTarget'].max(axis=2))  # 全部有效
    am_r = np.argmax(ref['charProbTarget'], axis=2)
    am_m = np.argmax(mine['charProbTarget'], axis=2)
    out['charProb_argmax_agree'] = float(np.mean(am_r[both] == am_m[both]))

    cs_r, cs_m = ref['charStartTarget'] > 0.5, mine['charStartTarget'] > 0.5
    out['charStart_agree'] = float(np.mean(cs_r == cs_m))

    ig_r, ig_m = ref['ignoreErrorHere'] > 0.5, mine['ignoreErrorHere'] > 0.5
    out['ignoreError_agree'] = float(np.mean(ig_r == ig_m))
    return out


def main():
    """Rerun Step2 HMM labeling for selected sessions/partitions and log agreement.

    For each requested data directory and cross-validation partition, runs
    the official ``labelDataset`` + ``constructRNNTargets`` pipeline on the
    Step1 warped cubes, saves the labels to
    ``RNNTrainingSteps/Step2_HMMLabels_rerun/<cvPart>/`` (skipping files
    that already exist), compares them against the official labels, prints
    per-session agreement stats, and appends a summary row per
    (dataDir, cvPart) to ``result/m2_step2_relabel_report.csv``.

    Parameters
    ----------
    None
        Options come from the command line: ``--dataDir`` restricts to one
        session directory and ``--cvPart`` to one partition; by default all
        10 sessions x 2 partitions are processed.

    Returns
    -------
    None
    """
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataDir', default=None)
    ap.add_argument('--cvPart', default=None, choices=[None, 'HeldOutBlocks', 'HeldOutTrials'])
    cli = vars(ap.parse_args())
    data_dirs = [cli['dataDir']] if cli['dataDir'] else DATA_DIRS
    cv_parts = [cli['cvPart']] if cli['cvPart'] else CV_PARTS

    char_def = getHandwritingCharacterDefinitions()
    os.makedirs(OUT_ROOT, exist_ok=True)
    rows = []
    header_written = os.path.isfile(REPORT)

    for data_dir in data_dirs:
        print('Labeling ' + data_dir, flush=True)
        sentence_dat = scipy.io.loadmat(os.path.join(ROOT, 'Datasets', data_dir, 'sentences.mat'))
        single_letter_dat = scipy.io.loadmat(os.path.join(ROOT, 'Datasets', data_dir, 'singleLetters.mat'))
        tw_cubes = scipy.io.loadmat(os.path.join(ROOT, 'RNNTrainingSteps', 'Step1_TimeWarping', data_dir + '_warpedCubes.mat'))

        for x in range(sentence_dat['sentencePrompt'].shape[0]):
            sentence_dat['sentencePrompt'][x, 0][0] = sentence_dat['sentencePrompt'][x, 0][0].replace('#', '')

        sentences = sentence_dat['sentencePrompt'][:, 0]
        n_tbins = sentence_dat['neuralActivityCube'].shape[1]

        for cv_part in cv_parts:
            out_dir = os.path.join(OUT_ROOT, cv_part)
            out_path = os.path.join(out_dir, data_dir + '_timeSeriesLabels.mat')
            if os.path.isfile(out_path):
                print(f'  {cv_part}: 已存在，跳过', flush=True)
                continue
            part = scipy.io.loadmat(os.path.join(ROOT, 'RNNTrainingSteps', 'trainTestPartitions_' + cv_part + '.mat'))
            letter_starts, letter_durations, blank_windows = labelDataset(
                sentence_dat, single_letter_dat, tw_cubes,
                part[data_dir + '_train'], part[data_dir + '_test'], char_def)
            char_start_target, char_prob_target, ignore_error_here = constructRNNTargets(
                letter_starts, letter_durations, n_tbins, sentences, char_def)

            out_dir = os.path.join(OUT_ROOT, cv_part)
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, data_dir + '_timeSeriesLabels.mat')
            scipy.io.savemat(out_path, {
                'letterStarts': letter_starts, 'letterDurations': letter_durations,
                'charStartTarget': char_start_target.astype(np.float32),
                'charProbTarget': char_prob_target.astype(np.float32),
                'ignoreErrorHere': ignore_error_here.astype(np.float32),
                'blankWindows': blank_windows,
                'timeBinsPerSentence': sentence_dat['numTimeBinsPerSentence']})

            ref_path = os.path.join(ROOT, 'RNNTrainingSteps', 'Step2_HMMLabels', cv_part, data_dir + '_timeSeriesLabels.mat')
            stats = compare(ref_path, out_path)
            print(f'  {cv_part}: letterStarts中位差={stats["letterStarts_medDiff_bins"]:.1f} bins '
                  f'(≤2帧一致率 {stats["letterStarts_within2_rate"]*100:.2f}%, n={stats["letterStarts_n"]}), '
                  f'charProb argmax一致率={stats["charProb_argmax_agree"]*100:.2f}%, '
                  f'charStart一致率={stats["charStart_agree"]*100:.2f}%, '
                  f'ignoreError一致率={stats["ignoreError_agree"]*100:.2f}%', flush=True)
            rows.append([data_dir, cv_part, stats['letterStarts_medDiff_bins'], stats['letterStarts_within2_rate'],
                         stats['letterStarts_n'], stats['charProb_argmax_agree'], stats['charStart_agree'],
                         stats['ignoreError_agree']])

    with open(REPORT, 'a' if header_written else 'w') as f:
        if not header_written:
            f.write('dataDir,cvPart,letterStarts_medDiff_bins,letterStarts_within2_rate,letterStarts_n,'
                    'charProb_argmax_agree,charStart_agree,ignoreError_agree\n')
        for r in rows:
            f.write(','.join(str(x) for x in r) + '\n')
    print('report ->', REPORT)


if __name__ == '__main__':
    main()
