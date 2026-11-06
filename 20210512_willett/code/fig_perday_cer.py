# -*- coding: utf-8 -*-
"""M4 组件：Fig. 2c 风格 per-session（per-day）raw CER 柱状图。

数据源：Step5_RNNInference errCounts（官方 run 或自训 run，--which 选择）。
口径：raw 档（无语言模型），val 集，按 session 聚合 CER。

用法：
    python fig_perday_cer.py --which official --out result/figs/perday_cer_official.png
    python fig_perday_cer.py --which ours --runInferDir <infer.py 输出目录> --out result/figs/perday_cer_ours.png
"""
import argparse
import os

import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

REPO = r'.\20210512_willett'
ROOT = os.path.join(REPO, 'data', '20210512_willett', 'handwritingBCIData')

SESSIONS = ['t5.2019.05.08', 't5.2019.11.25', 't5.2019.12.09', 't5.2019.12.11', 't5.2019.12.18',
            't5.2019.12.20', 't5.2020.01.06', 't5.2020.01.08', 't5.2020.01.13', 't5.2020.01.15']


def per_session_cer(err_dir, cv_part):
    out = {}
    for sess in SESSIONS:
        p = os.path.join(err_dir, cv_part, sess + '_errCounts.mat')
        if not os.path.isfile(p):
            continue
        dat = scipy.io.loadmat(p)
        part = scipy.io.loadmat(os.path.join(ROOT, 'RNNTrainingSteps', 'trainTestPartitions_' + cv_part + '.mat'))
        vi = part[sess + '_test']
        if len(vi) == 0:
            continue
        vi = vi[0, :]
        cc = np.sum(dat['charCounts'][0, vi])
        ce = np.sum(dat['charErrors'][0, vi])
        out[sess] = 100 * ce / cc if cc > 0 else np.nan
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--which', choices=['official', 'ours'], default='official')
    ap.add_argument('--runInferDir', default=None)
    ap.add_argument('--out', required=True)
    cli = vars(ap.parse_args())

    if cli['which'] == 'official':
        base = os.path.join(ROOT, 'RNNTrainingSteps', 'Step5_RNNInference')
    else:
        base = cli['runInferDir']

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, cv_part in zip(axes, ['HeldOutBlocks', 'HeldOutTrials']):
        d = per_session_cer(base, cv_part)
        sess = [s for s in SESSIONS if s in d]
        vals = [d[s] for s in sess]
        labels = [s.replace('t5.', '') for s in sess]
        ax.bar(range(len(vals)), vals, color='#3b6fb6')
        for i, v in enumerate(vals):
            ax.text(i, v + 0.1, f'{v:.1f}', ha='center', fontsize=8)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
        ax.set_title(f'{cv_part} (raw CER, official run)' if cli['which'] == 'official'
                     else f'{cv_part} (raw CER, ours)')
        ax.set_ylabel('Character error rate (%)')
        ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    os.makedirs(os.path.dirname(cli['out']), exist_ok=True)
    plt.savefig(cli['out'], dpi=200)
    print('saved', cli['out'])


if __name__ == '__main__':
    main()
