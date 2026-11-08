# -*- coding: utf-8 -*-
"""M4 组件：Fig. 2c 风格 per-session（per-day）raw CER 柱状图——官方 checkpoint vs 自训模型并排。

数据源：官方 Step5_RNNInference errCounts（<cvPart>/<sess>_errCounts.mat，两级目录）
      + 自训 runs/<run>_synth/step5_inference errCounts（平铺目录）。
口径：raw 档（无语言模型），val 集，按 session 聚合 CER。

用法：
    python fig_perday_cer_ours.py            # 产出 result/figs/perday_cer_official_vs_ours.png
"""
import os

import numpy as np
import scipy.io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

REPO = r'.\20210512_willett'
ROOT = os.path.join(REPO, 'data', '20210512_willett', 'handwritingBCIData')
RUNS = os.path.join(REPO, 'data', '20210512_willett', 'runs')

SESSIONS = ['t5.2019.05.08', 't5.2019.11.25', 't5.2019.12.09', 't5.2019.12.11', 't5.2019.12.18',
            't5.2019.12.20', 't5.2020.01.06', 't5.2020.01.08', 't5.2020.01.13', 't5.2020.01.15']


def per_session_cer(err_dir, cv_part, flat):
    """Compute per-session raw character error rate from Step5 errCounts files.

    Parameters
    ----------
    err_dir : str
        Directory holding the ``<session>_errCounts.mat`` files (produced by
        Step5_RNNInference or the self-trained step5_inference run).
    cv_part : str
        Cross-validation partition name, e.g. ``'HeldOutBlocks'`` or
        ``'HeldOutTrials'``. Selects the ``trainTestPartitions_<cv_part>.mat``
        split and, when ``flat`` is False, the per-partition subdirectory.
    flat : bool
        If True, read errCounts files directly under ``err_dir``; if False,
        read them under the ``err_dir/<cv_part>`` two-level layout used by
        the official Step5_RNNInference output.

    Returns
    -------
    dict
        Mapping from session name (e.g. ``'t5.2019.05.08'``) to raw CER in
        percent for every session whose errCounts file exists and whose test
        partition is non-empty. Sessions with zero total reference characters
        map to ``np.nan``; missing files/sessions are skipped entirely.
    """
    out = {}
    for sess in SESSIONS:
        p = os.path.join(err_dir, sess + '_errCounts.mat') if flat \
            else os.path.join(err_dir, cv_part, sess + '_errCounts.mat')
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
    """Render and save the Fig. 2c-style per-session raw CER comparison figure.

    Draws one panel per cross-validation partition (HeldOutBlocks,
    HeldOutTrials), each showing grouped bars of raw val CER per session for
    the official checkpoint versus the self-trained model, with per-bar value
    labels. The figure is saved to
    ``result/figs/perday_cer_official_vs_ours.png`` (directories are created
    as needed).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    for ax, cv in zip(axes, ['HeldOutBlocks', 'HeldOutTrials']):
        off = per_session_cer(os.path.join(ROOT, 'RNNTrainingSteps', 'Step5_RNNInference'), cv, flat=False)
        ours = per_session_cer(os.path.join(RUNS, cv + '_synth', 'step5_inference'), cv, flat=True)
        sess = [s for s in SESSIONS if s in off and s in ours]
        x = np.arange(len(sess))
        ax.bar(x - 0.2, [off[s] for s in sess], width=0.4, color='#3b6fb6', label='official ckpt')
        ax.bar(x + 0.2, [ours[s] for s in sess], width=0.4, color='#e2894a', label='self-trained')
        for i, s in enumerate(sess):
            ax.text(i - 0.2, off[s] + 0.1, f'{off[s]:.1f}', ha='center', fontsize=7)
            ax.text(i + 0.2, ours[s] + 0.1, f'{ours[s]:.1f}', ha='center', fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('t5.', '') for s in sess], rotation=45, ha='right', fontsize=8)
        ax.set_title(f'{cv} (raw CER, val)')
        ax.set_ylabel('Character error rate (%)')
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(REPO, 'result', 'figs', 'perday_cer_official_vs_ours.png')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=200)
    print('saved', out)


if __name__ == '__main__':
    main()
