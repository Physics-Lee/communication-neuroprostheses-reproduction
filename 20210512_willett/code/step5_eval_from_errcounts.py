# -*- coding: utf-8 -*-
"""M1a：从包内 Step5/Step6/Step7 的 errCounts.mat 重算两划分 x 三档 CER/WER（含 bootstrap 95% CI）。

复刻官方 SummarizeRNNPerformance.ipynb 的逻辑：
- 对每个 cvPart x 档位，聚合 10 个 session 的 charCounts/charErrors/wordCounts/wordErrors（按 partition 的 test 索引）
- CER = 100*sum(charErrors)/sum(charCounts)，WER 同理
- bootstrap 10000 次重采样出 95% CI（这里固定 seed 保证可复现，官方未固定 seed，CI 端点差异应在小数点后两位内）

对齐目标（官方 README，作者原始 run）：
  HeldOutTrials  Raw 2.78 [2.20,3.41] / 12.88 [10.28,15.63]
  HeldOutTrials  Bigram 0.80 [0.44,1.22] / 3.64 [2.11,5.34]
  HeldOutTrials  +GPT-2 0.34 [0.14,0.61] / 1.97 [0.78,3.41]
  HeldOutBlocks  Raw 5.32 [4.81,5.86] / 23.28 [21.27,25.41]
  HeldOutBlocks  Bigram 1.69 [1.32,2.10] / 6.10 [4.97,7.25]
  HeldOutBlocks  +GPT-2 0.90 [0.62,1.23] / 3.21 [2.37,4.11]
"""
import numpy as np
import scipy.io
import os
import sys

ROOT = r'.\20210512_willett\data\20210512_willett\handwritingBCIData'
OUT = r'.\20210512_willett\result'
# 自训模型的 errCounts 目录（infer.py 产物）：用法 python step5_eval_from_errcounts.py <errCounts目录> [cvPart]
# 目录结构需为 <root>/<cvPart>/<session>_errCounts.mat；partition 文件仍读官方包。

DATA_DIRS = ['t5.2019.05.08', 't5.2019.11.25', 't5.2019.12.09', 't5.2019.12.11', 't5.2019.12.18',
             't5.2019.12.20', 't5.2020.01.06', 't5.2020.01.08', 't5.2020.01.13', 't5.2020.01.15']

RESULTS_DIRS = ['RNNTrainingSteps/Step5_RNNInference', 'RNNTrainingSteps/Step6_ApplyBigramLM',
                'RNNTrainingSteps/Step7_GPT2Rescore']
RESULTS_NAMES = ['Raw', 'Bigram LM', 'Bigram LM + GPT-2 Rescore']

OFFICIAL = {  # (cvPart, name): (cer, cer_lo, cer_hi, wer, wer_lo, wer_hi)
    ('HeldOutTrials', 'Raw'): (2.78, 2.20, 3.41, 12.88, 10.28, 15.63),
    ('HeldOutTrials', 'Bigram LM'): (0.80, 0.44, 1.22, 3.64, 2.11, 5.34),
    ('HeldOutTrials', 'Bigram LM + GPT-2 Rescore'): (0.34, 0.14, 0.61, 1.97, 0.78, 3.41),
    ('HeldOutBlocks', 'Raw'): (5.32, 4.81, 5.86, 23.28, 21.27, 25.41),
    ('HeldOutBlocks', 'Bigram LM'): (1.69, 1.32, 2.10, 6.10, 4.97, 7.25),
    ('HeldOutBlocks', 'Bigram LM + GPT-2 Rescore'): (0.90, 0.62, 1.23, 3.21, 2.37, 4.11),
}

N_RESAMPLES = 10000
SEED = 0


OFFICIAL_ROOT = r'.\20210512_willett\data\20210512_willett\handwritingBCIData'


def summarize(cv_part, result_idx, rng, flat=False):
    all_err_counts = []
    for data_dir in DATA_DIRS:
        if flat:
            file_path = os.path.join(ROOT, data_dir + '_errCounts.mat')
        else:
            file_path = os.path.join(ROOT, RESULTS_DIRS[result_idx], cv_part, data_dir + '_errCounts.mat')
        if not os.path.isfile(file_path):
            continue
        dat = scipy.io.loadmat(file_path)
        part = scipy.io.loadmat(os.path.join(OFFICIAL_ROOT, 'RNNTrainingSteps', 'trainTestPartitions_' + cv_part + '.mat'))
        val_idx = part[data_dir + '_test']
        if len(val_idx) == 0:
            continue
        val_idx = val_idx[0, :]
        all_err_counts.append(np.stack([dat['charCounts'][0, val_idx],
                                        dat['charErrors'][0, val_idx],
                                        dat['wordCounts'][0, val_idx],
                                        dat['wordErrors'][0, val_idx]], axis=0).T)
    if not all_err_counts:
        return None
    concat = np.squeeze(np.concatenate(all_err_counts, axis=0))
    cer = 100 * (np.sum(concat[:, 1]) / np.sum(concat[:, 0]))
    wer = 100 * (np.sum(concat[:, 3]) / np.sum(concat[:, 2]))

    rates = np.zeros([N_RESAMPLES, 2])
    n = concat.shape[0]
    for k in range(N_RESAMPLES):
        idx = rng.randint(0, n, n)
        rates[k, 0] = 100 * (np.sum(concat[idx, 1]) / np.sum(concat[idx, 0]))
        rates[k, 1] = 100 * (np.sum(concat[idx, 3]) / np.sum(concat[idx, 2]))
    char_ci = np.percentile(rates[:, 0], [2.5, 97.5])
    word_ci = np.percentile(rates[:, 1], [2.5, 97.5])
    return dict(n_sentences=n, cer=float(cer), cer_ci=char_ci.tolist(),
                wer=float(wer), wer_ci=word_ci.tolist())


def main():
    rng = np.random.RandomState(SEED)
    root = sys.argv[1] if len(sys.argv) > 1 else OFFICIAL_ROOT
    only_part = sys.argv[2] if len(sys.argv) > 2 else None
    flat = root != OFFICIAL_ROOT  # 自训模型：平铺 <root>/<session>_errCounts.mat，只算 Raw 档
    global ROOT
    ROOT = root
    lines = []
    tiers = [(0, 'Raw')] if flat else list(enumerate(RESULTS_NAMES))
    parts = [only_part] if only_part else ['HeldOutBlocks', 'HeldOutTrials']
    print(f'{"partition":14s} {"tier":28s} {"CER% (95%CI)":26s} {"WER% (95%CI)":26s} official(CER/WER)')
    for cv_part in parts:
        for ri, name in tiers:
            r = summarize(cv_part, ri, rng, flat=flat)
            if r is None:
                print(f'{cv_part:14s} {name:28s} MISSING')
                continue
            o = OFFICIAL[(cv_part, 'Raw' if flat else name)]
            dcer = r['cer'] - o[0]
            dwer = r['wer'] - o[3]
            cer_s = '%.2f [%.2f, %.2f]' % (r['cer'], r['cer_ci'][0], r['cer_ci'][1])
            wer_s = '%.2f [%.2f, %.2f]' % (r['wer'], r['wer_ci'][0], r['wer_ci'][1])
            print(f'{cv_part:14s} {name:28s} {cer_s:26s} {wer_s:26s} {o[0]:.2f}/{o[3]:.2f}  dCER={dcer:+.2f} dWER={dwer:+.2f} n={r["n_sentences"]}')
            lines.append(f'{cv_part},{name},{r["cer"]:.4f},{r["cer_ci"][0]:.4f},{r["cer_ci"][1]:.4f},{r["wer"]:.4f},{r["wer_ci"][0]:.4f},{r["wer_ci"][1]:.4f},{r["n_sentences"]}')
    suffix = '_ours' if flat else ''
    with open(os.path.join(OUT, f'eval_from_errcounts{suffix}.csv'), 'w') as f:
        f.write('cvPart,tier,CER,CER_lo,CER_hi,WER,WER_lo,WER_hi,nSentences\n')
        f.write('\n'.join(lines) + '\n')
    print(f'\nsaved -> result/eval_from_errcounts{suffix}.csv')


if __name__ == '__main__':
    main()
