# -*- coding: utf-8 -*-
"""M1b-A：从包内 Step6 KaldiOutput（n-best 文本 + ac/lm 分数）复现选句与 errCounts。

复刻官方 Step6_bigramLanguageModel.ipynb cell5 逻辑：
- 读 {i}_transcript.txt / {i}_best_acscore.ark / {i}_best_lmscore.ark
- bestIdx = argmin(acScore + 2.0*lmScore)
- 真值替换：'>'→' '，'~'→'.'，'#'→''（保留空格，与 Step5 raw 口径不同）
- charErrs = wer(真值字符列表, 解码字符列表)；wordErrs = wer(split, split)
- 按 partition test 索引聚合 → CER/WER，与包内 Step6 errCounts（经 M1a：Blocks 1.69/6.10，Trials 0.80/3.64）对齐
"""
import os
import numpy as np
import scipy.io

ROOT = r'.\20210512_willett\data\20210512_willett\handwritingBCIData'

DATA_DIRS = ['t5.2019.05.08', 't5.2019.11.25', 't5.2019.12.09', 't5.2019.12.11', 't5.2019.12.18',
             't5.2019.12.20', 't5.2020.01.06', 't5.2020.01.08', 't5.2020.01.13', 't5.2020.01.15']


def wer(r, h):
    """Levenshtein 距离（官方 rnnEval.wer 的直接移植）。"""
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
    """官方 kaldiReadWrite.readKaldiLatticeFile 移植。"""
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


def main():
    for cv_part in ['HeldOutBlocks', 'HeldOutTrials']:
        val_err_counts = []
        for data_dir in DATA_DIRS:
            kaldi_dir = os.path.join(ROOT, 'RNNTrainingSteps/Step6_ApplyBigramLM', cv_part, 'KaldiOutput', data_dir)
            if not os.path.isdir(kaldi_dir):
                continue
            sentence_dat = scipy.io.loadmat(os.path.join(ROOT, 'Datasets', data_dir, 'sentences.mat'))
            part = scipy.io.loadmat(os.path.join(ROOT, 'RNNTrainingSteps', 'trainTestPartitions_' + cv_part + '.mat'))
            val_idx = part[data_dir + '_test']
            if len(val_idx) == 0:
                continue
            val_idx = val_idx[0, :]

            n_files = int(len(os.listdir(kaldi_dir)) / 9)
            for file_idx in range(n_files):
                content = read_kaldi_lattice_file(os.path.join(kaldi_dir, str(file_idx) + '_transcript.txt'), 'string')
                ac_score = read_kaldi_lattice_file(os.path.join(kaldi_dir, str(file_idx) + '_best_acscore.ark'), 'numeric')
                lm_score = read_kaldi_lattice_file(os.path.join(kaldi_dir, str(file_idx) + '_best_lmscore.ark'), 'numeric')
                best_idx = np.argmin(ac_score + 2.0 * lm_score)
                dec_sent = content[best_idx]

                true_text = sentence_dat['sentencePrompt'][file_idx, 0][0]
                true_text = true_text.replace('>', ' ').replace('~', '.').replace('#', '')

                char_errs = wer(true_text, dec_sent)
                word_errs = wer(true_text.split(), dec_sent.split())

                if file_idx in val_idx:
                    val_err_counts.append([char_errs, len(true_text), word_errs, len(true_text.split())])

        concat = np.array(val_err_counts)
        cer = 100 * np.sum(concat[:, 0]) / np.sum(concat[:, 1])
        wer_rate = 100 * np.sum(concat[:, 2]) / np.sum(concat[:, 3])
        # 与包内 Step6 errCounts 逐句对账
        ref_all = []
        for data_dir in DATA_DIRS:
            p = os.path.join(ROOT, 'RNNTrainingSteps/Step6_ApplyBigramLM', cv_part, data_dir + '_errCounts.mat')
            if not os.path.isfile(p):
                continue
            dat = scipy.io.loadmat(p)
            part = scipy.io.loadmat(os.path.join(ROOT, 'RNNTrainingSteps', 'trainTestPartitions_' + cv_part + '.mat'))
            vi = part[data_dir + '_test']
            if len(vi) == 0:
                continue
            vi = vi[0, :]
            ref_all.append(np.stack([dat['charCounts'][0, vi], dat['charErrors'][0, vi],
                                     dat['wordCounts'][0, vi], dat['wordErrors'][0, vi]], axis=0).T)
        ref = np.squeeze(np.concatenate(ref_all, axis=0))
        mine = concat[:, [1, 0, 3, 2]]  # 统一为 [charCounts, charErrors, wordCounts, wordErrors]
        exact = np.array_equal(mine, ref)
        print(f'{cv_part}: CER={cer:.4f}% WER={wer_rate:.4f}%  n={concat.shape[0]}  与包内 errCounts 逐句完全一致: {exact}')
        if not exact:
            diff = np.where(concat != ref)
            print('  不一致句数:', len(set(zip(*diff[:1]))), '字段差异数:', len(diff[0]))


if __name__ == '__main__':
    main()
