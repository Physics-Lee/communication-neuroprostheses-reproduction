# -*- coding: utf-8 -*-
"""Step3 合成数据增强的移植（官方 Step3_syntheticDataAugmentation.ipynb + makeSyntheticSentences.py）。

与官方的差异（如实记录）：
- 输出格式：官方写 .tfrecord（TF1 API）；本移植写 3 个 .npy（inputs/labels/errW），便于训练时 mmap 顺序读。
- 片段库：官方存 .mat cell 数组；本移植存 .pkl（仅本流水线自用）。
- 种子：官方以 datetime.now().microsecond 为基（每次运行都不同）；本移植保持同模式（非确定性复刻官方行为）。
- 生成算法本体逐行照搬 makeSyntheticSentences.py（片段挑选/笔位约束/时间规整/blank 插入/词表采样）。

用法：
    python step3_make_synthetic.py --phase snippets          # 阶段A：切片段库（快）
    python step3_make_synthetic.py --phase generate          # 阶段B：生成合成句（慢，10进程并行）
    python step3_make_synthetic.py --phase generate --dataDir t5.2019.05.08 --cvPart HeldOutTrials
产出：
    data/20210512_willett/handwritingBCIData/RNNTrainingSteps/Step3_SyntheticSentences/
        <cvPart>/<dataDir>_snippets.pkl
        <cvPart>/<dataDir>_syntheticSentences/bat_{x}_{inputs,labels,errW}.npy  (x=0..19)
磁盘：约 5.2GB × 10 session × 2 partition ≈ 104GB（fp32）。
"""
import argparse
import multiprocessing
import os
import pickle
import sys
import types
from datetime import datetime

import numpy as np
import scipy.io

REPO = r'.\20210512_willett'
OFFICIAL_REPO = os.path.join(REPO, r'open_source_code_for_reference\20210512_willett_handwritingBCI')
sys.path.insert(0, OFFICIAL_REPO)

# 官方 makeSyntheticSentences.py 顶层 import tensorflow（仅用于 tfrecord 写出，本移植不调用）。
# 本环境不装 TF（见计划 D1），这里塞一个惰性 stub 让 numpy 部分可 import。
if 'tensorflow' not in sys.modules:
    _tf_stub = types.ModuleType('tensorflow')
    _tf_stub.__getattr__ = lambda name: (_ for _ in ()).throw(
        AttributeError(f'tensorflow stub: {name} 不可用（本移植不使用 TF 功能）'))
    sys.modules['tensorflow'] = _tf_stub

from characterDefinitions import getHandwritingCharacterDefinitions  # noqa: E402
from dataPreprocessing import normalizeSentenceDataCube  # noqa: E402
from makeSyntheticSentences import (extractCharacterSnippets, addSingleLetterSnippets,
                                    makeSyntheticDataFromRawSnippets)  # noqa: E402

ROOT = os.path.join(REPO, 'data', '20210512_willett', 'handwritingBCIData')
STEP3_DIR = os.path.join(ROOT, 'RNNTrainingSteps', 'Step3_SyntheticSentences')

DATA_DIRS = ['t5.2019.05.08', 't5.2019.11.25', 't5.2019.12.09', 't5.2019.12.11', 't5.2019.12.18',
             't5.2019.12.20', 't5.2020.01.06', 't5.2020.01.08', 't5.2020.01.13', 't5.2020.01.15']
CV_PARTS = ['HeldOutBlocks', 'HeldOutTrials']


def phase_snippets(data_dirs, cv_parts):
    char_def = getHandwritingCharacterDefinitions()
    for data_dir in data_dirs:
        print('Processing ' + data_dir, flush=True)
        sentence_dat = scipy.io.loadmat(os.path.join(ROOT, 'Datasets', data_dir, 'sentences.mat'))
        single_letter_dat = scipy.io.loadmat(os.path.join(ROOT, 'Datasets', data_dir, 'singleLetters.mat'))
        tw_cubes = scipy.io.loadmat(os.path.join(ROOT, 'RNNTrainingSteps', 'Step1_TimeWarping', data_dir + '_warpedCubes.mat'))

        # 官方：去掉最后两个 session 的 # 标记（暂停标记不是字符）
        for x in range(sentence_dat['sentencePrompt'].shape[0]):
            sentence_dat['sentencePrompt'][x, 0][0] = sentence_dat['sentencePrompt'][x, 0][0].replace('#', '')

        neural_cube = normalizeSentenceDataCube(sentence_dat, single_letter_dat)

        for cv_part in cv_parts:
            print('--' + cv_part, flush=True)
            part = scipy.io.loadmat(os.path.join(ROOT, 'RNNTrainingSteps', 'trainTestPartitions_' + cv_part + '.mat'))
            train_idx = part[data_dir + '_train']
            labels = scipy.io.loadmat(os.path.join(ROOT, 'RNNTrainingSteps', 'Step2_HMMLabels', cv_part, data_dir + '_timeSeriesLabels.mat'))

            snippet_dict = extractCharacterSnippets(labels['letterStarts'], labels['blankWindows'], neural_cube,
                                                    sentence_dat['sentencePrompt'][:, 0],
                                                    sentence_dat['numTimeBinsPerSentence'][:, 0],
                                                    train_idx, char_def)
            snippet_dict = addSingleLetterSnippets(snippet_dict, single_letter_dat, tw_cubes, char_def)

            os.makedirs(os.path.join(STEP3_DIR, cv_part), exist_ok=True)
            out = os.path.join(STEP3_DIR, cv_part, data_dir + '_snippets.pkl')
            with open(out, 'wb') as f:
                pickle.dump({k: (v if isinstance(v, (int, float, str)) else
                                 [np.asarray(x, dtype=np.float64) if isinstance(x, np.ndarray) else x for x in v])
                             for k, v in snippet_dict.items()}, f)
            n_per_char = {k: len(v) for k, v in snippet_dict.items() if isinstance(v, list)}
            print('   saved', out, 'blank:', len(snippet_dict['blank']), 'a:', len(snippet_dict['a']), flush=True)


def _to_mat_style(d):
    """把 pkl 里的普通 list 转成官方代码期望的 loadmat 风格结构（[0,i] 索引）。"""
    out = {}
    for k, v in d.items():
        if not isinstance(v, list):
            out[k] = v
        elif k.endswith('_penEndState'):
            out[k] = np.asarray([v], dtype=np.float64)
        else:
            arr = np.empty((1, len(v)), dtype=object)
            arr[0, :] = v
            out[k] = arr
    return out


def _generate_one(work):
    args, bat_idx = work
    np.random.seed(args['seed'])

    with open(args['snippetFile'], 'rb') as f:
        char_snippets = _to_mat_style(pickle.load(f))

    word_list = [line.rstrip('\n') for line in open(args['wordListFile'])]
    if args['rareWordFile'] != 'None':
        rare_word_list = (np.squeeze(scipy.io.loadmat(args['rareWordFile'])['rareIdx']) - 1).tolist()
        rare_letter_increase = True
    else:
        rare_word_list = []
        rare_letter_increase = False

    n_steps_to_generate = args['nSteps'] + 2000
    synth_neural, synth_char_prob, synth_char_start = makeSyntheticDataFromRawSnippets(
        charDef=args['charDef'], charSnippets=char_snippets, nSentences=args['nSentences'],
        nSteps=n_steps_to_generate, wordList=word_list, blankProb=0.20,
        accountForPenState=args['accountForPenState'], rareLetterIncrease=rare_letter_increase,
        rareWordList=rare_word_list)

    synth_targets = np.concatenate([synth_char_prob, synth_char_start[:, :, np.newaxis]], axis=2)
    del synth_char_prob, synth_char_start

    # 随机截断起点（照抄官方）
    synth_neural_cut = np.zeros([args['nSentences'], args['nSteps'], synth_neural.shape[2]])
    synth_targets_cut = np.zeros([args['nSentences'], args['nSteps'], synth_targets.shape[2]])
    for t in range(args['nSentences']):
        rand_start = np.random.randint(n_steps_to_generate - args['nSteps'])
        synth_neural_cut[t, :, :] = synth_neural[t, rand_start:(rand_start + args['nSteps']), :]
        synth_targets_cut[t, :, :] = synth_targets[t, rand_start:(rand_start + args['nSteps']), :]
    synth_neural, synth_targets = synth_neural_cut, synth_targets_cut
    del synth_neural_cut, synth_targets_cut

    # 时间分箱（照抄官方滑窗均值）
    n_bins = np.ceil(args['nSteps'] / args['binSize']).astype(int)
    synth_neural_binned = np.zeros([args['nSentences'], n_bins, synth_neural.shape[2]], dtype=np.float32)
    synth_targets_binned = np.zeros([args['nSentences'], n_bins, synth_targets.shape[2]], dtype=np.float32)
    curr_idx = np.arange(0, args['binSize']).astype(int)
    for x in range(n_bins):
        synth_neural_binned[:, x, :] = np.mean(synth_neural[:, curr_idx, :], axis=1)
        synth_targets_binned[:, x, :] = np.mean(synth_targets[:, curr_idx, :], axis=1)
        curr_idx += args['binSize']

    # 误差掩码：不惩罚首字符开始之前的帧（照抄官方）
    err_weights = np.ones([synth_targets_binned.shape[0], synth_targets_binned.shape[1]], dtype=np.float32)
    for t in range(err_weights.shape[0]):
        char_starts = np.argwhere(synth_targets_binned[t, 1:, -1] - synth_targets_binned[t, 0:-1, -1] >= 0.1).astype(np.int32)
        if len(char_starts) == 0:
            err_weights[t, :] = 0
        else:
            err_weights[t, 0:char_starts[0, 0]] = 0

    np.save(args['saveFile'] + '_inputs.npy', synth_neural_binned)
    np.save(args['saveFile'] + '_labels.npy', synth_targets_binned)
    np.save(args['saveFile'] + '_errW.npy', err_weights)
    return args['saveFile']


def phase_generate(data_dirs, cv_parts, n_parallel):
    char_def = getHandwritingCharacterDefinitions()
    for data_dir in data_dirs:
        for cv_part in cv_parts:
            output_dir = os.path.join(STEP3_DIR, cv_part, data_dir + '_syntheticSentences')
            os.makedirs(output_dir, exist_ok=True)

            todo = [x for x in range(20)
                    if not os.path.isfile(os.path.join(output_dir, f'bat_{x}_errW.npy'))]
            if not todo:
                print(f'skip {data_dir} {cv_part} (20 bats 已存在)', flush=True)
                continue

            base_args = {'nSentences': 256, 'nSteps': 2400, 'binSize': 2,
                         'wordListFile': os.path.join(OFFICIAL_REPO, 'wordList', 'google-10000-english-usa.txt'),
                         'rareWordFile': os.path.join(OFFICIAL_REPO, 'wordList', 'rareWordIdx.mat'),
                         'snippetFile': os.path.join(STEP3_DIR, cv_part, data_dir + '_snippets.pkl'),
                         'accountForPenState': 1, 'charDef': char_def,
                         'seed': datetime.now().microsecond}
            work = []
            for x in todo:
                new_args = dict(base_args)
                new_args['saveFile'] = os.path.join(output_dir, 'bat_' + str(x))
                new_args['seed'] += x
                work.append((new_args, x))

            print(f'generating {data_dir} {cv_part}: {len(work)} bats x 256 sentences', flush=True)
            with multiprocessing.Pool(n_parallel) as pool:
                for i, _ in enumerate(pool.imap_unordered(_generate_one, work), 1):
                    print(f'  [{i}/{len(work)}] {os.path.basename(_)}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['snippets', 'generate', 'both'], default='both')
    ap.add_argument('--dataDir', default=None, help='只处理一个 session（调试用）')
    ap.add_argument('--cvPart', default=None, choices=[None, 'HeldOutBlocks', 'HeldOutTrials'])
    ap.add_argument('--nParallel', type=int, default=6,
                    help='并行进程数；float64 中间量约 4GB/进程，10 进程会爆 32GB 内存')
    cli = vars(ap.parse_args())

    data_dirs = [cli['dataDir']] if cli['dataDir'] else DATA_DIRS
    cv_parts = [cli['cvPart']] if cli['cvPart'] else CV_PARTS

    if cli['phase'] in ('snippets', 'both'):
        phase_snippets(data_dirs, cv_parts)
    if cli['phase'] in ('generate', 'both'):
        phase_generate(data_dirs, cv_parts, cli['nParallel'])


if __name__ == '__main__':
    main()
