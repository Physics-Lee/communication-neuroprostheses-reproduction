# -*- coding: utf-8 -*-
"""训练数据增强的 numpy 移植（官方 charSeqRNN.py 的 extractSentenceSnippet / addMeanNoise / addWhiteNoise / parseDataset 噪声部分）。

已知官方怪癖（忠实保留，可配置）：randomWalk 噪声的 cumsum 沿特征轴（axis=1）而非时间轴，
效果是"通道 i 的漂移 = 通道 0..i 的随机数累加"，通道间相关且方差随索引增大。
官方训练即在此实现下取得论文结果，默认 axis_mode='official'；'time' 为直觉修正版，仅供消融。
"""
import numpy as np


def extract_sentence_snippet(inputs, targets, err_weight, bins_per_trial, n_steps, directionality, rng):
    """从整句中随机截取 n_steps 长片段，并按官方逻辑重算 errWeight：
    起始信号首个上升沿之前的帧权重清零（单向）；双向还要把最后一个不完整字符之后清零。
    输入均为单个样本：inputs (T,N), targets (T,C), err_weight (T,), bins_per_trial int。
    返回 (inputs_snip, targets_snip, ews)，长度均为 n_steps（不足补零）。
    """
    hi = max(bins_per_trial + (n_steps - 100) - 400, 1)   # 官方 uniform 上界（排他）
    start = rng.integers(0, hi)

    def pad_to(arr, length):
        if arr.shape[0] >= length:
            return arr[:length]
        out = np.zeros((length,) + arr.shape[1:], dtype=arr.dtype)
        out[:arr.shape[0]] = arr
        return out

    inp = pad_to(inputs[start:start + n_steps], n_steps)
    targ = pad_to(targets[start:start + n_steps], n_steps)

    rising = targ[1:, -1] - targ[:-1, -1] >= 0.1
    char_starts = np.argwhere(rising)[:, 0]

    if len(char_starts) == 0:
        ews = np.zeros(n_steps)
    else:
        first = int(char_starts[0])
        last = int(char_starts[-1])
        if directionality == 'unidirectional':
            ews = np.concatenate([np.zeros(first), err_weight[start + first:start + n_steps]])
        else:
            ews = np.concatenate([np.zeros(first),
                                  err_weight[start + first:start + last],
                                  np.zeros(n_steps - last)])
        ews = pad_to(ews, n_steps)

    return inp, targ, ews


def add_mean_noise(inputs, constant_offset_sd, random_walk_sd, rng, axis_mode='official'):
    """官方 addMeanNoise：常量偏移 (1,N) + 随机游走。axis_mode='official' 时 cumsum 沿特征轴（axis=1，照抄官方），
    'time' 沿时间轴（直觉语义）。inputs (T,N)。"""
    noise = rng.normal(0.0, constant_offset_sd, size=(1, inputs.shape[1]))
    walk = rng.normal(0.0, random_walk_sd, size=inputs.shape)
    noise = noise + np.cumsum(walk, axis=1 if axis_mode == 'official' else 0)
    return inputs + noise


def add_white_noise(inputs, white_noise_sd, rng):
    """官方 addWhiteNoise：逐时间步白噪声 (T,N)。"""
    return inputs + rng.normal(0.0, white_noise_sd, size=inputs.shape)


def apply_train_noise(inputs, targets, err_weight, bins_per_trial, args, rng, axis_mode='official'):
    """训练噪声总入口：常量偏移 + 随机游走 + 白噪声（参数取自 args，与官方 parseDataset/_makeTrainingDatasetFromRealData 一致）。"""
    if args['constantOffsetSD'] > 0 or args['randomWalkSD'] > 0:
        inputs = add_mean_noise(inputs, args['constantOffsetSD'], args['randomWalkSD'], rng, axis_mode)
    if args['whiteNoiseSD'] > 0:
        inputs = add_white_noise(inputs, args['whiteNoiseSD'], rng)
    return inputs, targets, err_weight
