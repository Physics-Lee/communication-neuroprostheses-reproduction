# dataset.py — npz 数据集 + 官方增广/平滑的 PyTorch 等价（2026-10-10）
# 对照官方 brainToText_trainDecoder.datasetLayerTransform（训练增广：白噪 1.5 + 常数偏移 0.22；
# 训练与验证都做高斯时间平滑 SD=2、SAME padding；staticGain/randomWalk/randomCut 官方为 0 不实现）。
# gaussSmooth 核的构造与官方 gaussSmooth.py 数值一致：exp(-0.5*(d/sd)^2)，截断阈值 >0.01，归一化
# （sd=2 时 |d|<=6，13 抽头；与 scipy gaussian_filter1d 脉冲响应再截断的结果相同）。
import numpy as np
import torch
import torch.nn.functional as F


def gaussian_kernel(sd=2.0, thresh=0.01):
    radius = int(4 * sd + 0.5)  # scipy 默认 truncate=4
    d = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (d / sd) ** 2)
    k = k[k > thresh]  # 官方：gaussKernel > 0.01 截断
    k = k / k.sum()    # 官方：归一化
    return torch.tensor(k, dtype=torch.float32)


_SMOOTH_KERNEL = None


def gauss_smooth(x, sd=2.0):
    """x: (B,T,C) 沿时间 depthwise 高斯卷积，SAME padding（官方 tf.nn.depthwise_conv2d padding='SAME'）。"""
    global _SMOOTH_KERNEL
    if _SMOOTH_KERNEL is None:
        _SMOOTH_KERNEL = gaussian_kernel(sd)
    k = _SMOOTH_KERNEL.to(x.device)
    b, t, c = x.shape
    pad = k.numel() // 2  # 13 抽头 -> pad 6，长度不变（SAME）
    y = x.permute(0, 2, 1).reshape(b * c, 1, t)
    y = F.conv1d(y, k.view(1, 1, -1), padding=pad)
    return y.reshape(b, c, t).permute(0, 2, 1)


class TypingDataset:
    """每 trial 变长 (T,768)。按官方语义：batch 内 pad 0，真长由 nTimeSteps 提供。"""

    def __init__(self, npz_path, split='train'):
        z = np.load(npz_path, allow_pickle=True)
        raw = z[f'{split}_features']
        if raw.ndim != 1:  # 单 trial 时 np.array([arr], dtype=object) 会被升维成 (1,T,C)——压回 1-D object
            raw = np.empty(raw.shape[0], dtype=object)
            for i in range(raw.shape[0]):
                raw[i] = np.asarray(z[f'{split}_features'][i], dtype=np.float32)
        self.features = [np.ascontiguousarray(f, dtype=np.float32) for f in raw]
        self.seq_ids = z[f'{split}_seqClassIDs']
        self.n_ts = z[f'{split}_nTimeSteps']
        self.n_el = z[f'{split}_nSeqElements']
        self.transcriptions = z[f'{split}_transcription']
        self.labels = [torch.tensor(r[r > 0] - 1, dtype=torch.long) for r in self.seq_ids]  # 0..29；pad 的 0 丢弃（对应官方 sparse.from_dense）

    def __len__(self):
        return len(self.features)

    def make_batch(self, idx, augment=False, white_noise_sd=1.5, const_offset_sd=0.22, device=None):
        feats = [torch.tensor(self.features[i], dtype=torch.float32) for i in idx]
        tmax = max(f.shape[0] for f in feats)
        b = len(idx)
        c = feats[0].shape[1]
        x = torch.zeros(b, tmax, c)
        for j, f in enumerate(feats):
            x[j, :f.shape[0]] = f
        if device is not None:  # 增广与平滑放 GPU（CPU 卷积曾是每批 ~15s 的瓶颈）
            x = x.to(device)
        if augment:
            if white_noise_sd > 0:
                x = x + torch.randn_like(x) * white_noise_sd
            if const_offset_sd > 0:
                x = x + torch.randn(b, 1, c, device=x.device) * const_offset_sd
        x = gauss_smooth(x)  # 官方：平滑在增广之后，训练验证都做
        input_lengths = torch.tensor([(self.n_ts[i] - 15) // 6 + 1 for i in idx], dtype=torch.long)  # T' = floor((T-patch)/stride)+1
        targets = torch.cat([self.labels[i] for i in idx])
        target_lengths = torch.tensor([self.n_el[i] for i in idx], dtype=torch.long)
        return x, input_lengths, targets, target_lengths


def greedy_ctc_decode(logits, input_len, blank=30):
    """官方 tf.nn.ctc_greedy_decoder(merge_repeated=True) 等价。logits: (T',31) 单 trial。"""
    seq = logits[:input_len].argmax(dim=-1)
    out, prev = [], None
    for s in seq.tolist():
        if s != prev and s != blank:
            out.append(s)
        prev = s
    return out


def edit_distance(a, b):
    """官方 tf.edit_distance(normalize=False) 等价的 Levenshtein（token 级）。"""
    la, lb = len(a), len(b)
    d = np.zeros((la + 1, lb + 1), dtype=np.int64)
    d[:, 0] = np.arange(la + 1)
    d[0, :] = np.arange(lb + 1)
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            d[i, j] = d[i - 1, j - 1] if a[i - 1] == b[j - 1] else min(d[i - 1][j - 1] + 1, d[i][j - 1] + 1, d[i - 1][j] + 1)
    return int(d[la, lb])
