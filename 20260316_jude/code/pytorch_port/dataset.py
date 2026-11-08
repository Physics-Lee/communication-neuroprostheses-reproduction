"""npz dataset plus a PyTorch equivalent of the official augmentation/smoothing.

Mirrors the official brainToText_trainDecoder.datasetLayerTransform: training
augmentation adds white noise (sd 1.5) plus a constant offset (sd 0.22); both
training and validation get Gaussian temporal smoothing with SD=2 and SAME
padding; staticGain/randomWalk/randomCut are 0 in the official config and are
therefore not implemented.

The gaussSmooth kernel is built exactly as in the official gaussSmooth.py:
exp(-0.5*(d/sd)^2), truncated at values <= 0.01, then normalized (for sd=2
this keeps |d| <= 6, i.e. 13 taps; identical to truncating the impulse
response of scipy's gaussian_filter1d).

Notes
-----
Written 2026-10-10.
"""
import numpy as np
import torch
import torch.nn.functional as F


def gaussian_kernel(sd=2.0, thresh=0.01):
    """Build the 1-D Gaussian smoothing kernel used by :func:`gauss_smooth`.

    Follows the official gaussSmooth.py construction: samples of
    ``exp(-0.5*(d/sd)**2)`` are dropped where they do not exceed ``thresh``
    and the rest are normalized to sum to 1.

    Parameters
    ----------
    sd : float, optional
        Standard deviation of the Gaussian. The sampling radius follows
        scipy's default truncate=4, i.e. ``int(4*sd + 0.5)``.
    thresh : float, optional
        Kernel values not exceeding this threshold are truncated away.

    Returns
    -------
    torch.Tensor
        1-D float32 normalized Gaussian kernel (13 taps for the defaults).
    """
    radius = int(4 * sd + 0.5)  # scipy 默认 truncate=4
    d = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-0.5 * (d / sd) ** 2)
    k = k[k > thresh]  # 官方：gaussKernel > 0.01 截断
    k = k / k.sum()    # 官方：归一化
    return torch.tensor(k, dtype=torch.float32)


_SMOOTH_KERNEL = None


def gauss_smooth(x, sd=2.0):
    """x: (B,T,C) 沿时间 depthwise 高斯卷积，SAME padding（官方 tf.nn.depthwise_conv2d padding='SAME'）。

    The kernel is built once on the first call and cached in the
    module-level ``_SMOOTH_KERNEL`` global.

    Parameters
    ----------
    x : torch.Tensor
        Input of shape ``(B, T, C)``.
    sd : float, optional
        Gaussian standard deviation; must match the first call's value
        because the kernel is cached.

    Returns
    -------
    torch.Tensor
        Smoothed tensor with the same shape ``(B, T, C)``.
    """
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
    """每 trial 变长 (T,768)。按官方语义：batch 内 pad 0，真长由 nTimeSteps 提供。

    Loads one split of the npz produced by export_tfrecord_to_npz.py and
    exposes per-trial variable-length feature arrays plus CTC labels.

    Attributes
    ----------
    features : list of numpy.ndarray
        Per-trial float32 arrays of shape ``(T_i, 768)``.
    seq_ids : numpy.ndarray
        Dense per-trial label id arrays (1-based, 0-padded).
    n_ts : numpy.ndarray
        Per-trial number of 20 ms time steps.
    n_el : numpy.ndarray
        Per-trial number of sequence elements (label length).
    transcriptions : numpy.ndarray
        Dense per-trial character-code arrays (0-padded).
    labels : list of torch.Tensor
        Per-trial 0-based label tensors (pad zeros dropped -> classes 0..29).
    """

    def __init__(self, npz_path, split='train'):
        """Load one split of the exported npz into per-trial arrays.

        Parameters
        ----------
        npz_path : str or os.PathLike
            Path to the ``m1_torch_data.npz`` produced by
            export_tfrecord_to_npz.py.
        split : str, optional
            ``'train'`` or ``'test'``; selects the ``<split>_*`` npz keys.
        """
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
        """Return the number of trials in this split.

        Returns
        -------
        int
            Number of trials.
        """
        return len(self.features)

    def make_batch(self, idx, augment=False, white_noise_sd=1.5, const_offset_sd=0.22, device=None):
        """Assemble, pad, augment and smooth a batch of trials.

        Trials are zero-padded to the batch's max length; optional training
        augmentation (white noise + per-channel constant offset) is applied,
        and Gaussian smoothing always follows — matching the official order
        (smoothing after augmentation, on both train and validation).

        Parameters
        ----------
        idx : sequence of int
            Trial indices to include in the batch.
        augment : bool, optional
            Apply the official training augmentation.
        white_noise_sd : float, optional
            Standard deviation of additive white noise.
        const_offset_sd : float, optional
            Standard deviation of the per-channel constant offset.
        device : torch.device or str, optional
            If given, augmentation and smoothing run on this device.

        Returns
        -------
        tuple of torch.Tensor
            ``(x, input_lengths, targets, target_lengths)``: padded smoothed
            batch ``(B, T, C)``; per-trial downsampled lengths
            ``T' = floor((T - 15) / 6) + 1``; concatenated label tensor; and
            per-trial label lengths.
        """
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
    """官方 tf.nn.ctc_greedy_decoder(merge_repeated=True) 等价。logits: (T',31) 单 trial。

    Parameters
    ----------
    logits : torch.Tensor
        Per-frame logits of shape ``(T', 31)`` for a single trial.
    input_len : int
        Number of valid frames ``T'`` (frames beyond it are padding).
    blank : int, optional
        Index of the CTC blank class (last position in native order).

    Returns
    -------
    list of int
        Greedy CTC label sequence with repeats and blanks removed.
    """
    seq = logits[:input_len].argmax(dim=-1)
    out, prev = [], None
    for s in seq.tolist():
        if s != prev and s != blank:
            out.append(s)
        prev = s
    return out


def edit_distance(a, b):
    """官方 tf.edit_distance(normalize=False) 等价的 Levenshtein（token 级）。

    Parameters
    ----------
    a : sequence
        Reference token sequence (e.g. true label ids).
    b : sequence
        Hypothesis token sequence (e.g. decoded label ids).

    Returns
    -------
    int
        Unnormalized token-level edit distance between ``a`` and ``b``.
    """
    la, lb = len(a), len(b)
    d = np.zeros((la + 1, lb + 1), dtype=np.int64)
    d[:, 0] = np.arange(la + 1)
    d[0, :] = np.arange(lb + 1)
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            d[i, j] = d[i - 1, j - 1] if a[i - 1] == b[j - 1] else min(d[i - 1][j - 1] + 1, d[i][j - 1] + 1, d[i - 1][j] + 1)
    return int(d[la, lb])
