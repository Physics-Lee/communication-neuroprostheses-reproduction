# -*- coding: utf-8 -*-
"""CharSeqRNN 的 PyTorch 移植（官方 TF1 charSeqRNN.py 的忠实重写）。

对齐口径：指标级（两划分 raw CER 同数量级），非逐权重复刻。
结构忠实于官方图：
    高斯平滑(SD=4/rnnBinSize, 截断>0.01, SAME) -> 每 session 专属线性输入层(恒等初始化)
    -> GRU 层1(512 单元, 全时间步) -> GRU 层2(512 单元, 每 skipLen 步取 1 步, 读出后 repeat_interleave 上采样)
    -> 线性读出(B, T, 32) = 31 字符 logit + 1 字符起始信号 logit。
GRU 采用 PyTorch nn.GRU：CUDA 上走 cuDNN 内核，门控/偏置排布与 tf.contrib.cudnn_rnn.CudnnGRU
同约定（Keras 语境的 reset_after=True）。已验证的超参见数据包 Step4_RNNTraining/*/args.p。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d


def build_gauss_kernel(rnn_bin_size):
    """官方 gaussSmooth 的核：冲激响应经 gaussian_filter1d(SD=4/rnnBinSize)，截断 >0.01，归一化。

    Parameters
    ----------
    rnn_bin_size : int
        Number of raw time steps per RNN bin; the Gaussian SD is
        ``4.0 / rnn_bin_size`` bins, matching the official smoothing.

    Returns
    -------
    numpy.ndarray
        1-D float32 kernel (impulse response entries above 0.01,
        normalised to sum 1).
    """
    inp = np.zeros(100)
    inp[50] = 1
    kernel = gaussian_filter1d(inp, 4.0 / rnn_bin_size)
    kernel = kernel[kernel > 0.01]
    kernel = np.squeeze(kernel / np.sum(kernel))
    return kernel.astype(np.float32)


class CharSeqRNN(nn.Module):
    """PyTorch port of the official TF1 CharSeqRNN handwriting decoder.

    Pipeline: per-feature Gaussian smoothing -> per-session identity-
    initialised linear input layer -> GRU layer 1 over all frames -> GRU
    layer 2 on every ``skip_len``-th frame -> linear readout producing 31
    character logits plus 1 character-start logit, upsampled by
    ``repeat_interleave``.

    Parameters
    ----------
    n_inputs : int, optional
        Number of input neural features (default 192).
    n_outputs : int, optional
        Readout width: 31 character logits + 1 start-signal logit
        (default 32).
    n_units : int, optional
        Hidden units per GRU layer (default 512).
    directionality : str, optional
        ``'unidirectional'`` or ``'bidirectional'`` (default
        unidirectional).
    n_input_layers : int, optional
        Number of per-session input layers in the ParameterLists
        (default 10).
    skip_len : int, optional
        Time-step stride fed to the second GRU (default 5).
    rnn_bin_size : int, optional
        Bins per RNN step, sets the smoothing kernel SD (default 2).
    smooth_inputs : bool, optional
        Apply the Gaussian smoothing buffer (default True).
    trainable_input : bool, optional
        Whether the per-session input layers are trainable (default True).
    trainable_backend : bool, optional
        Whether GRU/readout weights are trainable (default True).
    """

    def __init__(self, n_inputs=192, n_outputs=32, n_units=512, directionality='unidirectional',
                 n_input_layers=10, skip_len=5, rnn_bin_size=2, smooth_inputs=True,
                 trainable_input=True, trainable_backend=True):
        """Build the layers and apply the official initialisation scheme.

        Parameters
        ----------
        n_inputs : int, optional
            Number of input neural features (default 192).
        n_outputs : int, optional
            Readout width (default 32 = 31 chars + start signal).
        n_units : int, optional
            Hidden units per GRU layer (default 512).
        directionality : str, optional
            ``'unidirectional'`` or ``'bidirectional'``.
        n_input_layers : int, optional
            Number of per-session input layers (default 10).
        skip_len : int, optional
            Stride for the second GRU (default 5).
        rnn_bin_size : int, optional
            Bins per RNN step for the smoothing kernel (default 2).
        smooth_inputs : bool, optional
            Register the Gaussian smoothing kernel (default True).
        trainable_input : bool, optional
            Make the input-layer parameters trainable (default True).
        trainable_backend : bool, optional
            Make GRU/readout/start-state parameters trainable (default
            True).

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            If ``directionality`` is not one of the two allowed values.
        """
        super().__init__()
        assert directionality in ('unidirectional', 'bidirectional')
        assert n_units % 1 == 0
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.n_units = n_units
        self.directionality = directionality
        self.bi_dir = 2 if directionality == 'bidirectional' else 1
        self.n_input_layers = n_input_layers
        self.skip_len = skip_len
        self.smooth_inputs = smooth_inputs
        self.trainable_input = trainable_input
        self.trainable_backend = trainable_backend

        # 高斯平滑：按特征分组的 conv1d（官方逐特征卷积后拼接的等价实现），TF 'SAME' 对称 pad
        if smooth_inputs:
            kernel = build_gauss_kernel(rnn_bin_size)
            self.kernel_len = len(kernel)
            self.register_buffer('smooth_weight', torch.from_numpy(kernel)[None, None, :].repeat(n_inputs, 1, 1))
            self.smooth_pad = (self.kernel_len - 1) // 2

        # 每 session 专属输入层：W 恒等初始化、b 零初始化（官方 np.identity / np.zeros）
        self.input_W = nn.ParameterList(
            [nn.Parameter(torch.eye(n_inputs), requires_grad=trainable_input) for _ in range(n_input_layers)])
        self.input_b = nn.ParameterList(
            [nn.Parameter(torch.zeros(n_inputs), requires_grad=trainable_input) for _ in range(n_input_layers)])

        # 两层 GRU；初始状态为可训练零向量，两层共享同一个 start state（官方行为）
        self.gru1 = nn.GRU(n_inputs, n_units, batch_first=True,
                           bidirectional=(directionality == 'bidirectional'))
        self.gru2 = nn.GRU(n_units * self.bi_dir, n_units, batch_first=True,
                           bidirectional=(directionality == 'bidirectional'))
        for gru in (self.gru1, self.gru2):
            for name, p in gru.named_parameters():
                if name.startswith('weight'):
                    nn.init.xavier_uniform_(p)      # 近似 cuDNN 默认 glorot；文档见 README
                elif name.startswith('bias'):
                    nn.init.zeros_(p)               # 官方 bias_initializer=0.0
                p.requires_grad_(trainable_backend)
        self.start_state = nn.Parameter(torch.zeros(self.bi_dir, 1, n_units), requires_grad=trainable_backend)

        # 线性读出：W ~ N(0, 0.05)，b = 0（官方 random_normal_initializer(stddev=0.05) / zeros）
        self.readout_W = nn.Parameter(torch.randn(n_units * self.bi_dir, n_outputs) * 0.05,
                                       requires_grad=trainable_backend)
        self.readout_b = nn.Parameter(torch.zeros(n_outputs), requires_grad=trainable_backend)

    def gauss_smooth(self, x):
        """Smooth each feature channel with the registered Gaussian kernel.

        Implemented as a grouped 1-D convolution with symmetric padding,
        which is equivalent to the official TF ``'SAME'`` per-feature
        convolution.

        Parameters
        ----------
        x : torch.Tensor
            Inputs of shape ``(B, T, N)``.

        Returns
        -------
        torch.Tensor
            Smoothed inputs, shape ``(B, T, N)``.
        """
        # x: (B, T, N) -> (B, N, T) 分组卷积 -> (B, T, N)，stride 1 奇数核对称 pad 等价 TF 'SAME'
        xt = x.transpose(1, 2)
        xt = F.conv1d(F.pad(xt, (self.smooth_pad, self.smooth_pad)), self.smooth_weight, groups=self.n_inputs)
        return xt.transpose(1, 2)

    def forward(self, x, input_layer_idx=0):
        """Run the full CharSeqRNN forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Inputs of shape ``(B, T, N)`` float32; ``T`` must be divisible
            by ``skip_len`` (e.g. 1200/5 officially).
        input_layer_idx : int, optional
            Index into the per-session input ParameterLists (default 0).

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, T, n_outputs)``; layer-2 outputs are
            computed every ``skip_len`` frames and upsampled back to ``T``
            via ``repeat_interleave`` (equivalent to the official expIdx
            gather).

        Raises
        ------
        AssertionError
            If ``T`` is not divisible by ``skip_len``.
        """
        # x: (B, T, N) float32；T 必须能被 skip_len 整除（官方 1200/5）
        B, T, N = x.shape
        assert T % self.skip_len == 0, f'T({T}) 必须被 skip_len({self.skip_len}) 整除'
        if self.smooth_inputs:
            x = self.gauss_smooth(x)
        x = x @ self.input_W[input_layer_idx] + self.input_b[input_layer_idx]
        h0 = self.start_state.expand(-1, B, -1).contiguous()
        y1, _ = self.gru1(x, h0)
        # 官方 rnnOutput[:,0::skipLen,:]：对时间维取步；contiguous 保证 GRU 走 cuDNN 快速路径
        y2, _ = self.gru2(y1[:, ::self.skip_len, :].contiguous(), h0)
        logits_ds = y2 @ self.readout_W + self.readout_b      # (B, T/skip_len, n_outputs)
        return logits_ds.repeat_interleave(self.skip_len, dim=1)  # 官方 expIdx gather 的等价上采样


def char_seq_loss(logits, targets, err_weight, time_steps=1200, output_delay=50):
    """官方损失：outputDelay 对齐切片后，
        CE 项  = mean_B( sum_t bw*CE_t / timeSteps )          （注意除以 timeSteps 而非有效帧数）
        起始项 = 5 * mean_B( sum_t bw*(sigmoid(logit)-label)^2 / timeSteps )
    返回 (total, ce_term, trans_term)。

    Parameters
    ----------
    logits : torch.Tensor
        Model outputs, shape ``(B, T, n_outputs)``; last channel is the
        character-start logit.
    targets : torch.Tensor
        Labels, shape ``(B, T, n_outputs)``; last channel is the binary
        start signal.
    err_weight : torch.Tensor
        Per-frame weights, shape ``(B, T)``.
    time_steps : int, optional
        Divisor applied to the per-sentence sums (default 1200; the
        official code divides by timeSteps, not by the count of valid
        frames).
    output_delay : int, optional
        Number of leading frames dropped for alignment (default 50).

    Returns
    -------
    tuple of (torch.Tensor, torch.Tensor, torch.Tensor)
        ``(total, ce_term, trans_term)`` scalar losses; ``total`` is the
        sum of the cross-entropy term and 5x the start-signal MSE term.
    """
    labels = targets[:, :-output_delay, :]
    logits_d = logits[:, output_delay:, :]
    bw = err_weight[:, :-output_delay]
    char_logits = logits_d[:, :, :-1]
    char_labels = labels[:, :, :-1]
    ce = -(char_labels * F.log_softmax(char_logits, dim=-1)).sum(-1)      # (B, T-d)
    ce_term = (bw * ce).sum(1).div(time_steps).mean()
    trans = (torch.sigmoid(logits_d[:, :, -1]) - labels[:, :, -1]).pow(2)
    trans_term = 5 * (bw * trans).sum(1).div(time_steps).mean()
    return ce_term + trans_term, ce_term, trans_term


def l2_cost(model):
    """官方 L2 惩罚集合：readout_W + 各输入层 W + 两层 GRU 的全部参数（cuDNN 权重 blob 含偏置，故含 bias）。
    不含 readout_b、输入层 b、start_state（与官方 weightVars 集合一致）。

    Parameters
    ----------
    model : CharSeqRNN
        The model whose penalised weights are summed.

    Returns
    -------
    torch.Tensor
        Scalar sum of squared parameters over the official weightVars set.
    """
    total = model.readout_W.pow(2).sum()
    for W in model.input_W:
        total = total + W.pow(2).sum()
    for gru in (model.gru1, model.gru2):
        for p in gru.parameters():
            total = total + p.pow(2).sum()
    return total


def compute_frame_accuracy(logit_output, targets, err_weight, output_delay=50):
    """官方 computeFrameAccuracy 的 numpy 移植（帧级字符分类准确率，供训练监控）。

    Parameters
    ----------
    logit_output : numpy.ndarray
        Model outputs, shape ``(B, T, n_outputs)``; the first
        ``n_outputs-1`` channels are character logits.
    targets : numpy.ndarray
        Labels, shape ``(B, T, n_outputs)``; same channel convention.
    err_weight : numpy.ndarray
        Per-frame weights, shape ``(B, T)``.
    output_delay : int, optional
        Leading frames dropped for alignment (default 50).

    Returns
    -------
    float
        Error-weighted fraction of frames whose argmax character matches
        the target argmax.
    """
    best_class = np.argmax(logit_output[:, output_delay:, :-1], axis=2)
    indicated_class = np.argmax(targets[:, :-output_delay, :-1], axis=2)
    bw = err_weight[:, :-output_delay]
    acc = np.sum(bw * np.equal(best_class, indicated_class)) / np.sum(bw)
    return acc
