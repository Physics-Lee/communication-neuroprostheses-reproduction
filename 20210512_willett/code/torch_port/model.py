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
    """官方 gaussSmooth 的核：冲激响应经 gaussian_filter1d(SD=4/rnnBinSize)，截断 >0.01，归一化。"""
    inp = np.zeros(100)
    inp[50] = 1
    kernel = gaussian_filter1d(inp, 4.0 / rnn_bin_size)
    kernel = kernel[kernel > 0.01]
    kernel = np.squeeze(kernel / np.sum(kernel))
    return kernel.astype(np.float32)


class CharSeqRNN(nn.Module):
    def __init__(self, n_inputs=192, n_outputs=32, n_units=512, directionality='unidirectional',
                 n_input_layers=10, skip_len=5, rnn_bin_size=2, smooth_inputs=True,
                 trainable_input=True, trainable_backend=True):
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
        # x: (B, T, N) -> (B, N, T) 分组卷积 -> (B, T, N)，stride 1 奇数核对称 pad 等价 TF 'SAME'
        xt = x.transpose(1, 2)
        xt = F.conv1d(F.pad(xt, (self.smooth_pad, self.smooth_pad)), self.smooth_weight, groups=self.n_inputs)
        return xt.transpose(1, 2)

    def forward(self, x, input_layer_idx=0):
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
    不含 readout_b、输入层 b、start_state（与官方 weightVars 集合一致）。"""
    total = model.readout_W.pow(2).sum()
    for W in model.input_W:
        total = total + W.pow(2).sum()
    for gru in (model.gru1, model.gru2):
        for p in gru.parameters():
            total = total + p.pow(2).sum()
    return total


def compute_frame_accuracy(logit_output, targets, err_weight, output_delay=50):
    """官方 computeFrameAccuracy 的 numpy 移植（帧级字符分类准确率，供训练监控）。"""
    best_class = np.argmax(logit_output[:, output_delay:, :-1], axis=2)
    indicated_class = np.argmax(targets[:, :-output_delay, :-1], axis=2)
    bw = err_weight[:, :-output_delay]
    acc = np.sum(bw * np.equal(best_class, indicated_class)) / np.sum(bw)
    return acc
