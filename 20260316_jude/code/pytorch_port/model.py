# model.py — 官方 utils/models.py GRU + 会话输入网的 PyTorch 等价移植（2026-10-10）
# 保真要点（对照官方 models.py GRU 与 brainToText_trainDecoder.__init__）：
# - patch: extract_patches(size=15, stride=6, VALID)；此处用 unfold 实现（patch 内元素排列序与 TF 不同，从零训练无影响）
# - GRU: Keras(2.x) GRU 默认 reset_after=True，与 nn.GRU 门控语义一致；Keras dropout 作用于每层输入
#   （含第 1 层）且为 variational（同一掩码跨时间步共享）——本移植用"5 个单层 GRU 手动堆叠 + (B,1,F)
#   variational 掩码"实现同一语义。不用 nn.GRU(num_layers=5, dropout=...)：该写法在本机 Windows torch
#   上会掉出 cuDNN 快路径（实测前向 2.04s vs 0.23s，慢 9 倍），单层堆叠则走 cuDNN（fwd+bwd 0.79s）。
# - 初始状态：官方 initStates 为可学习 (1,units) glorot 初始化、仅用于第 0 层，其余层零初始
# - 输入网：Dense(768->768) softsign + Dropout(0.2)，核初始化为单位阵（官方 kernel_initializer=identity）
# - 输出头：Dense(31) logits（30 字符类 + blank，blank=最后一个，官方 ctc blank_index=-1）
import torch
import torch.nn as nn
import torch.nn.functional as F


class InputNet(nn.Module):
    """官方每会话输入网：Dense(softsign) + Dropout，核初始化单位阵。"""

    def __init__(self, n_in=768, n_out=768, dropout=0.2):
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(n_out, n_in))
            self.fc.bias.zero_()
        self.dropout = dropout

    def forward(self, x, training=False):
        x = F.softsign(self.fc(x))
        if training and self.dropout > 0:
            b = x.shape[0]
            f = x.shape[-1]
            mask = (torch.rand(b, 1, f, device=x.device) > self.dropout) / (1 - self.dropout)
            x = x * mask
        return x


class TypingGRU(nn.Module):
    """官方 GRU 类的等价移植：patch 化 -> [variational dropout -> 单层 GRU] x5 -> Dense(31)。"""

    def __init__(self, n_features=768, units=512, n_layers=5, n_classes=31,
                 patch_size=15, patch_stride=6, gru_dropout=0.6):
        super().__init__()
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.layers = nn.ModuleList([
            nn.GRU(n_features * patch_size if i == 0 else units, units, num_layers=1, batch_first=True)
            for i in range(n_layers)
        ])
        self.gru_dropout = gru_dropout
        self.init_states = nn.Parameter(torch.empty(1, units))
        nn.init.xavier_uniform_(self.init_states)  # 官方 glorot_uniform
        self.dense = nn.Linear(units, n_classes)

    def forward(self, x, training=False):
        # x: (B, T, F) -> patches (B, T', patch*F)
        b, t, f = x.shape
        if t < self.patch_size:
            raise ValueError(f'序列长度 {t} < patch_size {self.patch_size}')
        x = x.unfold(1, self.patch_size, self.patch_stride)        # (B, T', F, patch)
        x = x.permute(0, 1, 3, 2).reshape(b, x.shape[1], self.patch_size * f)
        for i, layer in enumerate(self.layers):
            if training and self.gru_dropout > 0:
                # Keras 式 variational dropout：掩码 (B,1,F) 跨时间步共享，作用于该层输入
                mask = (torch.rand(b, 1, x.shape[-1], device=x.device) > self.gru_dropout) / (1 - self.gru_dropout)
                x = x * mask
            if i == 0:
                h0 = self.init_states.expand(b, -1).unsqueeze(0).contiguous()  # (1,B,U) 可学习初始状态，仅层 0
                x, _ = layer(x, h0)
            else:
                x, _ = layer(x)
        return self.dense(x)

    def l2_reg(self, weight_reg):
        """Keras L2 正则等价：1e-5 * Σw²，仅作用于 GRU 输入核（weight_ih）——官方只给 kernel 加了正则。"""
        reg = sum((w ** 2).sum() for name, w in self.named_parameters() if 'weight_ih' in name)
        return weight_reg * reg
