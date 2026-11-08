"""PyTorch port of the official utils/models.py GRU plus the per-session input net.

Fidelity notes (against the official models.py GRU and
brainToText_trainDecoder.__init__):

- patch: extract_patches(size=15, stride=6, VALID); implemented here with
  ``unfold`` (the intra-patch element order differs from TF, which has no
  effect when training from scratch)
- GRU: Keras(2.x) GRU defaults to reset_after=True, matching nn.GRU gate
  semantics; Keras dropout applies to every layer's input (including layer 1)
  and is variational (the same mask is shared across time steps) — this port
  realizes the same semantics by manually stacking 5 single-layer GRUs plus a
  (B, 1, F) variational mask. nn.GRU(num_layers=5, dropout=...) is
  deliberately NOT used: that form falls off the cuDNN fast path on the local
  Windows torch (measured forward 2.04 s vs 0.23 s, 9x slower), while stacked
  single layers stay on cuDNN (fwd+bwd 0.79 s).
- initial state: the official initStates is a learnable (1, units) tensor,
  glorot-initialized and used only for layer 0; remaining layers start from
  zeros
- input net: Dense(768->768) softsign + Dropout(0.2) with the kernel
  initialized to the identity matrix (official kernel_initializer=identity)
- output head: Dense(31) logits (30 character classes + blank, blank last;
  official ctc blank_index=-1)

Notes
-----
Written 2026-10-10.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class InputNet(nn.Module):
    """官方每会话输入网：Dense(softsign) + Dropout，核初始化单位阵。

    Equivalent of the official per-session input network: a linear layer
    initialized as the identity, followed by softsign activation and
    Keras-style variational dropout.
    """

    def __init__(self, n_in=768, n_out=768, dropout=0.2):
        """Initialize the identity-initialized linear layer.

        Parameters
        ----------
        n_in : int, optional
            Input feature dimension (768 for both T18 and T17 npz exports).
        n_out : int, optional
            Output feature dimension.
        dropout : float, optional
            Variational dropout probability applied in ``forward`` when
            ``training=True``.
        """
        super().__init__()
        self.fc = nn.Linear(n_in, n_out)
        with torch.no_grad():
            self.fc.weight.copy_(torch.eye(n_out, n_in))
            self.fc.bias.zero_()
        self.dropout = dropout

    def forward(self, x, training=False):
        """Apply the linear layer, softsign and optional variational dropout.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, T, F)`` (any shape ending in ``F`` works).
        training : bool, optional
            When True, apply variational dropout with a (B, 1, F) mask
            shared across time steps (Keras semantics).

        Returns
        -------
        torch.Tensor
            Transformed features with the same shape as ``x``.
        """
        x = F.softsign(self.fc(x))
        if training and self.dropout > 0:
            b = x.shape[0]
            f = x.shape[-1]
            mask = (torch.rand(b, 1, f, device=x.device) > self.dropout) / (1 - self.dropout)
            x = x * mask
        return x


class TypingGRU(nn.Module):
    """官方 GRU 类的等价移植：patch 化 -> [variational dropout -> 单层 GRU] x5 -> Dense(31)。

    PyTorch equivalent of the official GRU decoder: patch extraction
    (unfold) followed by 5 stacked single-layer GRUs with Keras-style
    variational dropout on each layer's input, then a Dense(31) logit head.

    Attributes
    ----------
    layers : nn.ModuleList
        The 5 single-layer GRUs (input dim is ``n_features * patch_size``
        for layer 0, ``units`` afterwards).
    init_states : nn.Parameter
        Learnable (1, units) initial hidden state (glorot/xavier
        initialized), used only by layer 0.
    dense : nn.Linear
        Output projection to ``n_classes`` logits.
    """

    def __init__(self, n_features=768, units=512, n_layers=5, n_classes=31,
                 patch_size=15, patch_stride=6, gru_dropout=0.6):
        """Build the stacked-GRU decoder.

        Parameters
        ----------
        n_features : int, optional
            Number of input features per time step (768 = 384 channels x 2).
        units : int, optional
            GRU hidden size per layer.
        n_layers : int, optional
            Number of stacked GRU layers.
        n_classes : int, optional
            Number of output classes (30 characters + blank).
        patch_size : int, optional
            Temporal patch length (official extract_patches size).
        patch_stride : int, optional
            Temporal stride between patch starts.
        gru_dropout : float, optional
            Variational dropout probability applied to each layer's input
            during training.
        """
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
        """Patch the input, run the stacked GRUs and project to class logits.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(B, T, F)``.
        training : bool, optional
            When True, apply variational dropout before each GRU layer.

        Returns
        -------
        torch.Tensor
            Logits of shape ``(B, T', n_classes)`` with
            ``T' = floor((T - patch_size) / patch_stride) + 1``.

        Raises
        ------
        ValueError
            If the sequence length ``T`` is shorter than ``patch_size``.
        """
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
        """Keras L2 正则等价：1e-5 * Σw²，仅作用于 GRU 输入核（weight_ih）——官方只给 kernel 加了正则。

        Parameters
        ----------
        weight_reg : float
            L2 coefficient (official weight_reg, e.g. 1e-5).

        Returns
        -------
        torch.Tensor
            Scalar ``weight_reg * sum(weight_ih ** 2)`` over all GRU input
            kernels.
        """
        reg = sum((w ** 2).sum() for name, w in self.named_parameters() if 'weight_ih' in name)
        return weight_reg * reg
