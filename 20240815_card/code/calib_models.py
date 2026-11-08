# -*- coding: utf-8 -*-
"""BiGRUDecoder: Yoon 2026-style upgrade of the official GRUDecoder.

Same day-layer + patch front-end and CTC interface; changes:
- 5-layer GRU is bidirectional;
- 'expanded classification head': Linear(2*n_units -> n_units) + GELU +
  Linear(n_units -> n_classes) instead of the single Linear.

Drop-in compatible with the official trainer via monkeypatching
rnn_model.GRUDecoder before rnn_trainer is imported (see train_bigrun.py).
"""
import torch
from torch import nn


class BiGRUDecoder(nn.Module):
    """Bidirectional GRU CTC decoder with day-layer and optional patching.

    A drop-in upgrade of the official (unidirectional) ``GRUDecoder`` from the
    Card et al. release: the 5-layer GRU is made bidirectional and the single
    linear classification head is replaced by an "expanded" head
    (Linear -> GELU -> Linear). The per-day affine front-end (learnable
    per-day weight matrix + bias followed by softsign) and the CTC logit
    interface are unchanged, so it can be monkeypatched into the official
    trainer (see ``train_bigrun.py``).
    """

    def __init__(self, neural_dim, n_units, n_days, n_classes,
                 rnn_dropout=0.0, input_dropout=0.0, n_layers=5,
                 patch_size=0, patch_stride=0):
        """Initialize the layers and parameters of the decoder.

        Parameters
        ----------
        neural_dim : int
            Number of input neural features per time step.
        n_units : int
            Hidden size of each GRU layer (per direction).
        n_days : int
            Number of recording days; one learnable day weight/bias pair is
            created per day.
        n_classes : int
            Number of output classes (CTC vocabulary size, including blank).
        rnn_dropout : float, optional
            Dropout between GRU layers (0 disables it). Default 0.0.
        input_dropout : float, optional
            Dropout applied after the day layer (0 disables it). Default 0.0.
        n_layers : int, optional
            Number of stacked GRU layers. Default 5.
        patch_size : int, optional
            Number of consecutive time steps grouped into one patch; 0
            disables patching. Default 0.
        patch_stride : int, optional
            Step between consecutive patches (only used when ``patch_size >
            0``). Default 0.
        """
        super().__init__()
        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_days = n_days
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        self.day_layer_activation = nn.Softsign()
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(neural_dim)) for _ in range(n_days)])
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, neural_dim)) for _ in range(n_days)])
        self.day_layer_dropout = nn.Dropout(input_dropout)

        self.input_size = neural_dim
        if patch_size > 0:
            self.input_size *= patch_size

        self.gru = nn.GRU(
            input_size=self.input_size, hidden_size=n_units,
            num_layers=n_layers, dropout=rnn_dropout, batch_first=True,
            bidirectional=True)
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        self.out = nn.Sequential(
            nn.Linear(2 * n_units, n_units),
            nn.GELU(),
            nn.Linear(n_units, n_classes),
        )
        nn.init.xavier_uniform_(self.out[0].weight)
        nn.init.xavier_uniform_(self.out[2].weight)

        # one learnable init state per direction, shared across layers
        self.h0 = nn.Parameter(nn.init.xavier_uniform_(torch.zeros(2, 1, n_units)))

    def forward(self, x, day_idx, states=None, return_state=False):
        """Run the decoder on a batch of neural sequences.

        Applies the per-day affine layer and softsign activation, optionally
        patches consecutive time steps, then runs the bidirectional GRU and
        the expanded classification head to produce frame-wise CTC logits.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape ``(batch, time, neural_dim)``.
        day_idx : sequence of int
            Recording-day index (into the learnable day weights/biases) for
            each batch element.
        states : torch.Tensor, optional
            Initial GRU hidden state of shape ``(n_layers * 2, batch,
            n_units)``. If None, a learnable init state is broadcast to the
            batch.
        return_state : bool, optional
            If True, also return the final GRU hidden state. Default False.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            Logits of shape ``(batch, time, n_classes)`` if ``return_state``
            is False; otherwise a tuple ``(logits, hidden_states)``.
        """
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([self.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)
        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_layer_activation(x)
        if self.day_layer_dropout.p > 0:
            x = self.day_layer_dropout(x)

        if self.patch_size > 0:
            x = x.unsqueeze(1).permute(0, 3, 1, 2)
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)
            x_unfold = x_unfold.squeeze(2).permute(0, 2, 3, 1)
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)

        if states is None:
            states = self.h0.repeat(self.n_layers, 1, 1)  # (layers*2, 1, units)
            states = states.expand(-1, x.shape[0], -1).contiguous()

        output, hidden_states = self.gru(x, states)
        logits = self.out(output)
        if return_state:
            return logits, hidden_states
        return logits
