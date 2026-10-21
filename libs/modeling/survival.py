"""Causal multi-state survival heads used by SurvVTG.

The head consumes the query-conditioned level-0 HEM feature sequence.  Every
prediction at index ``t`` is computed from indices ``<= t``; unlike a standard
same-padded Conv1d, the causal convolutions never expose a future feature.
"""

from copy import deepcopy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import LayerNorm


class CausalMaskedConv1D(nn.Module):
    """Left-padded Conv1d that preserves a boolean temporal mask."""

    def __init__(self, in_dim, out_dim, kernel_size=3, dilation=1, bias=True):
        super().__init__()
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_dim,
            out_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias,
        )

    def forward(self, x, mask):
        if mask is None:
            mask = torch.ones_like(x[:, :1], dtype=torch.bool)
        x = x * mask.to(dtype=x.dtype)
        x = self.conv(F.pad(x, (self.left_padding, 0)))
        return x * mask.to(dtype=x.dtype), mask


def _logit(prob):
    prob = min(max(float(prob), 1e-5), 1.0 - 1e-5)
    return math.log(prob / (1.0 - prob))


class SurvivalHead(nn.Module):
    """Predict start/end hazards, dynamic cure, and event-state logits.

    Inputs are HEM FPN tuples with tensors shaped ``[B, W, D, T]``.  The
    survival process is defined on the finest (level-0) time grid and returns
    tensors shaped ``[B, W, T]`` plus state logits ``[B, W, T, 3]``.
    """

    def __init__(
        self,
        embd_dim,
        hidden_dim=256,
        n_layers=2,
        kernel_size=3,
        dropout=0.1,
        prior_start=0.01,
        prior_end=0.01,
        prior_cure=0.10,
        enable_cure=True,
        enable_state=True,
    ):
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")

        dims = [embd_dim] + [hidden_dim] * n_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for layer_idx in range(n_layers):
            self.convs.append(
                CausalMaskedConv1D(
                    dims[layer_idx],
                    dims[layer_idx + 1],
                    kernel_size=kernel_size,
                    dilation=2 ** layer_idx,
                    bias=False,
                )
            )
            self.norms.append(LayerNorm(hidden_dim))
        self.dropout = nn.Dropout(dropout)

        self.start = CausalMaskedConv1D(hidden_dim, 1, kernel_size=1)
        self.end = CausalMaskedConv1D(hidden_dim, 1, kernel_size=1)
        self.cure = CausalMaskedConv1D(hidden_dim, 1, kernel_size=1) if enable_cure else None
        self.state = CausalMaskedConv1D(hidden_dim, 3, kernel_size=1) if enable_state else None

        nn.init.constant_(self.start.conv.bias, _logit(prior_start))
        nn.init.constant_(self.end.conv.bias, _logit(prior_end))
        if self.cure is not None:
            nn.init.constant_(self.cure.conv.bias, _logit(prior_cure))

    def forward(self, fpn, fpn_masks):
        x, mask = fpn[0], fpn_masks[0]
        batch_size, num_windows, channels, seq_len = x.shape
        x = x.reshape(batch_size * num_windows, channels, seq_len)
        mask = mask.reshape(batch_size * num_windows, 1, seq_len)

        for conv, norm in zip(self.convs, self.norms):
            x, mask = conv(x, mask)
            x = self.dropout(F.gelu(norm(x)))

        start_logits, _ = self.start(x, mask)
        end_logits, _ = self.end(x, mask)
        cure_logits = self.cure(x, mask)[0] if self.cure is not None else None
        state_logits = self.state(x, mask)[0] if self.state is not None else None

        def scalar_view(value):
            return value.squeeze(1).reshape(batch_size, num_windows, seq_len)

        return {
            "start_logits": scalar_view(start_logits),
            "end_logits": scalar_view(end_logits),
            "cure_logits": scalar_view(cure_logits) if cure_logits is not None else None,
            "state_logits": state_logits.transpose(1, 2).reshape(
                batch_size, num_windows, seq_len, 3
            ) if state_logits is not None else None,
            "mask": mask.squeeze(1).reshape(batch_size, num_windows, seq_len),
        }


def make_survival_head(opt):
    return SurvivalHead(**deepcopy(opt))
