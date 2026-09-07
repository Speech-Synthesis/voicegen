import torch
import torch.nn as nn
import math


class ProsodyEncoder(nn.Module):
    """
    Prosody Encoder that converts frame/phoneme features of shape [B, T, 3]
    to embeddings of shape [B, T, Dp].
    Uses a Conv1D stack with strict padding masks to prevent feature leakage.

    Features are expected to be z-scored (mean 0, std 1) per speaker/globally.
    """
    def __init__(self, in_dim=3, hidden=256, out_dim=128,
                 n_layers=4, kernel=5, dropout=0.1):
        super().__init__()

        # Input layer normalization for stability
        self.input_ln = nn.LayerNorm(in_dim)

        # Input projection with careful initialization
        self.inp = nn.Linear(in_dim, hidden)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(n_layers)
        ])
        # LayerNorm expects features last: shape [B, T, hidden]
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.out = nn.Linear(hidden, out_dim)

        # Initialize weights for stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small values to prevent gradient explosion."""
        # Input projection: Xavier uniform scaled down
        nn.init.xavier_uniform_(self.inp.weight, gain=0.1)
        nn.init.zeros_(self.inp.bias)

        # Conv layers: small initialization
        for block in self.blocks:
            conv = block[0]  # First element is Conv1d
            nn.init.xavier_uniform_(conv.weight, gain=0.1)
            nn.init.zeros_(conv.bias)

        # Output projection: small initialization
        nn.init.xavier_uniform_(self.out.weight, gain=0.1)
        nn.init.zeros_(self.out.bias)

    def forward(self, p_feat, p_mask):
        # p_feat: [B, T, 3] - should be z-scored features
        # p_mask: [B, T] bool or float mask

        # Input normalization for stability
        h = self.input_ln(p_feat)
        h = self.inp(h)                         # [B, T, hidden]
        m = p_mask.unsqueeze(-1).float()        # [B, T, 1]

        for conv, ln in zip(self.blocks, self.norms):
            # Apply mask to block padding leakage before convolution
            masked_h = h * m
            # Conv1D expects [B, C, T]
            conv_in = masked_h.transpose(1, 2)
            conv_out = conv(conv_in)
            r = conv_out.transpose(1, 2)        # [B, T, hidden]

            # Pre-norm residual connection (more stable than post-norm)
            h = h + r * m  # Apply mask to residual
            h = ln(h)

        return self.out(h) * m                  # [B, T, out_dim], zero out padding


class ProsodyRecon(nn.Module):
    """
    Information bottleneck sanity check wrapper.
    Embeds prosody features and reconstructs them using a small decoder head.
    Used for standalone pretraining of the ProsodyEncoder.
    """
    def __init__(self, enc):
        super().__init__()
        self.enc = enc
        self.dec = nn.Sequential(
            nn.Linear(enc.out.out_features, 128),
            nn.GELU(),
            nn.Linear(128, 3)
        )

    def forward(self, p_feat, p_mask):
        return self.enc(p_feat, p_mask)

    def loss(self, p_feat, p_mask, voiced):
        # p_feat: [B, T, 3]
        # p_mask: [B, T] bool
        # voiced: [B, T] uint8/bool voicing mask
        z = self.forward(p_feat, p_mask)
        rec = self.dec(z)

        m = p_mask.unsqueeze(-1).float()

        # Pitch loss only on voiced phonemes
        vm = (p_mask.bool() & voiced.bool()).unsqueeze(-1).float()
        l_pitch = ((rec[..., :1] - p_feat[..., :1])**2 * vm).sum() / vm.sum().clamp(min=1.0)

        # Energy and duration loss on all real/valid phonemes
        l_rest = ((rec[..., 1:] - p_feat[..., 1:])**2 * m).sum() / m.sum().clamp(min=1.0)

        return l_pitch + l_rest
