import torch
import torch.nn as nn

class ProsodyEncoder(nn.Module):
    """
    Prosody Encoder that converts frame/phoneme features of shape [B, T, 3]
    to embeddings of shape [B, T, Dp].
    Uses a Conv1D stack with strict padding masks to prevent feature leakage.
    """
    def __init__(self, in_dim=3, hidden=256, out_dim=128,
                 n_layers=4, kernel=5, dropout=0.1):
        super().__init__()
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

    def forward(self, p_feat, p_mask):
        # p_feat: [B, T, 3]
        # p_mask: [B, T] bool or float mask
        h = self.inp(p_feat)                    # [B, T, hidden]
        m = p_mask.unsqueeze(-1).float()        # [B, T, 1]
        
        for conv, ln in zip(self.blocks, self.norms):
            # Apply mask to block padding leakage before convolution
            masked_h = h * m
            # Conv1D expects [B, C, T]
            conv_in = masked_h.transpose(1, 2)
            conv_out = conv(conv_in)
            r = conv_out.transpose(1, 2)        # [B, T, hidden]
            
            # Residual + LayerNorm
            h = ln(h + r)
            
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
