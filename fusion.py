import torch
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    """
    Fuses local prosody [B, T, Dp] and global timbre [B, Dt] representations
    with text encoder states [B, T, H] via multi-head cross-attention.
    """
    def __init__(self, h_dim=192, timbre_dim=192, prosody_dim=128,
                 n_heads=2, dropout=0.1):
        super().__init__()
        self.kv_t = nn.Linear(timbre_dim, h_dim)      # timbre -> KV space
        self.kv_p = nn.Linear(prosody_dim, h_dim)     # prosody -> KV space
        self.attn = nn.MultiheadAttention(h_dim, n_heads,
                                          dropout=dropout, batch_first=True)
        self.ln = nn.LayerNorm(h_dim)

    def forward(self, x, g, p, p_mask):
        # x: [B, T, H]  (text encoder hidden states as queries)
        # g: [B, Dt]    (global timbre vector)
        # p: [B, T, Dp] (prosody embeddings)
        # p_mask: [B, T] bool (True = valid, False = pad)
        B, T, H = x.shape
        
        # Project conditioning to shared key/value space
        t_tok = self.kv_t(g).unsqueeze(1)            # [B, 1, H]
        p_tok = self.kv_p(p)                         # [B, T, H]
        
        # Concatenate: first token is global timbre, remaining T tokens are local prosody
        kv = torch.cat([t_tok, p_tok], dim=1)        # [B, 1+T, H]
        
        # Padding mask: key_padding_mask requires True for tokens that should be IGNORED.
        # Global timbre token (index 0) is always attendable.
        # Prosody tokens are ignored if not in p_mask.
        pad_t = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
        pad_p = ~p_mask.bool()                       # True for padding positions
        pad = torch.cat([pad_t, pad_p], dim=1)       # [B, 1+T]
        
        # Multi-head attention: Q = x, K = V = kv
        out, attn_w = self.attn(x, kv, kv, key_padding_mask=pad,
                                need_weights=True, average_attn_weights=True)
        
        return self.ln(x + out), attn_w

class ConcatFusion(nn.Module):
    """
    Simple concatenation ablation module. Concatenates text encoder states,
    timbre broadcast, and prosody embeddings along feature dimension and projects back.
    """
    def __init__(self, h_dim=192, timbre_dim=192, prosody_dim=128):
        super().__init__()
        self.proj = nn.Linear(h_dim + timbre_dim + prosody_dim, h_dim)

    def forward(self, x, g, p, p_mask):
        # Broadcast timbre across time dimension T
        g_b = g.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, T, Dt]
        
        # Concatenate: [B, T, H + Dt + Dp]
        concat_feat = torch.cat([x, g_b, p], dim=-1)
        out = self.proj(concat_feat)
        
        # Mask padding features
        m = p_mask.unsqueeze(-1).float()
        return out * m, None

def build_fusion(cfg, h_dim):
    """
    Factory function to build the appropriate fusion module based on config.
    """
    if cfg.fusion_type == "cross_attention":
        return CrossAttentionFusion(
            h_dim=h_dim,
            timbre_dim=cfg.timbre_dim,
            prosody_dim=cfg.prosody_dim
        )
    elif cfg.fusion_type == "concat":
        return ConcatFusion(
            h_dim=h_dim,
            timbre_dim=cfg.timbre_dim,
            prosody_dim=cfg.prosody_dim
        )
    else:
        raise ValueError(f"Unknown fusion type: {cfg.fusion_type}")
