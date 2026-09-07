import torch
import torch.nn as nn
import torch.nn.functional as F


class CosineDisentangleLoss(nn.Module):
    """
    Penalizes similarity between speaker timbre and pooled prosody representations
    using squared cosine similarity (orthogonality constraint) in a shared space.
    Includes a variance preservation guard (VICReg-style) to prevent trivial representation collapse.
    """
    def __init__(self, timbre_dim=192, prosody_dim=128, shared=64, use_var_guard=True):
        super().__init__()
        self.wt = nn.Linear(timbre_dim, shared, bias=False)
        self.wp = nn.Linear(prosody_dim, shared, bias=False)
        self.use_var_guard = use_var_guard

        # Initialize with small weights
        nn.init.xavier_uniform_(self.wt.weight, gain=0.1)
        nn.init.xavier_uniform_(self.wp.weight, gain=0.1)

    def var_guard(self, z, eps=1e-4):
        # Variance preservation term: pushes variance of each dimension to be >= 1.0
        # z: [B, shared]
        if z.size(0) <= 1:
            return torch.tensor(0.0, device=z.device, dtype=z.dtype)
        std = torch.sqrt(z.var(dim=0) + eps)
        return F.relu(1.0 - std).mean()

    def forward(self, g, p, p_mask):
        # g: [B, Dt] speaker timbre embedding
        # p: [B, T, Dp] prosody embedding sequence
        # p_mask: [B, T] sequence mask
        m = p_mask.unsqueeze(-1).float()

        # Mean pool prosody embedding sequence across valid time frames
        denom = m.sum(1).clamp(min=1.0)  # [B, 1]
        p_bar = (p * m).sum(1) / denom   # [B, Dp]

        # Project to shared space
        zt_proj = self.wt(g)   # [B, shared]
        zp_proj = self.wp(p_bar)  # [B, shared]

        # Safe normalize: compute norm and clamp to prevent division by zero
        zt_norm = torch.norm(zt_proj, dim=-1, keepdim=True).clamp(min=1e-6)
        zp_norm = torch.norm(zp_proj, dim=-1, keepdim=True).clamp(min=1e-6)
        zt = zt_proj / zt_norm
        zp = zp_proj / zp_norm

        # Squared cosine similarity: values near 0 mean orthogonality (desired)
        # Clamp dot product to [-1, 1] to prevent numerical issues
        dot = (zt * zp).sum(-1).clamp(-1.0, 1.0)
        similarity = dot.pow(2).mean()

        loss = similarity
        if self.use_var_guard:
            v_t = self.var_guard(zt)
            v_p = self.var_guard(zp)
            loss = loss + 0.5 * (v_t + v_p)

        return loss


class MineDisentangleLoss(nn.Module):
    """
    Mutual Information Neural Estimator (MINE) based on the Donsker-Varadhan bound.
    The statistics network self.T is trained to maximize the bound (estimate MI),
    while the generator is penalized with the bound to minimize MI.
    """
    def __init__(self, timbre_dim=192, prosody_dim=128, hidden=128):
        super().__init__()
        self.T = nn.Sequential(
            nn.Linear(timbre_dim + prosody_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        self.register_buffer("ema", torch.tensor(1.0))

        # Initialize with small weights
        for m in self.T:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def get_p_bar(self, p, p_mask):
        m = p_mask.unsqueeze(-1).float()
        denom = m.sum(1).clamp(min=1.0)
        return (p * m).sum(1) / denom

    def dv_bound(self, g, p_bar):
        # g: [B, Dt]
        # p_bar: [B, Dp]
        B = g.size(0)

        # Joint samples: (X, Y) pairs
        joint = self.T(torch.cat([g, p_bar], dim=-1)).mean()

        # Marginal samples: shuffle prosody representations across batch
        perm_idx = torch.randperm(B, device=g.device)
        p_bar_perm = p_bar[perm_idx]

        # Marginal expectation term: E[exp(T(X, Y'))]
        # Clamp to prevent overflow
        t_marginal = self.T(torch.cat([g, p_bar_perm], dim=-1))
        t_marginal_clamped = torch.clamp(t_marginal, max=10.0)  # Prevent exp overflow
        et = torch.exp(t_marginal_clamped).mean()

        # Update exponential moving average of et (needed for stable gradients of log term)
        if self.training:
            with torch.no_grad():
                self.ema.copy_(0.99 * self.ema + 0.01 * et)

        # Bias-corrected Donsker-Varadhan bound estimate
        dv_estimate = joint - torch.log(et + 1e-8) * (et.detach() / (self.ema + 1e-8))
        return dv_estimate

    def forward(self, g, p, p_mask):
        """
        Computes the MI penalty for the generator (VITS).
        Returns the DV bound (clamped to 0) which the generator seeks to minimize.
        """
        p_bar = self.get_p_bar(p, p_mask)
        mi = self.dv_bound(g.detach(), p_bar)
        # Clamp to [0, +inf) to avoid rewarding early negative bounds
        return F.relu(mi)

    def update_statistics_net(self, g, p, p_mask, optimizer_T):
        """
        Updates the statistics network weights.
        Should be called inside the training loop to MAXIMIZE the MI bound.
        """
        self.T.train()
        p_bar = self.get_p_bar(p, p_mask)

        # We want to maximize the bound, so we minimize negative bound
        # detach inputs so gradients don't flow back to the generator/encoder
        loss_T = -self.dv_bound(g.detach(), p_bar.detach())

        optimizer_T.zero_grad()
        loss_T.backward()
        # Clip MINE gradients too
        torch.nn.utils.clip_grad_norm_(self.T.parameters(), 1.0)
        optimizer_T.step()

        return -loss_T.item()


def build_disentangle(cfg):
    """
    Factory function to build the appropriate disentanglement loss based on config.
    """
    if cfg.disentangle_loss == "none":
        return None
    elif cfg.disentangle_loss == "cosine":
        return CosineDisentangleLoss(
            timbre_dim=cfg.timbre_dim,
            prosody_dim=cfg.prosody_dim,
            shared=64
        )
    elif cfg.disentangle_loss == "mine":
        return MineDisentangleLoss(
            timbre_dim=cfg.timbre_dim,
            prosody_dim=cfg.prosody_dim,
            hidden=128
        )
    else:
        raise ValueError(f"Unknown disentangle loss type: {cfg.disentangle_loss}")
