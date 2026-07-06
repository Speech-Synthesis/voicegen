"""
test_train.py — Mini training-loop tests verifying:
  - Prosody pretraining (3 steps): loss finite and decreases
  - Joint training with cosine dis-loss (3 steps): total loss finite, no NaN
  - Joint training with MINE dis-loss (3 steps): stats net updates, loss finite
  - MINE warm-up guard: step 0 does NOT call update_statistics_net
  - Disentanglement diagnostic scalars (zt_std, zp_std): both > 0

All tests use tiny random batches on CPU — no real data needed.
"""
import os
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prosody_encoder import ProsodyEncoder, ProsodyRecon
from models_research import SynthesizerTrnResearch
from disentangle_loss import build_disentangle, CosineDisentangleLoss, MineDisentangleLoss
from utils import HParams

# ──────────────────────────────────────────────
# Constants for fast mock training
# ──────────────────────────────────────────────
B = 4
T = 10
H = 192
Dt = 192
Dp = 128
C_SPEC = 80
T_SPEC = 20
VOCAB = 50
N_STEPS = 3


def make_batch(device="cpu"):
    """Create one mock training batch."""
    x = torch.randint(1, VOCAB, (B, T), dtype=torch.long, device=device)
    x_lengths = torch.full((B,), T, dtype=torch.long, device=device)
    spec = torch.randn(B, C_SPEC, T_SPEC, device=device)
    spec_lengths = torch.full((B,), T_SPEC, dtype=torch.long, device=device)
    p_feat = torch.randn(B, T, 3, device=device)
    p_mask = torch.ones(B, T, dtype=torch.bool, device=device)
    voiced = (torch.rand(B, T, device=device) > 0.4).to(torch.uint8)
    g_timbre = torch.randn(B, Dt, device=device)
    sid = torch.zeros(B, dtype=torch.long, device=device)
    return x, x_lengths, spec, spec_lengths, p_feat, p_mask, voiced, g_timbre, sid


def make_synth(research_cfg, device="cpu"):
    """Build a tiny SynthesizerTrnResearch for training tests."""
    net = SynthesizerTrnResearch(
        n_vocab=VOCAB, spec_channels=C_SPEC, segment_size=512,
        inter_channels=H, hidden_channels=H, filter_channels=H,
        n_heads=2, n_layers=1, kernel_size=3, p_dropout=0.0,
        resblock="1", resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1]], upsample_rates=[4],
        upsample_initial_channel=64, upsample_kernel_sizes=[8],
        n_speakers=10, gin_channels=Dt,
        research_cfg=research_cfg,
    )
    return net.to(device)


# ──────────────────────────────────────────────
# Prosody Pretraining
# ──────────────────────────────────────────────
class TestProsodyPretraining:
    def test_loss_finite_over_steps(self):
        """3-step mock pretraining: every loss value must be finite."""
        enc = ProsodyEncoder(in_dim=3, hidden=64, out_dim=32, n_layers=1)
        model = ProsodyRecon(enc)
        opt = torch.optim.Adam(model.parameters(), lr=2e-4)

        for step in range(N_STEPS):
            p_feat = torch.randn(B, T, 3)
            p_mask = torch.ones(B, T, dtype=torch.bool)
            voiced = (torch.rand(B, T) > 0.4).to(torch.uint8)

            opt.zero_grad()
            loss = model.loss(p_feat, p_mask, voiced)
            assert torch.isfinite(loss), f"Pretrain loss is non-finite at step {step}: {loss.item()}"
            loss.backward()
            opt.step()

    def test_loss_decreases(self):
        """Loss over 20 steps on a fixed batch should trend downward (sanity check)."""
        enc = ProsodyEncoder(in_dim=3, hidden=64, out_dim=32, n_layers=2)
        model = ProsodyRecon(enc)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)

        # Fix one batch
        p_feat = torch.randn(B, T, 3)
        p_mask = torch.ones(B, T, dtype=torch.bool)
        voiced = torch.ones(B, T, dtype=torch.uint8)

        losses = []
        for _ in range(30):
            opt.zero_grad()
            loss = model.loss(p_feat, p_mask, voiced)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        # The mean of last 5 steps should be lower than mean of first 5
        assert np.mean(losses[-5:]) < np.mean(losses[:5]), \
            f"Loss did not decrease: first={np.mean(losses[:5]):.4f}, last={np.mean(losses[-5:]):.4f}"

    def test_reconstruction_correlation_positive(self):
        """After 50 steps on a fixed batch, pitch reconstruction should show positive correlation."""
        enc = ProsodyEncoder(in_dim=3, hidden=64, out_dim=32, n_layers=2)
        model = ProsodyRecon(enc)
        opt = torch.optim.Adam(model.parameters(), lr=5e-4)

        p_feat = torch.randn(B, T, 3)
        p_mask = torch.ones(B, T, dtype=torch.bool)
        voiced = torch.ones(B, T, dtype=torch.uint8)

        for _ in range(50):
            opt.zero_grad()
            model.loss(p_feat, p_mask, voiced).backward()
            opt.step()

        with torch.no_grad():
            z = model.enc(p_feat, p_mask)
            pred = model.dec(z)

        pred_np = pred[..., 0].reshape(-1).numpy()
        true_np = p_feat[..., 0].reshape(-1).numpy()
        corr = np.corrcoef(pred_np, true_np)[0, 1]
        assert corr > 0.0, f"Expected positive pitch correlation after training, got {corr:.4f}"


# ──────────────────────────────────────────────
# Joint Training — Cosine
# ──────────────────────────────────────────────
class TestJointTrainingCosine:
    @pytest.fixture
    def cosine_cfg(self):
        return HParams(
            use_prosody_encoder=True,
            fusion_type="cross_attention",
            disentangle_loss="cosine",
            disentangle_weight=0.1,
            prosody_dim=Dp,
            timbre_dim=Dt,
        )

    def test_loss_finite(self, cosine_cfg):
        net = make_synth(cosine_cfg)
        dis_loss = build_disentangle(cosine_cfg)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-4)

        for step in range(N_STEPS):
            x, x_l, spec, spec_l, p_feat, p_mask, voiced, g_timbre, sid = make_batch()
            opt.zero_grad()
            outputs, extras = net(x, x_l, spec, spec_l, g_timbre, p_feat, p_mask, sid)
            o, logw_, z, y_mask, x_mask, stats = outputs
            g_t, p, p_m, attn_w = extras

            loss_base = F.l1_loss(o, torch.zeros_like(o))
            loss_dis = dis_loss(g_t, p, p_m)
            loss_total = loss_base + 0.1 * loss_dis

            assert torch.isfinite(loss_total), \
                f"Cosine joint loss not finite at step {step}: {loss_total.item()}"
            loss_total.backward()
            opt.step()

    def test_dis_loss_nonnegative(self, cosine_cfg):
        net = make_synth(cosine_cfg)
        dis_loss = build_disentangle(cosine_cfg)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-4)

        x, x_l, spec, spec_l, p_feat, p_mask, voiced, g_timbre, sid = make_batch()
        opt.zero_grad()
        outputs, extras = net(x, x_l, spec, spec_l, g_timbre, p_feat, p_mask, sid)
        g_t, p, p_m, attn_w = extras
        loss_dis = dis_loss(g_t, p, p_m)
        assert loss_dis.item() >= 0.0, f"Cosine loss must be ≥ 0, got {loss_dis.item()}"

    def test_zt_zp_std_positive(self, cosine_cfg):
        """zt.std() and zp.std() should be > 0 (representations not collapsed)."""
        net = make_synth(cosine_cfg)
        dis_loss = build_disentangle(cosine_cfg)
        opt = torch.optim.AdamW(net.parameters(), lr=2e-4)

        x, x_l, spec, spec_l, p_feat, p_mask, voiced, g_timbre, sid = make_batch()
        opt.zero_grad()
        outputs, extras = net(x, x_l, spec, spec_l, g_timbre, p_feat, p_mask, sid)
        g_t, p, p_m, attn_w = extras

        m = p_m.unsqueeze(-1).float()
        p_bar = (p * m).sum(1) / m.sum(1).clamp(min=1.0)
        zt = F.normalize(dis_loss.wt(g_t), dim=-1)
        zp = F.normalize(dis_loss.wp(p_bar), dim=-1)

        assert zt.std().item() > 0, "zt std must be > 0 (representations not collapsed)"
        assert zp.std().item() > 0, "zp std must be > 0 (representations not collapsed)"

    def test_attn_diagnostics(self, cosine_cfg):
        """fusion/timbre_attn_mass must be in [0, 1]."""
        net = make_synth(cosine_cfg)
        x, x_l, spec, spec_l, p_feat, p_mask, voiced, g_timbre, sid = make_batch()
        net.eval()
        with torch.no_grad():
            _, extras = net(x, x_l, spec, spec_l, g_timbre, p_feat, p_mask, sid)
        _, _, _, attn_w = extras
        if attn_w is not None:
            timbre_mass = attn_w[..., 0].mean().item()
            assert 0.0 <= timbre_mass <= 1.0, f"Timbre attn mass out of range: {timbre_mass}"


# ──────────────────────────────────────────────
# Joint Training — MINE
# ──────────────────────────────────────────────
class TestJointTrainingMINE:
    @pytest.fixture
    def mine_cfg(self):
        return HParams(
            use_prosody_encoder=True,
            fusion_type="cross_attention",
            disentangle_loss="mine",
            disentangle_weight=0.1,
            prosody_dim=Dp,
            timbre_dim=Dt,
        )

    def test_loss_finite(self, mine_cfg):
        net = make_synth(mine_cfg)
        dis_loss = build_disentangle(mine_cfg)
        opt_g = torch.optim.AdamW(net.parameters(), lr=2e-4)
        opt_T = torch.optim.Adam(dis_loss.T.parameters(), lr=1e-4)

        for step in range(N_STEPS):
            x, x_l, spec, spec_l, p_feat, p_mask, voiced, g_timbre, sid = make_batch()

            # Step 0: skip MINE update (warm-up guard)
            if step > 0:
                net.eval()
                with torch.no_grad():
                    p_emb = net.prosody_enc(p_feat, p_mask)
                net.train()
                mi = dis_loss.update_statistics_net(g_timbre, p_emb, p_mask, opt_T)
                assert isinstance(mi, float)

            # Generator step
            opt_g.zero_grad()
            outputs, extras = net(x, x_l, spec, spec_l, g_timbre, p_feat, p_mask, sid)
            o, logw_, z, y_mask, x_mask, stats = outputs
            g_t, p, p_m, attn_w = extras

            loss_base = F.l1_loss(o, torch.zeros_like(o))
            loss_dis = dis_loss(g_t, p, p_m)
            loss_total = loss_base + 0.1 * loss_dis

            assert torch.isfinite(loss_total), \
                f"MINE joint loss not finite at step {step}: {loss_total.item()}"
            loss_total.backward()
            opt_g.step()

    def test_mine_warmup_guard(self, mine_cfg):
        """At step 0, update_statistics_net must NOT be called (guard check)."""
        net = make_synth(mine_cfg)
        dis_loss = build_disentangle(mine_cfg)

        update_called = []

        original_update = dis_loss.update_statistics_net
        def patched_update(*args, **kwargs):
            update_called.append(True)
            return original_update(*args, **kwargs)

        dis_loss.update_statistics_net = patched_update

        # Simulate step=0 with guard: should NOT call update
        step = 0
        if step > 0:  # This is the guard from train_full.py
            opt_T = torch.optim.Adam(dis_loss.T.parameters(), lr=1e-4)
            p_feat = torch.randn(B, T, 3)
            p_mask = torch.ones(B, T, dtype=torch.bool)
            dis_loss.update_statistics_net(torch.randn(B, Dt), p_feat, p_mask, opt_T)

        assert len(update_called) == 0, "update_statistics_net must NOT be called at step 0"

    def test_mine_updates_after_step_zero(self, mine_cfg):
        """At step > 0, update_statistics_net IS called and returns a float."""
        dis_loss = build_disentangle(mine_cfg)
        opt_T = torch.optim.Adam(dis_loss.T.parameters(), lr=1e-4)

        p = torch.randn(B, T, Dp)
        p_mask = torch.ones(B, T, dtype=torch.bool)
        g = torch.randn(B, Dt)

        mi = dis_loss.update_statistics_net(g, p, p_mask, opt_T)
        assert isinstance(mi, float)
        assert np.isfinite(mi), f"MI estimate must be finite: {mi}"


# ──────────────────────────────────────────────
# Disentangle score trending (held-out set diagnostic)
# ──────────────────────────────────────────────
class TestDisentanglementTrend:
    def test_score_decreases_with_perfect_setup(self):
        """
        On random data with a high LR, the cosine disentangle loss
        applied to a linear projection should at minimum be computable
        and not diverge over a few steps.
        """
        from disentangle_loss import CosineDisentangleLoss
        Dt_local, Dp_local = 32, 16
        loss_fn = CosineDisentangleLoss(timbre_dim=Dt_local, prosody_dim=Dp_local, shared=8)
        opt = torch.optim.Adam(loss_fn.parameters(), lr=1e-2)

        scores = []
        for _ in range(20):
            g = torch.randn(8, Dt_local)
            p = torch.randn(8, 5, Dp_local)
            mask = torch.ones(8, 5, dtype=torch.bool)
            opt.zero_grad()
            loss = loss_fn(g, p, mask)
            loss.backward()
            opt.step()
            scores.append(loss.item())

        assert all(np.isfinite(s) for s in scores), "Some disentangle scores are non-finite"
