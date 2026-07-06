"""
test_shapes.py — Forward-pass shape and NaN assertions for all core modules.

Tests:
  - ProsodyEncoder: input/output shapes, padding zeroed
  - CrossAttentionFusion: output shape, attn_w shape, residual preserved
  - ConcatFusion: output shape, returns None for attn_w
  - CosineDisentangleLoss: scalar output, finite, no NaN
  - MineDisentangleLoss: dv_bound finite, forward clamped ≥ 0
  - SynthesizerTrnResearch.forward: output shapes + extras tuple
  - SynthesizerTrnResearch.infer: audio tensor non-zero, no NaN
  - build_fusion factory: returns correct class for config
  - build_disentangle factory: returns correct class / None
"""
import torch
import pytest

from prosody_encoder import ProsodyEncoder, ProsodyRecon
from fusion import CrossAttentionFusion, ConcatFusion, build_fusion
from disentangle_loss import CosineDisentangleLoss, MineDisentangleLoss, build_disentangle
from models_research import SynthesizerTrnResearch


# ──────────────────────────────────────────────
# ProsodyEncoder
# ──────────────────────────────────────────────
class TestProsodyEncoder:
    def test_output_shape(self, p_feat, p_mask):
        enc = ProsodyEncoder(in_dim=3, hidden=64, out_dim=128, n_layers=2)
        out = enc(p_feat, p_mask)
        assert out.shape == (p_feat.shape[0], p_feat.shape[1], 128), \
            f"Expected [{p_feat.shape[0]}, {p_feat.shape[1]}, 128], got {out.shape}"

    def test_padding_zeroed(self, p_feat, p_mask):
        """Positions where p_mask is False must produce zero output."""
        enc = ProsodyEncoder(in_dim=3, hidden=64, out_dim=128, n_layers=2)
        out = enc(p_feat, p_mask)
        pad_positions = ~p_mask  # [B, T] where True = padding
        # All padded output values should be 0
        assert out[pad_positions].abs().max().item() == pytest.approx(0.0), \
            "Padded positions should output zeros"

    def test_no_nan(self, p_feat, p_mask):
        enc = ProsodyEncoder(in_dim=3, hidden=64, out_dim=128, n_layers=2)
        out = enc(p_feat, p_mask)
        assert not torch.isnan(out).any(), "NaN in ProsodyEncoder output"

    def test_configurable_out_dim(self, p_feat, p_mask):
        for out_dim in [64, 128, 256]:
            enc = ProsodyEncoder(in_dim=3, hidden=64, out_dim=out_dim, n_layers=1)
            out = enc(p_feat, p_mask)
            assert out.shape[-1] == out_dim

    def test_recon_loss_finite(self, p_feat, p_mask, voiced):
        enc = ProsodyEncoder(in_dim=3, hidden=64, out_dim=32, n_layers=1)
        recon = ProsodyRecon(enc)
        loss = recon.loss(p_feat, p_mask, voiced)
        assert loss.item() > 0, "Reconstruction loss should be positive"
        assert torch.isfinite(loss), "Reconstruction loss must be finite"


# ──────────────────────────────────────────────
# CrossAttentionFusion
# ──────────────────────────────────────────────
class TestCrossAttentionFusion:
    def test_output_shape(self, text_hidden, g_timbre, p_mask):
        B, T, H = text_hidden.shape
        Dp = 64
        p = torch.randn(B, T, Dp)
        fusion = CrossAttentionFusion(h_dim=H, timbre_dim=g_timbre.shape[1],
                                      prosody_dim=Dp, n_heads=2)
        out, attn_w = fusion(text_hidden, g_timbre, p, p_mask)
        assert out.shape == text_hidden.shape, \
            f"Fusion output must preserve shape {text_hidden.shape}, got {out.shape}"

    def test_attn_weight_shape(self, text_hidden, g_timbre, p_mask):
        B, T, H = text_hidden.shape
        Dp = 64
        p = torch.randn(B, T, Dp)
        fusion = CrossAttentionFusion(h_dim=H, timbre_dim=g_timbre.shape[1],
                                      prosody_dim=Dp, n_heads=2)
        _, attn_w = fusion(text_hidden, g_timbre, p, p_mask)
        # attn_w: [B, T_query, T_key] = [B, T, 1+T]
        assert attn_w is not None
        assert attn_w.shape[0] == B
        assert attn_w.shape[1] == T

    def test_no_nan(self, text_hidden, g_timbre, p_mask):
        B, T, H = text_hidden.shape
        Dp = 64
        p = torch.randn(B, T, Dp)
        fusion = CrossAttentionFusion(h_dim=H, timbre_dim=g_timbre.shape[1],
                                      prosody_dim=Dp, n_heads=2)
        out, _ = fusion(text_hidden, g_timbre, p, p_mask)
        assert not torch.isnan(out).any(), "NaN in CrossAttentionFusion output"

    def test_timbre_attn_mass_logged(self, text_hidden, g_timbre, p_mask):
        """The first column of attn_w is timbre attention — check it is a valid probability."""
        B, T, H = text_hidden.shape
        Dp = 64
        p = torch.randn(B, T, Dp)
        fusion = CrossAttentionFusion(h_dim=H, timbre_dim=g_timbre.shape[1],
                                      prosody_dim=Dp, n_heads=2)
        _, attn_w = fusion(text_hidden, g_timbre, p, p_mask)
        timbre_mass = attn_w[..., 0].mean().item()
        assert 0.0 <= timbre_mass <= 1.0, f"Timbre attention mass out of [0,1]: {timbre_mass}"


# ──────────────────────────────────────────────
# ConcatFusion
# ──────────────────────────────────────────────
class TestConcatFusion:
    def test_output_shape(self, text_hidden, g_timbre, p_mask):
        B, T, H = text_hidden.shape
        Dp = 64
        p = torch.randn(B, T, Dp)
        fusion = ConcatFusion(h_dim=H, timbre_dim=g_timbre.shape[1], prosody_dim=Dp)
        out, attn_w = fusion(text_hidden, g_timbre, p, p_mask)
        assert out.shape == text_hidden.shape
        assert attn_w is None, "ConcatFusion should return None for attn_w"

    def test_no_nan(self, text_hidden, g_timbre, p_mask):
        B, T, H = text_hidden.shape
        Dp = 64
        p = torch.randn(B, T, Dp)
        fusion = ConcatFusion(h_dim=H, timbre_dim=g_timbre.shape[1], prosody_dim=Dp)
        out, _ = fusion(text_hidden, g_timbre, p, p_mask)
        assert not torch.isnan(out).any()


# ──────────────────────────────────────────────
# build_fusion factory
# ──────────────────────────────────────────────
class TestBuildFusion:
    def test_cross_attention(self, research_cfg):
        fusion = build_fusion(research_cfg, h_dim=192)
        assert isinstance(fusion, CrossAttentionFusion)

    def test_concat(self, research_cfg_concat):
        fusion = build_fusion(research_cfg_concat, h_dim=192)
        assert isinstance(fusion, ConcatFusion)

    def test_invalid_type_raises(self, research_cfg):
        from utils import HParams
        bad_cfg = HParams(fusion_type="invalid", timbre_dim=192, prosody_dim=128)
        with pytest.raises(ValueError):
            build_fusion(bad_cfg, h_dim=192)


# ──────────────────────────────────────────────
# CosineDisentangleLoss
# ──────────────────────────────────────────────
class TestCosineDisentangleLoss:
    def test_scalar_output(self, g_timbre, p_mask):
        B, T = p_mask.shape
        Dp = 128
        p = torch.randn(B, T, Dp)
        loss_fn = CosineDisentangleLoss(timbre_dim=g_timbre.shape[1], prosody_dim=Dp)
        loss = loss_fn(g_timbre, p, p_mask)
        assert loss.shape == torch.Size([]), f"Expected scalar, got {loss.shape}"

    def test_finite_positive(self, g_timbre, p_mask):
        B, T = p_mask.shape
        Dp = 128
        p = torch.randn(B, T, Dp)
        loss_fn = CosineDisentangleLoss(timbre_dim=g_timbre.shape[1], prosody_dim=Dp)
        loss = loss_fn(g_timbre, p, p_mask)
        assert torch.isfinite(loss), f"Loss must be finite, got {loss.item()}"
        assert loss.item() >= 0.0, f"Cosine loss must be non-negative, got {loss.item()}"

    def test_no_nan(self, g_timbre, p_mask):
        B, T = p_mask.shape
        Dp = 128
        p = torch.randn(B, T, Dp)
        loss_fn = CosineDisentangleLoss(timbre_dim=g_timbre.shape[1], prosody_dim=Dp)
        loss = loss_fn(g_timbre, p, p_mask)
        assert not torch.isnan(loss)

    def test_var_guard_activates(self):
        """With perfectly correlated projections, var_guard should add a nonzero penalty."""
        B, Dp, Dt = 8, 128, 192
        p = torch.ones(B, 5, Dp)
        g = torch.ones(B, Dt)
        mask = torch.ones(B, 5, dtype=torch.bool)
        loss_fn = CosineDisentangleLoss(timbre_dim=Dt, prosody_dim=Dp, use_var_guard=True)
        loss = loss_fn(g, p, mask)
        assert torch.isfinite(loss)


# ──────────────────────────────────────────────
# MineDisentangleLoss
# ──────────────────────────────────────────────
class TestMineDisentangleLoss:
    def test_dv_bound_finite(self, g_timbre, p_mask):
        B, T = p_mask.shape
        Dp = 128
        p = torch.randn(B, T, Dp)
        loss_fn = MineDisentangleLoss(timbre_dim=g_timbre.shape[1], prosody_dim=Dp)
        p_bar = loss_fn.get_p_bar(p, p_mask)
        bound = loss_fn.dv_bound(g_timbre, p_bar)
        assert torch.isfinite(bound), f"DV bound must be finite, got {bound.item()}"

    def test_forward_nonnegative(self, g_timbre, p_mask):
        """forward() clamps output to [0, +inf) — must never be negative."""
        B, T = p_mask.shape
        Dp = 128
        p = torch.randn(B, T, Dp)
        loss_fn = MineDisentangleLoss(timbre_dim=g_timbre.shape[1], prosody_dim=Dp)
        penalty = loss_fn(g_timbre, p, p_mask)
        assert penalty.item() >= 0.0, f"MINE forward must be ≥ 0, got {penalty.item()}"

    def test_statistics_update_returns_scalar(self, g_timbre, p_mask):
        """update_statistics_net must return a float MI estimate."""
        B, T = p_mask.shape
        Dp = 128
        p = torch.randn(B, T, Dp)
        loss_fn = MineDisentangleLoss(timbre_dim=g_timbre.shape[1], prosody_dim=Dp)
        opt_T = torch.optim.Adam(loss_fn.T.parameters(), lr=1e-4)
        mi = loss_fn.update_statistics_net(g_timbre, p, p_mask, opt_T)
        assert isinstance(mi, float), f"MI estimate must be float, got {type(mi)}"
        assert not torch.tensor(mi).isnan(), "MI estimate must not be NaN"


# ──────────────────────────────────────────────
# build_disentangle factory
# ──────────────────────────────────────────────
class TestBuildDisentangle:
    def test_none_mode(self, research_cfg):
        from utils import HParams
        cfg = HParams(disentangle_loss="none", timbre_dim=192, prosody_dim=128)
        assert build_disentangle(cfg) is None

    def test_cosine_mode(self, research_cfg):
        result = build_disentangle(research_cfg)
        assert isinstance(result, CosineDisentangleLoss)

    def test_mine_mode(self, research_cfg_mine):
        result = build_disentangle(research_cfg_mine)
        assert isinstance(result, MineDisentangleLoss)

    def test_invalid_mode_raises(self):
        from utils import HParams
        cfg = HParams(disentangle_loss="unknown", timbre_dim=192, prosody_dim=128)
        with pytest.raises(ValueError):
            build_disentangle(cfg)


# ──────────────────────────────────────────────
# SynthesizerTrnResearch — forward + infer
# ──────────────────────────────────────────────
@pytest.fixture
def synth(research_cfg):
    """Small SynthesizerTrnResearch for shape testing."""
    return SynthesizerTrnResearch(
        n_vocab=50,
        spec_channels=80,
        segment_size=512,
        inter_channels=192,
        hidden_channels=192,
        filter_channels=192,
        n_heads=2,
        n_layers=2,
        kernel_size=3,
        p_dropout=0.0,
        resblock="1",
        resblock_kernel_sizes=[3],
        resblock_dilation_sizes=[[1, 3]],
        upsample_rates=[4],
        upsample_initial_channel=64,
        upsample_kernel_sizes=[8],
        n_speakers=10,
        gin_channels=192,
        research_cfg=research_cfg,
    )


class TestSynthesizerTrnResearch:
    def test_forward_output_tuple(self, synth, x_ids, x_lengths, spec,
                                  spec_lengths, g_timbre, p_feat, p_mask):
        synth.eval()
        with torch.no_grad():
            outputs, extras = synth(x_ids, x_lengths, spec, spec_lengths,
                                    g_timbre, p_feat, p_mask)
        assert isinstance(outputs, tuple), "outputs must be a tuple"
        assert isinstance(extras, tuple), "extras must be a tuple"
        assert len(extras) == 4, "extras must have 4 elements (g_t, p, p_mask, attn_w)"

    def test_forward_audio_shape(self, synth, x_ids, x_lengths, spec,
                                  spec_lengths, g_timbre, p_feat, p_mask):
        synth.eval()
        with torch.no_grad():
            outputs, extras = synth(x_ids, x_lengths, spec, spec_lengths,
                                    g_timbre, p_feat, p_mask)
        o = outputs[0]  # generated audio
        assert o.ndim >= 2, "Audio output must be at least 2D"
        assert not torch.isnan(o).any(), "NaN in synthesizer audio output"

    def test_forward_preserves_batch(self, synth, x_ids, x_lengths, spec,
                                      spec_lengths, g_timbre, p_feat, p_mask):
        synth.eval()
        with torch.no_grad():
            outputs, extras = synth(x_ids, x_lengths, spec, spec_lengths,
                                    g_timbre, p_feat, p_mask)
        assert outputs[0].shape[0] == p_feat.shape[0], "Batch dim must be preserved"

    def test_extras_prosody_shape(self, synth, x_ids, x_lengths, spec,
                                   spec_lengths, g_timbre, p_feat, p_mask):
        synth.eval()
        with torch.no_grad():
            _, extras = synth(x_ids, x_lengths, spec, spec_lengths,
                              g_timbre, p_feat, p_mask)
        g_t, p, p_mask_out, attn_w = extras
        B, T, _ = p_feat.shape
        assert p.shape == (B, T, 128), \
            f"Prosody embedding shape mismatch: expected ({B},{T},128), got {p.shape}"

    def test_attn_w_not_none(self, synth, x_ids, x_lengths, spec,
                              spec_lengths, g_timbre, p_feat, p_mask):
        synth.eval()
        with torch.no_grad():
            _, extras = synth(x_ids, x_lengths, spec, spec_lengths,
                              g_timbre, p_feat, p_mask)
        _, _, _, attn_w = extras
        assert attn_w is not None, "attn_w should not be None when prosody encoder is on"

    def test_infer_no_nan(self, synth, x_ids, x_lengths, g_timbre, p_feat, p_mask):
        synth.eval()
        with torch.no_grad():
            o, y_mask, attn_w = synth.infer(x_ids, x_lengths, g_timbre, p_feat, p_mask)
        assert not torch.isnan(o).any(), "NaN in synthesizer infer output"
        assert o.numel() > 0, "Infer output must be non-empty"
