"""
conftest.py — Shared pytest fixtures for HDVC test suite.
All fixtures produce small tensors so tests run fast on CPU.
"""
import sys
import os

# Ensure voicegen/ is on the path so all imports resolve
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
import numpy as np
from utils import HParams


# ──────────────────────────────────────────────
# Dimension constants used across all tests
# ──────────────────────────────────────────────
B = 4        # batch size
T = 12       # phoneme sequence length (short for speed)
H = 192      # VITS text-encoder hidden dim
Dt = 192     # timbre dim (ECAPA)
Dp = 128     # prosody dim
T_SPEC = 30  # spectrogram length
C_SPEC = 80  # mel channels


# ──────────────────────────────────────────────
# Research config fixture
# ──────────────────────────────────────────────
@pytest.fixture
def research_cfg():
    return HParams(
        use_prosody_encoder=True,
        fusion_type="cross_attention",
        disentangle_loss="cosine",
        disentangle_weight=0.1,
        prosody_dim=Dp,
        timbre_dim=Dt,
    )


@pytest.fixture
def research_cfg_mine():
    return HParams(
        use_prosody_encoder=True,
        fusion_type="cross_attention",
        disentangle_loss="mine",
        disentangle_weight=0.1,
        prosody_dim=Dp,
        timbre_dim=Dt,
    )


@pytest.fixture
def research_cfg_concat():
    return HParams(
        use_prosody_encoder=True,
        fusion_type="concat",
        disentangle_loss="cosine",
        disentangle_weight=0.1,
        prosody_dim=Dp,
        timbre_dim=Dt,
    )


# ──────────────────────────────────────────────
# Tensor fixtures
# ──────────────────────────────────────────────
@pytest.fixture
def p_feat():
    """Raw prosody features [B, T, 3]."""
    feat = torch.randn(B, T, 3)
    return feat


@pytest.fixture
def p_mask():
    """Boolean mask [B, T] — last 3 positions padded."""
    mask = torch.ones(B, T, dtype=torch.bool)
    mask[:, -3:] = False  # last 3 positions are padding
    return mask


@pytest.fixture
def voiced():
    """Voicing mask [B, T] — roughly 60% voiced."""
    v = torch.zeros(B, T, dtype=torch.uint8)
    v[:, :int(T * 0.6)] = 1
    return v


@pytest.fixture
def g_timbre():
    """Frozen ECAPA-TDNN timbre embeddings [B, Dt]."""
    return torch.randn(B, Dt)


@pytest.fixture
def text_hidden():
    """VITS text-encoder hidden states [B, T, H]."""
    return torch.randn(B, T, H)


@pytest.fixture
def spec():
    """Linear spectrogram [B, C_SPEC, T_SPEC]."""
    return torch.randn(B, C_SPEC, T_SPEC)


@pytest.fixture
def x_ids():
    """Phoneme ID sequences [B, T]."""
    return torch.randint(1, 50, (B, T), dtype=torch.long)


@pytest.fixture
def x_lengths():
    """Text sequence lengths [B] — all T for simplicity."""
    return torch.full((B,), T, dtype=torch.long)


@pytest.fixture
def spec_lengths():
    """Spectrogram frame lengths [B] — all T_SPEC."""
    return torch.full((B,), T_SPEC, dtype=torch.long)


# ──────────────────────────────────────────────
# Full hparams fixture (matches vctk_full.json schema)
# ──────────────────────────────────────────────
@pytest.fixture
def hps(research_cfg):
    return HParams(
        train=HParams(
            log_interval=1,
            eval_interval=10,
            seed=1234,
            learning_rate=2e-4,
            betas=[0.8, 0.99],
            eps=1e-9,
            batch_size=B,
            fp16_run=False,
            segment_size=8192,
        ),
        data=HParams(
            n_speakers=10,
            sampling_rate=22050,
            filter_length=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=C_SPEC,
        ),
        model=HParams(
            inter_channels=H,
            hidden_channels=H,
            filter_channels=H * 4,
            n_heads=2,
            n_layers=2,
            kernel_size=3,
            p_dropout=0.0,
            resblock="1",
            resblock_kernel_sizes=[3, 7],
            resblock_dilation_sizes=[[1, 3], [1, 3]],
            upsample_rates=[4, 4],
            upsample_initial_channel=128,
            upsample_kernel_sizes=[8, 8],
            gin_channels=Dt,
        ),
        research=research_cfg,
    )
