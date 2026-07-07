"""
Integration tests for models.py
Verifies compatibility with models_research.py and train_full.py
"""

import torch
import sys
import os

# Test configuration
BATCH_SIZE = 2
TEXT_LEN = 50
SPEC_LEN = 100
N_VOCAB = 100
SPEC_CHANNELS = 80
INTER_CHANNELS = 192
HIDDEN_CHANNELS = 192
FILTER_CHANNELS = 768
N_HEADS = 2
N_LAYERS = 6
KERNEL_SIZE = 3
P_DROPOUT = 0.1
N_SPEAKERS = 109
GIN_CHANNELS = 192

def test_basic_imports():
    """Test 1: Basic imports"""
    print("\n" + "="*60)
    print("TEST 1: Basic Imports")
    print("="*60)

    try:
        from models import (
            SynthesizerTrn,
            MultiPeriodDiscriminator,
            TextEncoder,
            PosteriorEncoder,
            Generator,
            ResidualCouplingBlock,
            StochasticDurationPredictor
        )
        print("[OK] All imports successful")
        return True
    except Exception as e:
        print(f"[FAIL] Import failed: {e}")
        return False


def test_text_encoder_interface():
    """Test 2: TextEncoder has encode() and project() methods"""
    print("\n" + "="*60)
    print("TEST 2: TextEncoder Interface (models_research.py compatibility)")
    print("="*60)

    try:
        from models import TextEncoder

        encoder = TextEncoder(
            n_vocab=N_VOCAB,
            out_channels=INTER_CHANNELS,
            hidden_channels=HIDDEN_CHANNELS,
            filter_channels=FILTER_CHANNELS,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            kernel_size=KERNEL_SIZE,
            p_dropout=P_DROPOUT
        )

        # Check methods exist
        assert hasattr(encoder, 'encode'), "Missing encode() method"
        assert hasattr(encoder, 'project'), "Missing project() method"
        assert hasattr(encoder, 'forward'), "Missing forward() method"
        print("[OK] All required methods exist")

        # Test encode() method
        x = torch.randint(0, N_VOCAB, (BATCH_SIZE, TEXT_LEN))
        x_lengths = torch.LongTensor([TEXT_LEN, TEXT_LEN-10])

        h, x_mask = encoder.encode(x, x_lengths)
        print(f"[OK] encode() output shapes: h={h.shape}, x_mask={x_mask.shape}")

        assert h.shape == (BATCH_SIZE, HIDDEN_CHANNELS, TEXT_LEN), f"Wrong h shape: {h.shape}"
        assert x_mask.shape == (BATCH_SIZE, 1, TEXT_LEN), f"Wrong x_mask shape: {x_mask.shape}"
        print("[OK] encode() output shapes correct")

        # Test project() method
        m, logs = encoder.project(h, x_mask)
        print(f"[OK] project() output shapes: m={m.shape}, logs={logs.shape}")

        assert m.shape == (BATCH_SIZE, INTER_CHANNELS, TEXT_LEN), f"Wrong m shape: {m.shape}"
        assert logs.shape == (BATCH_SIZE, INTER_CHANNELS, TEXT_LEN), f"Wrong logs shape: {logs.shape}"
        print("[OK] project() output shapes correct")

        return True

    except Exception as e:
        print(f"[FAIL] TextEncoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_synthesizer_forward():
    """Test 3: SynthesizerTrn forward pass"""
    print("\n" + "="*60)
    print("TEST 3: SynthesizerTrn Forward Pass")
    print("="*60)

    try:
        from models import SynthesizerTrn

        model = SynthesizerTrn(
            n_vocab=N_VOCAB,
            spec_channels=SPEC_CHANNELS,
            segment_size=32,
            inter_channels=INTER_CHANNELS,
            hidden_channels=HIDDEN_CHANNELS,
            filter_channels=FILTER_CHANNELS,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            kernel_size=KERNEL_SIZE,
            p_dropout=P_DROPOUT,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[8, 8, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[16, 16, 4, 4],
            n_speakers=N_SPEAKERS,
            gin_channels=GIN_CHANNELS,
            use_sdp=True
        )

        print(f"[OK] Model initialized")

        # Prepare inputs
        x = torch.randint(0, N_VOCAB, (BATCH_SIZE, TEXT_LEN))
        x_lengths = torch.LongTensor([TEXT_LEN, TEXT_LEN-10])
        spec = torch.randn(BATCH_SIZE, SPEC_CHANNELS, SPEC_LEN)
        spec_lengths = torch.LongTensor([SPEC_LEN, SPEC_LEN-20])
        sid = torch.LongTensor([0, 1])

        # Forward pass
        outputs = model(x, x_lengths, spec, spec_lengths, sid=sid)
        print(f"[OK] Forward pass successful")

        # Unpack outputs (matches train_full.py:205)
        o, logw_, z, y_mask, x_mask, stats = outputs
        m_p, logs_p, m_q, logs_q = stats

        print(f"[OK] Output unpacking successful")
        print(f"  o (waveform): {o.shape}")
        print(f"  logw_ (log duration): {logw_.shape}")
        print(f"  z (latent): {z.shape}")
        print(f"  y_mask: {y_mask.shape}")
        print(f"  x_mask: {x_mask.shape}")
        print(f"  m_p (prior mean): {m_p.shape}")
        print(f"  logs_p (prior logvar): {logs_p.shape}")
        print(f"  m_q (posterior mean): {m_q.shape}")
        print(f"  logs_q (posterior logvar): {logs_q.shape}")

        # Check no NaN/Inf
        assert not torch.isnan(o).any(), "NaN in output waveform"
        assert not torch.isinf(o).any(), "Inf in output waveform"
        print(f"[OK] No NaN/Inf in outputs")

        return True

    except Exception as e:
        print(f"[FAIL] SynthesizerTrn forward test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_g_timbre_conditioning():
    """Test 4: g_timbre conditioning (research mode)"""
    print("\n" + "="*60)
    print("TEST 4: ECAPA-TDNN Embedding Conditioning (Research Mode)")
    print("="*60)

    try:
        from models import SynthesizerTrn

        model = SynthesizerTrn(
            n_vocab=N_VOCAB,
            spec_channels=SPEC_CHANNELS,
            segment_size=32,
            inter_channels=INTER_CHANNELS,
            hidden_channels=HIDDEN_CHANNELS,
            filter_channels=FILTER_CHANNELS,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            kernel_size=KERNEL_SIZE,
            p_dropout=P_DROPOUT,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[8, 8, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[16, 16, 4, 4],
            n_speakers=0,  # No speaker embedding table
            gin_channels=GIN_CHANNELS,
            use_sdp=True
        )

        print(f"[OK] Model initialized (no speaker table)")

        # Prepare inputs
        x = torch.randint(0, N_VOCAB, (BATCH_SIZE, TEXT_LEN))
        x_lengths = torch.LongTensor([TEXT_LEN, TEXT_LEN-10])
        spec = torch.randn(BATCH_SIZE, SPEC_CHANNELS, SPEC_LEN)
        spec_lengths = torch.LongTensor([SPEC_LEN, SPEC_LEN-20])
        g_timbre = torch.randn(BATCH_SIZE, GIN_CHANNELS)  # ECAPA-TDNN embedding

        # Forward pass with g_timbre
        outputs = model(x, x_lengths, spec, spec_lengths, g_timbre=g_timbre)
        print(f"[OK] Forward pass with g_timbre successful")

        o, logw_, z, y_mask, x_mask, stats = outputs
        assert not torch.isnan(o).any(), "NaN in output"
        print(f"[OK] g_timbre conditioning works correctly")

        return True

    except Exception as e:
        print(f"[FAIL] g_timbre conditioning test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_research_model_compatibility():
    """Test 5: models_research.py compatibility"""
    print("\n" + "="*60)
    print("TEST 5: models_research.py Compatibility")
    print("="*60)

    try:
        from models_research import SynthesizerTrnResearch
        from utils import HParams

        # Create research config
        research_cfg = HParams(
            use_prosody_encoder=True,
            prosody_dim=128,
            timbre_dim=GIN_CHANNELS,
            fusion_type="cross_attention",
            fusion_heads=4,
            disentangle_loss="cosine",
            disentangle_weight=0.1
        )

        model = SynthesizerTrnResearch(
            n_vocab=N_VOCAB,
            spec_channels=SPEC_CHANNELS,
            segment_size=32,
            inter_channels=INTER_CHANNELS,
            hidden_channels=HIDDEN_CHANNELS,
            filter_channels=FILTER_CHANNELS,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            kernel_size=KERNEL_SIZE,
            p_dropout=P_DROPOUT,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[8, 8, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[16, 16, 4, 4],
            n_speakers=0,
            gin_channels=GIN_CHANNELS,
            use_sdp=True,
            research_cfg=research_cfg
        )

        print(f"[OK] SynthesizerTrnResearch initialized")

        # Prepare inputs
        x = torch.randint(0, N_VOCAB, (BATCH_SIZE, TEXT_LEN))
        x_lengths = torch.LongTensor([TEXT_LEN, TEXT_LEN-10])
        spec = torch.randn(BATCH_SIZE, SPEC_CHANNELS, SPEC_LEN)
        spec_lengths = torch.LongTensor([SPEC_LEN, SPEC_LEN-20])
        g_timbre = torch.randn(BATCH_SIZE, GIN_CHANNELS)
        p_feat = torch.randn(BATCH_SIZE, TEXT_LEN, 3)  # pitch, energy, duration
        p_mask = torch.ones(BATCH_SIZE, TEXT_LEN).byte()

        # Forward pass
        outputs, extras = model(x, x_lengths, spec, spec_lengths, g_timbre, p_feat, p_mask)
        print(f"[OK] Research model forward pass successful")

        # Unpack
        o, logw_, z, y_mask, x_mask, stats = outputs
        g_t, p, p_mask_out, attn_w = extras

        print(f"[OK] Research model output unpacking successful")
        print(f"  g_t (timbre): {g_t.shape}")
        print(f"  p (prosody): {p.shape if p is not None else None}")
        print(f"  attn_w (attention): {attn_w.shape if attn_w is not None else None}")

        assert not torch.isnan(o).any(), "NaN in output"
        print(f"[OK] Research model produces valid outputs")

        return True

    except Exception as e:
        print(f"[FAIL] Research model compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_discriminator():
    """Test 6: MultiPeriodDiscriminator"""
    print("\n" + "="*60)
    print("TEST 6: MultiPeriodDiscriminator")
    print("="*60)

    try:
        from models import MultiPeriodDiscriminator

        disc = MultiPeriodDiscriminator()
        print(f"[OK] Discriminator initialized")

        # Generate fake waveforms
        wav_real = torch.randn(BATCH_SIZE, 1, 8192)
        wav_fake = torch.randn(BATCH_SIZE, 1, 8192)

        # Forward pass
        y_d_rs, y_d_gs, fmap_rs, fmap_gs = disc(wav_real, wav_fake)
        print(f"[OK] Discriminator forward pass successful")

        print(f"  Number of period discriminators: {len(y_d_rs)}")
        print(f"  Real predictions shape: {[y.shape for y in y_d_rs]}")
        print(f"  Fake predictions shape: {[y.shape for y in y_d_gs]}")
        print(f"  Number of feature maps: {len(fmap_rs)}")

        # Check outputs
        assert len(y_d_rs) == 5, "Should have 5 period discriminators"
        assert len(y_d_gs) == 5, "Should have 5 period discriminators"
        assert len(fmap_rs) == 5, "Should have 5 feature map lists"
        assert len(fmap_gs) == 5, "Should have 5 feature map lists"
        print(f"[OK] Discriminator outputs correct")

        return True

    except Exception as e:
        print(f"[FAIL] Discriminator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_inference_mode():
    """Test 7: Inference mode"""
    print("\n" + "="*60)
    print("TEST 7: Inference Mode")
    print("="*60)

    try:
        from models import SynthesizerTrn

        model = SynthesizerTrn(
            n_vocab=N_VOCAB,
            spec_channels=SPEC_CHANNELS,
            segment_size=32,
            inter_channels=INTER_CHANNELS,
            hidden_channels=HIDDEN_CHANNELS,
            filter_channels=FILTER_CHANNELS,
            n_heads=N_HEADS,
            n_layers=N_LAYERS,
            kernel_size=KERNEL_SIZE,
            p_dropout=P_DROPOUT,
            resblock="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_rates=[8, 8, 2, 2],
            upsample_initial_channel=512,
            upsample_kernel_sizes=[16, 16, 4, 4],
            n_speakers=N_SPEAKERS,
            gin_channels=GIN_CHANNELS,
            use_sdp=True
        )

        model.eval()
        print(f"[OK] Model in eval mode")

        # Prepare inputs
        x = torch.randint(0, N_VOCAB, (1, TEXT_LEN))
        x_lengths = torch.LongTensor([TEXT_LEN])
        sid = torch.LongTensor([0])

        # Inference
        with torch.no_grad():
            o, attn, y_mask, (z, z_p, m_p, logs_p) = model.infer(
                x, x_lengths, sid=sid,
                noise_scale=0.667,
                length_scale=1.0,
                noise_scale_w=0.8
            )

        print(f"[OK] Inference successful")
        print(f"  Output waveform shape: {o.shape}")
        print(f"  Attention shape: {attn.shape}")
        print(f"  y_mask shape: {y_mask.shape}")

        assert o.shape[0] == 1, "Batch size should be 1"
        assert o.shape[1] == 1, "Should be mono audio"
        assert not torch.isnan(o).any(), "NaN in output"
        print(f"[OK] Inference output valid")

        return True

    except Exception as e:
        print(f"[FAIL] Inference mode test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "="*60)
    print("MODELS.PY INTEGRATION TEST SUITE")
    print("="*60)

    tests = [
        ("Basic Imports", test_basic_imports),
        ("TextEncoder Interface", test_text_encoder_interface),
        ("SynthesizerTrn Forward", test_synthesizer_forward),
        ("g_timbre Conditioning", test_g_timbre_conditioning),
        ("Research Model Compatibility", test_research_model_compatibility),
        ("MultiPeriodDiscriminator", test_discriminator),
        ("Inference Mode", test_inference_mode),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n[FAIL] Test '{name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "[OK] PASS" if result else "[FAIL] FAIL"
        print(f"{status}: {name}")

    print("="*60)
    print(f"TOTAL: {passed}/{total} tests passed")
    print("="*60)

    if passed == total:
        print("\n[OK] ALL TESTS PASSED - models.py is ready!")
        return 0
    else:
        print(f"\n[FAIL] {total - passed} test(s) failed - review errors above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
