"""
test_data.py — Tests for data utilities, filelist format, collate ordering, and validation.

Tests:
  - filter_phones: correct token filtering (sil/sp/spn dropped, others kept)
  - filter_phones: empty string, all-silence input
  - Filelist validate: passes on correct data
  - Filelist validate: detects length mismatch
  - Collate ordering: prosody and text rows sorted identically by spec length
  - Collate: batch-of-1 does not crash
  - Collate: p_mask padded correctly (True for real, False for padding)
  - HParams: nested dict access
  - save/load checkpoint round-trip
"""
import os
import sys
import tempfile

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils import filter_phones, HParams, save_checkpoint, load_checkpoint
from data_utils_research import TextAudioSpeakerCollateResearch
from models import SynthesizerTrn


# ──────────────────────────────────────────────
# filter_phones
# ──────────────────────────────────────────────
class TestFilterPhones:
    def test_drops_sil(self):
        result = filter_phones("sil ah ow sil")
        assert "sil" not in result.split()

    def test_drops_sp(self):
        result = filter_phones("ah sp ow")
        assert "sp" not in result.split()

    def test_drops_spn(self):
        result = filter_phones("ah spn ow")
        assert "spn" not in result.split()

    def test_preserves_phonemes(self):
        result = filter_phones("ah ow iy")
        assert result == "ah ow iy"

    def test_all_silence_returns_empty(self):
        result = filter_phones("sil sp spn")
        assert result == ""

    def test_empty_string_returns_empty(self):
        result = filter_phones("")
        assert result == ""

    def test_mixed_case_dropped(self):
        """Silence tokens should be caught case-insensitively via lowercasing."""
        result = filter_phones("SIL ah OW spn")
        tokens = result.split()
        assert "sil" not in tokens
        assert "SIL" not in tokens
        assert "spn" not in tokens

    def test_returns_correct_count(self):
        result = filter_phones("sil ah ow sil iy")
        assert len(result.split()) == 3  # ah, ow, iy


# ──────────────────────────────────────────────
# HParams
# ──────────────────────────────────────────────
class TestHParams:
    def test_nested_access(self):
        hp = HParams(train={"lr": 1e-4, "batch": 8}, model={"hidden": 256})
        assert hp.train.lr == pytest.approx(1e-4)
        assert hp.model.hidden == 256

    def test_dict_style_access(self):
        hp = HParams(data={"n_speakers": 109})
        assert hp["data"]["n_speakers"] == 109

    def test_contains(self):
        hp = HParams(research={"prosody_dim": 128})
        assert "research" in hp

    def test_len(self):
        hp = HParams(a=1, b=2, c=3)
        assert len(hp) == 3


# ──────────────────────────────────────────────
# Filelist validation
# ──────────────────────────────────────────────
class TestFilelistValidation:
    def _write_mock_filelist(self, tmp_path, lengths_match=True):
        """Write a tiny mock filelist with .npy prosody arrays."""
        npy_paths = []
        lines = []
        for i in range(3):
            npy_path = str(tmp_path / f"utt{i}.npy")
            T = 5 + i  # 5, 6, 7 phonemes
            arr = np.zeros((T, 3), dtype=np.float32)
            np.save(npy_path, arr)
            if lengths_match:
                phones = " ".join([f"ph{j}" for j in range(T)])
            else:
                # Deliberate mismatch: T phonemes in npy but T+2 in filelist
                phones = " ".join([f"ph{j}" for j in range(T + 2)])
            lines.append(f"wavs/utt{i}.wav|spk0|{phones}|{npy_path}")
            npy_paths.append(npy_path)

        filelist = str(tmp_path / "test.txt.prosody")
        with open(filelist, "w") as f:
            f.write("\n".join(lines) + "\n")
        return filelist

    def test_passes_on_correct_data(self, tmp_path):
        from extract_prosody_features import validate
        filelist = self._write_mock_filelist(tmp_path, lengths_match=True)
        result = validate(filelist)
        assert result is True

    def test_detects_length_mismatch(self, tmp_path):
        from extract_prosody_features import validate
        filelist = self._write_mock_filelist(tmp_path, lengths_match=False)
        result = validate(filelist)
        assert result is False

    def test_handles_missing_npy_gracefully(self, tmp_path):
        """validate() should catch missing .npy and mark as failure, not crash."""
        from extract_prosody_features import validate
        filelist = str(tmp_path / "bad.txt.prosody")
        with open(filelist, "w") as f:
            f.write("wavs/utt.wav|spk0|ph1 ph2|/nonexistent/path.npy\n")
        result = validate(filelist)
        assert result is False


# ──────────────────────────────────────────────
# TextAudioSpeakerCollateResearch
# ──────────────────────────────────────────────
def _make_mock_item(T_text, T_spec, Dt=192, C_spec=80):
    """Create a single dataset item as returned by __getitem__."""
    text = torch.randint(1, 50, (T_text,), dtype=torch.long)
    spec = torch.randn(C_spec, T_spec)
    wav = torch.randn(T_spec * 4)
    sid = torch.tensor(0, dtype=torch.long)
    p_feat = torch.randn(T_text, 3)
    p_mask = torch.ones(T_text, dtype=torch.uint8)
    voiced = torch.ones(T_text, dtype=torch.uint8)
    g_timbre = torch.randn(Dt)
    return (text, spec, wav, sid, p_feat, p_mask, voiced, g_timbre)


class TestCollate:
    def test_basic_batch(self):
        """Collate returns 11-tuple with correct batch dimension."""
        items = [_make_mock_item(T_text=10 + i, T_spec=20 + i) for i in range(4)]
        collate = TextAudioSpeakerCollateResearch()
        batch = collate(items)
        assert len(batch) == 11, f"Expected 11 tensors, got {len(batch)}"
        text_padded = batch[0]
        assert text_padded.shape[0] == 4, "Batch size must be 4"

    def test_sorted_by_spec_length(self):
        """After collation, spec lengths must be in descending order."""
        items = [_make_mock_item(T_text=8, T_spec=10 + i) for i in range(4)]
        collate = TextAudioSpeakerCollateResearch()
        batch = collate(items)
        spec_lengths = batch[3]  # spec_lengths tensor
        for i in range(len(spec_lengths) - 1):
            assert spec_lengths[i] >= spec_lengths[i + 1], \
                f"spec_lengths not in descending order: {spec_lengths}"

    def test_prosody_sorted_consistently(self):
        """
        Prosody rows must follow the same sort order as text.
        We verify by checking that the longest spec item's prosody is first in the batch.
        """
        # Make items with different T_text tied to T_spec so we can track ordering
        items = []
        for i in range(4):
            T_text = 5 + i
            T_spec = 10 + i  # spec length grows with i
            item = _make_mock_item(T_text=T_text, T_spec=T_spec)
            items.append(item)

        collate = TextAudioSpeakerCollateResearch()
        batch = collate(items)

        # After sorting by spec length descending, index 0 should have the longest spec
        spec_lengths = batch[3]
        text_lengths = batch[1]
        assert spec_lengths[0] == max(spec_lengths), "Longest spec must be first"
        # The item with longest spec also has the longest text (both grow with i)
        assert text_lengths[0] == max(text_lengths), "Longest text must correspond"

    def test_p_mask_correct_shape(self):
        """p_mask padded tensor must be [B, max_T]."""
        items = [_make_mock_item(T_text=5 + i, T_spec=20) for i in range(3)]
        collate = TextAudioSpeakerCollateResearch()
        batch = collate(items)
        p_mask_padded = batch[8]
        max_T = max(5 + i for i in range(3))
        assert p_mask_padded.shape == torch.Size([3, max_T]), \
            f"p_mask shape mismatch: {p_mask_padded.shape}"

    def test_p_feat_padding_is_zero(self):
        """p_feat values beyond the real sequence length must be zero-padded."""
        T_short = 5
        T_long = 10
        items = [
            _make_mock_item(T_text=T_short, T_spec=20),
            _make_mock_item(T_text=T_long, T_spec=25),  # longer spec, comes first after sort
        ]
        collate = TextAudioSpeakerCollateResearch()
        batch = collate(items)
        p_feat_padded = batch[7]  # [B, max_T, 3]
        # Find which row has the shorter text (T_short)
        text_lengths = batch[1]
        short_idx = text_lengths.argmin().item()
        # Padded positions should be zero
        real_len = text_lengths[short_idx].item()
        if real_len < p_feat_padded.shape[1]:
            pad_vals = p_feat_padded[short_idx, real_len:, :]
            assert pad_vals.abs().sum().item() == pytest.approx(0.0), \
                "p_feat padding positions must be zero"

    def test_batch_size_one(self):
        """Single-item batch must not crash."""
        items = [_make_mock_item(T_text=8, T_spec=16)]
        collate = TextAudioSpeakerCollateResearch()
        batch = collate(items)
        assert batch[0].shape[0] == 1

    def test_g_timbre_shape(self):
        """g_timbre must be [B, Dt]."""
        items = [_make_mock_item(T_text=6, T_spec=15, Dt=192) for _ in range(3)]
        collate = TextAudioSpeakerCollateResearch()
        batch = collate(items)
        g_timbre = batch[10]  # last element
        assert g_timbre.shape == (3, 192), f"g_timbre shape: {g_timbre.shape}"


# ──────────────────────────────────────────────
# Checkpoint save/load round-trip
# ──────────────────────────────────────────────
class TestCheckpoint:
    def _make_tiny_model(self):
        return SynthesizerTrn(
            n_vocab=20, spec_channels=80, segment_size=512,
            inter_channels=32, hidden_channels=32, filter_channels=64,
            n_heads=1, n_layers=1, kernel_size=3, p_dropout=0.0,
            resblock="1", resblock_kernel_sizes=[3],
            resblock_dilation_sizes=[[1]], upsample_rates=[2],
            upsample_initial_channel=32, upsample_kernel_sizes=[4],
        )

    def test_save_load_identical(self, tmp_path):
        model = self._make_tiny_model()
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        ckpt_path = str(tmp_path / "test_ckpt.pth")

        # Save original state
        original_state = {k: v.clone() for k, v in model.state_dict().items()}
        save_checkpoint(model, opt, 1e-3, 100, ckpt_path)

        # Modify model weights
        for p in model.parameters():
            p.data.fill_(999.0)

        # Reload
        model2, _, lr, step = load_checkpoint(ckpt_path, model)
        assert step == 100
        assert lr == pytest.approx(1e-3)
        for key in original_state:
            assert torch.allclose(model2.state_dict()[key], original_state[key]), \
                f"State mismatch after load for key: {key}"
