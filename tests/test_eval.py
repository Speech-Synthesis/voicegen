"""
test_eval.py — Smoke tests for the evaluation pipeline (eval.py).

Tests:
  - compute_correlation: known inputs produce expected Pearson r
  - compute_wer: exact match → WER = 0; fully wrong → WER > 0
  - evaluate_pairs same_reference: produces metrics.json + pair_metrics.csv
  - evaluate_pairs cross_reference: produces correct files with correct keys
  - metrics.json has required keys
  - pair_metrics.csv has correct row count
  - Edge case: empty voiced mask in pitch correlation
"""
import os
import sys
import json
import csv
import tempfile

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval as eval_module  # import the whole module to access functions


# ──────────────────────────────────────────────
# compute_correlation
# ──────────────────────────────────────────────
class TestComputeCorrelation:
    def test_perfect_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = eval_module.compute_correlation(x, x)
        assert r == pytest.approx(1.0, abs=1e-5)

    def test_perfect_negative_correlation(self):
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = -x
        r = eval_module.compute_correlation(x, y)
        assert r == pytest.approx(-1.0, abs=1e-5)

    def test_zero_correlation_constant(self):
        """Constant arrays → zero denominator → returns 0."""
        x = np.ones(5)
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        r = eval_module.compute_correlation(x, y)
        assert r == pytest.approx(0.0, abs=1e-5)

    def test_short_array_returns_zero(self):
        """Arrays shorter than 2 elements → returns 0.0."""
        r = eval_module.compute_correlation(np.array([1.0]), np.array([2.0]))
        assert r == 0.0

    def test_finite_output(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(50)
        y = rng.standard_normal(50)
        r = eval_module.compute_correlation(x, y)
        assert np.isfinite(r)
        assert -1.0 <= r <= 1.0


# ──────────────────────────────────────────────
# compute_wer
# ──────────────────────────────────────────────
class TestComputeWer:
    def test_exact_match_zero_wer(self):
        wer = eval_module.compute_wer("hello world", "hello world")
        assert wer == pytest.approx(0.0, abs=1e-5)

    def test_completely_wrong_nonzero(self):
        wer = eval_module.compute_wer("abc def ghi", "xyz uvw rst")
        assert wer > 0.0

    def test_partial_match(self):
        wer_partial = eval_module.compute_wer("hello world foo", "hello world bar")
        assert 0.0 < wer_partial < 100.0

    def test_case_insensitive(self):
        """WER normalises to lowercase, so casing should not matter."""
        wer = eval_module.compute_wer("Hello World", "hello world")
        assert wer == pytest.approx(0.0, abs=1e-5)

    def test_punctuation_stripped(self):
        """Punctuation is stripped, so 'hello.' == 'hello'."""
        wer = eval_module.compute_wer("hello.", "hello")
        assert wer == pytest.approx(0.0, abs=1e-5)

    def test_returns_percentage(self):
        """WER is returned as a percentage (0–100 scale)."""
        wer = eval_module.compute_wer("one two three", "one two three")
        assert wer < 1.0  # 0% WER


# ──────────────────────────────────────────────
# evaluate_pairs — same_reference
# ──────────────────────────────────────────────
REQUIRED_METRICS_KEYS = {
    "model", "track", "n_pairs",
    "secs_timbre", "secs_prosody_ref_leak",
    "utmos", "wer", "pitch_corr", "energy_corr", "dis_score",
}


def make_mock_pairs(mode, count=3):
    pairs = []
    for i in range(count):
        if mode == "same_reference":
            pairs.append({
                "text": f"This is sentence {i}.",
                "ref_wav": f"nonexistent_wav_{i}.wav"
            })
        else:
            pairs.append({
                "text": f"This is sentence {i}.",
                "wav_A": f"nonexistent_A_{i}.wav",
                "wav_B": f"nonexistent_B_{i}.wav",
            })
    return pairs


class TestEvaluatePairsSameReference:
    def test_produces_metrics_json(self, tmp_path):
        pairs = make_mock_pairs("same_reference", count=3)
        stats_path = str(tmp_path / "stats.json")
        out_dir = str(tmp_path / "out_same")

        eval_module.evaluate_pairs(
            net_g=None,
            pairs=pairs,
            mode="same_reference",
            out_dir=out_dir,
            stats_path=stats_path,
            device="cpu",
        )

        metrics_path = os.path.join(out_dir, "metrics.json")
        assert os.path.exists(metrics_path), "metrics.json must be created"

    def test_metrics_json_has_required_keys(self, tmp_path):
        pairs = make_mock_pairs("same_reference", count=3)
        out_dir = str(tmp_path / "out_same_keys")

        eval_module.evaluate_pairs(
            net_g=None, pairs=pairs, mode="same_reference",
            out_dir=out_dir, stats_path="", device="cpu",
        )

        with open(os.path.join(out_dir, "metrics.json")) as f:
            metrics = json.load(f)

        for key in REQUIRED_METRICS_KEYS:
            assert key in metrics, f"Missing key in metrics.json: {key}"

    def test_metrics_values_finite(self, tmp_path):
        pairs = make_mock_pairs("same_reference", count=3)
        out_dir = str(tmp_path / "out_same_finite")

        eval_module.evaluate_pairs(
            net_g=None, pairs=pairs, mode="same_reference",
            out_dir=out_dir, stats_path="", device="cpu",
        )

        with open(os.path.join(out_dir, "metrics.json")) as f:
            metrics = json.load(f)

        for key in REQUIRED_METRICS_KEYS - {"model", "track"}:
            val = metrics[key]
            if isinstance(val, (int, float)):
                assert np.isfinite(val), f"metrics.json[{key}] is not finite: {val}"

    def test_produces_csv(self, tmp_path):
        pairs = make_mock_pairs("same_reference", count=5)
        out_dir = str(tmp_path / "out_same_csv")

        eval_module.evaluate_pairs(
            net_g=None, pairs=pairs, mode="same_reference",
            out_dir=out_dir, stats_path="", device="cpu",
        )

        csv_path = os.path.join(out_dir, "pair_metrics.csv")
        assert os.path.exists(csv_path), "pair_metrics.csv must be created"

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 5, f"Expected 5 rows in CSV, got {len(rows)}"

    def test_n_pairs_matches(self, tmp_path):
        N = 4
        pairs = make_mock_pairs("same_reference", count=N)
        out_dir = str(tmp_path / "out_same_npairs")

        eval_module.evaluate_pairs(
            net_g=None, pairs=pairs, mode="same_reference",
            out_dir=out_dir, stats_path="", device="cpu",
        )

        with open(os.path.join(out_dir, "metrics.json")) as f:
            metrics = json.load(f)
        assert metrics["n_pairs"] == N


# ──────────────────────────────────────────────
# evaluate_pairs — cross_reference
# ──────────────────────────────────────────────
class TestEvaluatePairsCrossReference:
    def test_produces_metrics_json(self, tmp_path):
        pairs = make_mock_pairs("cross_reference", count=3)
        out_dir = str(tmp_path / "out_cross")

        eval_module.evaluate_pairs(
            net_g=None, pairs=pairs, mode="cross_reference",
            out_dir=out_dir, stats_path="", device="cpu",
        )

        assert os.path.exists(os.path.join(out_dir, "metrics.json"))

    def test_track_field_is_cross_reference(self, tmp_path):
        pairs = make_mock_pairs("cross_reference", count=2)
        out_dir = str(tmp_path / "out_cross_track")

        eval_module.evaluate_pairs(
            net_g=None, pairs=pairs, mode="cross_reference",
            out_dir=out_dir, stats_path="", device="cpu",
        )

        with open(os.path.join(out_dir, "metrics.json")) as f:
            metrics = json.load(f)
        assert metrics["track"] == "cross_reference"

    def test_secs_leak_present(self, tmp_path):
        """cross_reference must report secs_prosody_ref_leak (leak detector)."""
        pairs = make_mock_pairs("cross_reference", count=3)
        out_dir = str(tmp_path / "out_cross_leak")

        eval_module.evaluate_pairs(
            net_g=None, pairs=pairs, mode="cross_reference",
            out_dir=out_dir, stats_path="", device="cpu",
        )

        with open(os.path.join(out_dir, "metrics.json")) as f:
            metrics = json.load(f)
        assert "secs_prosody_ref_leak" in metrics
        assert isinstance(metrics["secs_prosody_ref_leak"], float)

    def test_csv_has_all_pairs(self, tmp_path):
        N = 6
        pairs = make_mock_pairs("cross_reference", count=N)
        out_dir = str(tmp_path / "out_cross_csv")

        eval_module.evaluate_pairs(
            net_g=None, pairs=pairs, mode="cross_reference",
            out_dir=out_dir, stats_path="", device="cpu",
        )

        with open(os.path.join(out_dir, "pair_metrics.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == N


# ──────────────────────────────────────────────
# Edge cases
# ──────────────────────────────────────────────
class TestEdgeCases:
    def test_empty_voiced_mask_pitch_corr(self, tmp_path):
        """When no voiced frames exist, evaluate_pairs should not crash."""
        pairs = make_mock_pairs("same_reference", count=1)
        out_dir = str(tmp_path / "out_edge")

        # Should complete without exception even if no voiced frames are detected
        eval_module.evaluate_pairs(
            net_g=None, pairs=pairs, mode="same_reference",
            out_dir=out_dir, stats_path="", device="cpu",
        )
        assert os.path.exists(os.path.join(out_dir, "metrics.json"))

    def test_stats_json_loaded_gracefully(self, tmp_path):
        """evaluate_pairs should not crash when stats.json exists and is valid JSON."""
        stats = {"global": {"duration_mean": 0.0, "duration_std": 1.0}}
        stats_path = str(tmp_path / "stats.json")
        with open(stats_path, "w") as f:
            json.dump(stats, f)

        pairs = make_mock_pairs("same_reference", count=2)
        out_dir = str(tmp_path / "out_stats")

        eval_module.evaluate_pairs(
            net_g=None, pairs=pairs, mode="same_reference",
            out_dir=out_dir, stats_path=stats_path, device="cpu",
        )
        assert os.path.exists(os.path.join(out_dir, "metrics.json"))

    def test_single_pair(self, tmp_path):
        """Evaluation on exactly 1 pair must produce valid output."""
        pairs = make_mock_pairs("same_reference", count=1)
        out_dir = str(tmp_path / "out_single")

        eval_module.evaluate_pairs(
            net_g=None, pairs=pairs, mode="same_reference",
            out_dir=out_dir, stats_path="", device="cpu",
        )

        with open(os.path.join(out_dir, "metrics.json")) as f:
            metrics = json.load(f)
        assert metrics["n_pairs"] == 1
