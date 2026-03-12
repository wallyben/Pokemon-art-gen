"""
Extended tests for StencilScorer and factory quality threshold.

Covers:
- The new region_size_score metric.
- Updated default weights (0.35 / 0.25 / 0.25 / 0.15).
- FactoryConfig.quality_threshold field.
- FactoryRunner quality threshold filtering (mocked, no SD inference).

All tests use synthetic images; no GPU or internet access required.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from pokemon_stencil.config import FactoryConfig, PipelineConfig, ProcessingConfig
from pokemon_stencil.factory.scoring import StencilScorer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _solid(value: int = 128, h: int = 64, w: int = 64) -> Image.Image:
    arr = np.full((h, w, 3), value, dtype=np.uint8)
    return Image.fromarray(arr)


def _checkerboard(h: int = 64, w: int = 64, tile: int = 8) -> Image.Image:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if ((y // tile) + (x // tile)) % 2 == 0:
                arr[y, x] = 255
    return Image.fromarray(arr)


def _two_region(h: int = 64, w: int = 64) -> Image.Image:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, w // 2:] = 200
    return Image.fromarray(arr)


def _many_tiny_dots(h: int = 64, w: int = 64, spacing: int = 6) -> Image.Image:
    """
    Black background with many small white 2×2 pixel dots.

    Each dot is above the 127 threshold so appears as a separate foreground
    component in connected-component analysis, giving a low mean area score.
    """
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(2, h - 2, spacing):
        for x in range(2, w - 2, spacing):
            arr[y:y + 2, x:x + 2] = 200  # 4-pixel white blob
    return Image.fromarray(arr)


# ─────────────────────────────────────────────────────────────────────────────
# Default weights
# ─────────────────────────────────────────────────────────────────────────────

class TestDefaultWeights:
    def test_weights_sum_to_one(self):
        s = StencilScorer()
        total = (
            s.weight_clarity
            + s.weight_component
            + s.weight_contrast
            + s.weight_region_size
        )
        assert abs(total - 1.0) < 1e-9

    def test_clarity_weight_is_35_percent(self):
        assert StencilScorer().weight_clarity == pytest.approx(0.35)

    def test_component_weight_is_25_percent(self):
        assert StencilScorer().weight_component == pytest.approx(0.25)

    def test_contrast_weight_is_25_percent(self):
        assert StencilScorer().weight_contrast == pytest.approx(0.25)

    def test_region_size_weight_is_15_percent(self):
        assert StencilScorer().weight_region_size == pytest.approx(0.15)


# ─────────────────────────────────────────────────────────────────────────────
# _region_size_score
# ─────────────────────────────────────────────────────────────────────────────

class TestRegionSizeScore:
    def test_returns_float_in_range(self):
        s = StencilScorer()
        for img in [_solid(), _checkerboard(), _two_region(), _many_tiny_dots()]:
            arr = np.array(img.convert("RGB"), dtype=np.uint8)
            score = s._region_size_score(arr)
            assert 0.0 <= score <= 1.0, f"Out-of-range score {score}"

    def test_solid_white_returns_one(self):
        """All-white image has one giant component → max region size score."""
        s = StencilScorer()
        arr = np.full((64, 64, 3), 255, dtype=np.uint8)
        score = s._region_size_score(arr)
        assert score == pytest.approx(1.0)

    def test_many_tiny_dots_scores_lower_than_two_region(self):
        """An image with many tiny dots should score lower than a clean 2-region image."""
        s = StencilScorer()
        arr_dots = np.array(_many_tiny_dots().convert("RGB"), dtype=np.uint8)
        arr_two = np.array(_two_region().convert("RGB"), dtype=np.uint8)
        score_dots = s._region_size_score(arr_dots)
        score_two = s._region_size_score(arr_two)
        assert score_dots < score_two, (
            f"Dot image ({score_dots:.3f}) should score lower than two-region ({score_two:.3f})"
        )

    def test_custom_mean_area_threshold(self):
        """Lower threshold → same image scores higher."""
        s_loose = StencilScorer(mean_area_threshold=0.001)
        s_tight = StencilScorer(mean_area_threshold=0.5)
        arr = np.array(_two_region(), dtype=np.uint8)
        assert s_loose._region_size_score(arr) >= s_tight._region_size_score(arr)

    def test_all_black_image(self):
        """All-black image → background only (label 0) → score = 1.0."""
        s = StencilScorer()
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        score = s._region_size_score(arr)
        assert score == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# score() includes region_size
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreIncludesRegionSize:
    def test_all_weight_on_region_size_equals_region_score(self):
        """weight_region_size=1 and others=0 → score == _region_size_score."""
        s = StencilScorer(
            weight_clarity=0.0,
            weight_component=0.0,
            weight_contrast=0.0,
            weight_region_size=1.0,
        )
        img = _two_region()
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        assert abs(s.score(img) - s._region_size_score(arr)) < 1e-6

    def test_score_in_range_with_region_weight(self):
        for img in [_solid(), _checkerboard(), _many_tiny_dots()]:
            s = StencilScorer()
            score = s.score(img)
            assert 0.0 <= score <= 1.0

    def test_clean_image_higher_than_dotty(self):
        """Two-region image should score at least as well as one full of dots."""
        s = StencilScorer()
        score_two = s.score(_two_region())
        score_dots = s.score(_many_tiny_dots())
        # With region_size penalty, the dotty image should score lower
        assert score_two >= score_dots


# ─────────────────────────────────────────────────────────────────────────────
# FactoryConfig.quality_threshold
# ─────────────────────────────────────────────────────────────────────────────

class TestFactoryConfigQualityThreshold:
    def test_default_quality_threshold(self):
        cfg = FactoryConfig()
        assert cfg.quality_threshold == pytest.approx(0.35)

    def test_custom_quality_threshold(self):
        cfg = FactoryConfig(quality_threshold=0.5)
        assert cfg.quality_threshold == pytest.approx(0.5)

    def test_zero_threshold_accepts_all(self):
        cfg = FactoryConfig(quality_threshold=0.0)
        assert cfg.quality_threshold == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# FactoryRunner quality threshold filtering (mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestFactoryRunnerThresholdFiltering:
    """
    Verify that FactoryRunner discards candidates below quality_threshold.

    The SD pipeline is fully mocked so no model weights are needed.
    """

    def _make_pipeline_config(self, threshold: float = 0.35) -> PipelineConfig:
        cfg = PipelineConfig()
        cfg.factory.count = 3
        cfg.factory.top_k = 3
        cfg.factory.quality_threshold = threshold
        cfg.factory.workers = 1
        cfg.skip_generation = True
        return cfg

    def test_candidates_below_threshold_discarded(self):
        """Candidates whose score < threshold must not appear in ranked output."""
        from pokemon_stencil.factory.factory_runner import CandidateResult, FactoryRunner

        cfg = self._make_pipeline_config(threshold=0.5)
        runner = FactoryRunner(cfg)

        # Inject pre-built candidates with known scores
        low_score = CandidateResult(index=0, score=0.1, simplified_path=None)
        high_score = CandidateResult(index=1, score=0.8, simplified_path=None)

        # Monkeypatch _run_sequential to return our controlled candidates
        runner._run_sequential = MagicMock(return_value=[low_score, high_score])

        # Also skip the stencil conversion stage
        runner._run_stencil_on_candidate = MagicMock(
            side_effect=RuntimeError("stencil skipped")
        )

        result = runner.run(pokemon_name="Pikachu")

        # Only the high-score candidate (0.8 >= 0.5) should be in ranked
        # (The stencil stage raises so pipeline_results is empty,
        #  but we check that errors come from only 1 candidate being tried.)
        errors_from_stencil = [
            e for e in result.errors if "stencil skipped" in e
        ]
        # Exactly 1 stencil attempt (the high-score one)
        assert len(errors_from_stencil) == 1

    def test_all_below_threshold_yields_no_stencil_conversions(self):
        """When all candidates are below threshold, nothing is converted."""
        from pokemon_stencil.factory.factory_runner import CandidateResult, FactoryRunner

        cfg = self._make_pipeline_config(threshold=0.99)
        runner = FactoryRunner(cfg)

        low_candidates = [
            CandidateResult(index=i, score=0.1 * (i + 1)) for i in range(3)
        ]
        runner._run_sequential = MagicMock(return_value=low_candidates)
        runner._run_stencil_on_candidate = MagicMock()

        result = runner.run(pokemon_name="Mewtwo")
        assert runner._run_stencil_on_candidate.call_count == 0
        assert result.succeeded == 0

    def test_zero_threshold_accepts_all_valid(self):
        """threshold=0.0 should let all error-free candidates through."""
        from pokemon_stencil.factory.factory_runner import CandidateResult, FactoryRunner

        cfg = self._make_pipeline_config(threshold=0.0)
        runner = FactoryRunner(cfg)

        candidates = [
            CandidateResult(index=i, score=0.01 * (i + 1)) for i in range(3)
        ]
        runner._run_sequential = MagicMock(return_value=candidates)
        runner._run_stencil_on_candidate = MagicMock(
            side_effect=RuntimeError("stencil skipped in test")
        )

        result = runner.run(pokemon_name="Bulbasaur")
        # All 3 candidates passed threshold → 3 stencil attempts
        assert runner._run_stencil_on_candidate.call_count == 3
