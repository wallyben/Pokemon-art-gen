"""
Tests for pokemon_stencil.factory.diversity.DesignDiversityFilter.

All tests use synthetic PIL Images and temporary files.
No Stable Diffusion execution is required.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from unittest.mock import patch

import numpy as np
import pytest
from PIL import Image

from pokemon_stencil.factory.diversity import (
    DesignDiversityFilter,
    _FP_SIZE,
    _is_diverse,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fake data
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class FakeCandidate:
    """Minimal CandidateResult-compatible object for testing."""
    index: int
    score: float
    simplified_path: Optional[Path] = None
    error: Optional[str] = None


def _uniform_image(size: tuple = (64, 64), colour: tuple = (128, 128, 128)) -> Image.Image:
    return Image.new("RGB", size, colour)


def _checkerboard_image(size: tuple = (128, 128), block: int = 16) -> Image.Image:
    """Image with strong high-frequency edges."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for y in range(size[1]):
        for x in range(size[0]):
            if (x // block + y // block) % 2 == 0:
                arr[y, x, :] = 255
    return Image.fromarray(arr)


def _gradient_image(size: tuple = (128, 128)) -> Image.Image:
    """Image with smooth left-to-right gradient; few Canny edges."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for x in range(size[0]):
        val = int(255 * x / max(size[0] - 1, 1))
        arr[:, x, :] = val
    return Image.fromarray(arr)


def _save_image(img: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprint generation
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeFingerprint:
    def test_returns_numpy_array(self):
        filt = DesignDiversityFilter()
        fp = filt.compute_fingerprint(_uniform_image())
        assert isinstance(fp, np.ndarray)

    def test_correct_length(self):
        filt = DesignDiversityFilter()
        fp = filt.compute_fingerprint(_uniform_image())
        assert fp.shape == (_FP_SIZE * _FP_SIZE,)

    def test_dtype_is_float32(self):
        filt = DesignDiversityFilter()
        fp = filt.compute_fingerprint(_gradient_image())
        assert fp.dtype == np.float32

    def test_deterministic(self):
        filt = DesignDiversityFilter()
        img = _checkerboard_image()
        fp1 = filt.compute_fingerprint(img)
        fp2 = filt.compute_fingerprint(img)
        np.testing.assert_array_equal(fp1, fp2)

    def test_different_images_produce_different_fingerprints(self):
        filt = DesignDiversityFilter()
        fp_check = filt.compute_fingerprint(_checkerboard_image())
        fp_uni = filt.compute_fingerprint(Image.new("RGB", (128, 128), (0, 0, 0)))
        # Checkerboard has many edges; black image has none.
        assert not np.array_equal(fp_check, fp_uni)

    def test_accepts_rgba_input(self):
        filt = DesignDiversityFilter()
        img = Image.new("RGBA", (64, 64), (255, 0, 0, 255))
        fp = filt.compute_fingerprint(img)
        assert fp.shape == (_FP_SIZE * _FP_SIZE,)

    def test_values_in_valid_range(self):
        filt = DesignDiversityFilter()
        fp = filt.compute_fingerprint(_checkerboard_image())
        assert fp.min() >= 0.0
        assert fp.max() <= 255.0


# ─────────────────────────────────────────────────────────────────────────────
# Cosine similarity
# ─────────────────────────────────────────────────────────────────────────────

class TestCosineSimilarity:
    def test_identical_vectors_return_one(self):
        filt = DesignDiversityFilter()
        v = np.ones(100, dtype=np.float32)
        assert filt.cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors_return_zero(self):
        filt = DesignDiversityFilter()
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert filt.cosine_similarity(a, b) == pytest.approx(0.0)

    def test_zero_vector_returns_zero(self):
        """Avoid division by zero for blank fingerprints."""
        filt = DesignDiversityFilter()
        zero = np.zeros(100, dtype=np.float32)
        v = np.ones(100, dtype=np.float32)
        assert filt.cosine_similarity(zero, v) == 0.0
        assert filt.cosine_similarity(v, zero) == 0.0

    def test_both_zero_vectors_return_zero(self):
        filt = DesignDiversityFilter()
        zero = np.zeros(100, dtype=np.float32)
        assert filt.cosine_similarity(zero, zero) == 0.0

    def test_result_in_range(self):
        filt = DesignDiversityFilter()
        rng = np.random.default_rng(42)
        a = rng.random(1024).astype(np.float32)
        b = rng.random(1024).astype(np.float32)
        sim = filt.cosine_similarity(a, b)
        assert 0.0 <= sim <= 1.0

    def test_symmetry(self):
        filt = DesignDiversityFilter()
        rng = np.random.default_rng(0)
        a = rng.random(256).astype(np.float32)
        b = rng.random(256).astype(np.float32)
        assert filt.cosine_similarity(a, b) == pytest.approx(
            filt.cosine_similarity(b, a)
        )


# ─────────────────────────────────────────────────────────────────────────────
# Diverse candidate selection
# ─────────────────────────────────────────────────────────────────────────────

class TestSelectDiverse:
    def test_selects_up_to_top_k(self, tmp_path):
        """Should never return more than top_k candidates."""
        filt = DesignDiversityFilter(threshold=0.99)  # very permissive

        candidates = []
        for i in range(5):
            img_path = tmp_path / f"cand_{i}.png"
            _save_image(_checkerboard_image(block=4 * (i + 1)), img_path)
            candidates.append(FakeCandidate(index=i, score=float(5 - i), simplified_path=img_path))

        result = filt.select_diverse(candidates, top_k=3)
        assert len(result) <= 3

    def test_returns_highest_scoring_first(self, tmp_path):
        """First returned candidate should have the highest score."""
        filt = DesignDiversityFilter(threshold=0.99)

        imgs = [_checkerboard_image(block=4 * (i + 1)) for i in range(4)]
        candidates = []
        for i, img in enumerate(imgs):
            p = tmp_path / f"c{i}.png"
            _save_image(img, p)
            candidates.append(FakeCandidate(index=i, score=float(i), simplified_path=p))

        result = filt.select_diverse(candidates, top_k=4)
        if len(result) >= 2:
            assert result[0].score >= result[1].score

    def test_identical_images_reduced_to_one(self, tmp_path):
        """Multiple copies of the same image should yield only one selection."""
        filt = DesignDiversityFilter(threshold=0.85)

        img = _checkerboard_image()
        candidates = []
        for i in range(4):
            p = tmp_path / f"same_{i}.png"
            _save_image(img, p)
            candidates.append(
                FakeCandidate(index=i, score=float(4 - i), simplified_path=p)
            )

        result = filt.select_diverse(candidates, top_k=4)
        # All images are identical → only first should pass the filter.
        assert len(result) == 1

    def test_completely_different_images_all_selected(self, tmp_path):
        """Structurally very different images should all pass."""
        filt = DesignDiversityFilter(threshold=0.85)

        # Use completely black vs white images – Canny finds no edges on
        # either so fingerprints are zero vectors → cosine_similarity=0.
        imgs = [
            Image.new("RGB", (128, 128), (0, 0, 0)),
            Image.new("RGB", (128, 128), (255, 255, 255)),
        ]
        candidates = []
        for i, img in enumerate(imgs):
            p = tmp_path / f"diff_{i}.png"
            _save_image(img, p)
            candidates.append(FakeCandidate(index=i, score=float(2 - i), simplified_path=p))

        result = filt.select_diverse(candidates, top_k=2)
        # Both should be accepted (cosine similarity of zero vectors = 0).
        assert len(result) == 2

    def test_missing_simplified_path_skipped(self, tmp_path):
        """Candidates with simplified_path=None are silently skipped."""
        filt = DesignDiversityFilter(threshold=0.99)

        p = tmp_path / "good.png"
        _save_image(_checkerboard_image(), p)

        candidates = [
            FakeCandidate(index=0, score=10.0, simplified_path=None),
            FakeCandidate(index=1, score=9.0, simplified_path=p),
        ]
        result = filt.select_diverse(candidates, top_k=2)
        assert len(result) == 1
        assert result[0].index == 1

    def test_empty_candidates_returns_empty(self):
        filt = DesignDiversityFilter()
        result = filt.select_diverse([], top_k=3)
        assert result == []

    def test_top_k_zero_returns_empty(self, tmp_path):
        filt = DesignDiversityFilter(threshold=0.99)
        p = tmp_path / "img.png"
        _save_image(_checkerboard_image(), p)
        candidates = [FakeCandidate(index=0, score=1.0, simplified_path=p)]
        result = filt.select_diverse(candidates, top_k=0)
        assert result == []

    def test_threshold_boundary_strict(self, tmp_path):
        """similarity >= threshold → reject.

        Use threshold=0.95 to avoid floating-point precision issues near 1.0.
        Two copies of the same checkerboard image have cosine similarity very
        close to 1.0, which reliably exceeds 0.95.
        """
        filt = DesignDiversityFilter(threshold=0.95)
        img = _checkerboard_image()
        candidates = []
        for i in range(2):
            p = tmp_path / f"t_{i}.png"
            _save_image(img, p)
            candidates.append(FakeCandidate(index=i, score=float(2 - i), simplified_path=p))

        result = filt.select_diverse(candidates, top_k=2)
        assert len(result) == 1

    def test_nonexistent_path_skipped(self, tmp_path):
        """A candidate whose file no longer exists is silently skipped."""
        filt = DesignDiversityFilter(threshold=0.99)
        p = tmp_path / "ghost.png"  # file not created
        candidates = [FakeCandidate(index=0, score=1.0, simplified_path=p)]
        result = filt.select_diverse(candidates, top_k=1)
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# Threshold validation
# ─────────────────────────────────────────────────────────────────────────────

class TestThresholdValidation:
    def test_valid_threshold_zero(self):
        filt = DesignDiversityFilter(threshold=0.0)
        assert filt.threshold == 0.0

    def test_valid_threshold_one(self):
        filt = DesignDiversityFilter(threshold=1.0)
        assert filt.threshold == 1.0

    def test_invalid_threshold_negative_raises(self):
        with pytest.raises(ValueError, match="diversity_threshold"):
            DesignDiversityFilter(threshold=-0.1)

    def test_invalid_threshold_above_one_raises(self):
        with pytest.raises(ValueError, match="diversity_threshold"):
            DesignDiversityFilter(threshold=1.1)


# ─────────────────────────────────────────────────────────────────────────────
# _is_diverse helper
# ─────────────────────────────────────────────────────────────────────────────

class TestIsDiverse:
    def test_empty_existing_always_diverse(self):
        fp = np.ones(10, dtype=np.float32)
        assert _is_diverse(fp, [], threshold=0.9) is True

    def test_identical_above_threshold_not_diverse(self):
        fp = np.ones(10, dtype=np.float32)
        assert _is_diverse(fp, [fp], threshold=0.9) is False

    def test_orthogonal_below_threshold_is_diverse(self):
        a = np.array([1.0, 0.0], dtype=np.float32)
        b = np.array([0.0, 1.0], dtype=np.float32)
        assert _is_diverse(a, [b], threshold=0.9) is True


# ─────────────────────────────────────────────────────────────────────────────
# Integration with FactoryConfig diversity_threshold
# ─────────────────────────────────────────────────────────────────────────────

class TestFactoryConfigIntegration:
    def test_factory_config_has_diversity_threshold(self):
        from pokemon_stencil.config import FactoryConfig
        cfg = FactoryConfig()
        assert hasattr(cfg, "diversity_threshold")
        assert cfg.diversity_threshold == pytest.approx(0.85)

    def test_factory_runner_uses_diversity_threshold(self):
        from pokemon_stencil.config import FactoryConfig, PipelineConfig
        from pokemon_stencil.factory.factory_runner import FactoryRunner

        config = PipelineConfig()
        config.factory.diversity_threshold = 0.7
        runner = FactoryRunner(config)
        assert runner._diversity_filter.threshold == pytest.approx(0.7)
