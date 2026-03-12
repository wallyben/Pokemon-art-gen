"""
Tests for pokemon_stencil.factory.scoring.StencilScorer.

All tests use synthetic PIL Images generated in-memory; no external files,
internet access, or GPU required.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pokemon_stencil.factory.scoring import StencilScorer


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures and helpers
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def scorer() -> StencilScorer:
    """Default StencilScorer with default weights."""
    return StencilScorer()


def _solid(value: int = 128, h: int = 64, w: int = 64) -> Image.Image:
    """Solid-colour image – featureless, no edges."""
    arr = np.full((h, w, 3), value, dtype=np.uint8)
    return Image.fromarray(arr)


def _checkerboard(h: int = 64, w: int = 64, tile: int = 8) -> Image.Image:
    """High-contrast checkerboard – many edges, good contrast."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        for x in range(w):
            if ((y // tile) + (x // tile)) % 2 == 0:
                arr[y, x] = 255
    return Image.fromarray(arr)


def _gradient(h: int = 64, w: int = 64) -> Image.Image:
    """Smooth horizontal gradient – mid contrast, few edges."""
    row = np.linspace(0, 255, w, dtype=np.uint8)
    arr = np.tile(row, (h, 1))
    arr_rgb = np.stack([arr, arr, arr], axis=2)
    return Image.fromarray(arr_rgb)


def _noisy(h: int = 64, w: int = 64, seed: int = 0) -> Image.Image:
    """Random-noise image – maximum edge density."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def _two_region(h: int = 64, w: int = 64) -> Image.Image:
    """Image split into two clean halves – moderate edges, good contrast."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, w // 2 :] = 200
    return Image.fromarray(arr)


# ─────────────────────────────────────────────────────────────────────────────
# StencilScorer construction
# ─────────────────────────────────────────────────────────────────────────────

class TestStencilScorerInit:
    def test_default_weights_sum_to_one(self):
        s = StencilScorer()
        total = s.weight_clarity + s.weight_component + s.weight_contrast
        assert abs(total - 1.0) < 1e-9

    def test_custom_weights_stored(self):
        s = StencilScorer(weight_clarity=0.5, weight_component=0.3, weight_contrast=0.2)
        assert s.weight_clarity == 0.5
        assert s.weight_component == 0.3
        assert s.weight_contrast == 0.2

    def test_target_edge_density_stored(self):
        s = StencilScorer(target_edge_density=0.15)
        assert s.target_edge_density == 0.15

    def test_max_small_components_stored(self):
        s = StencilScorer(max_small_components=10)
        assert s.max_small_components == 10

    def test_contrast_normaliser_stored(self):
        s = StencilScorer(contrast_normaliser=32.0)
        assert s.contrast_normaliser == 32.0


# ─────────────────────────────────────────────────────────────────────────────
# score() – public API
# ─────────────────────────────────────────────────────────────────────────────

class TestScore:
    def test_returns_float(self, scorer):
        assert isinstance(scorer.score(_solid()), float)

    def test_score_in_range(self, scorer):
        for img in [_solid(), _checkerboard(), _gradient(), _noisy()]:
            s = scorer.score(img)
            assert 0.0 <= s <= 1.0, f"Score {s} out of [0, 1]"

    def test_accepts_rgba_image(self, scorer):
        arr = np.full((32, 32, 4), 128, dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGBA")
        s = scorer.score(img)
        assert 0.0 <= s <= 1.0

    def test_accepts_grayscale_image(self, scorer):
        arr = np.full((32, 32), 128, dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        s = scorer.score(img)
        assert 0.0 <= s <= 1.0

    def test_solid_grey_scores_low_clarity(self, scorer):
        """A featureless solid image has no edges → low clarity score."""
        s = scorer.score(_solid(128))
        # A solid image has near-zero edge density, which deviates from target
        assert s < 1.0

    def test_two_region_scores_higher_than_solid(self, scorer):
        """A two-region image should score better than a solid grey image."""
        s_two = scorer.score(_two_region())
        s_solid = scorer.score(_solid(128))
        assert s_two >= s_solid

    def test_deterministic(self, scorer):
        """Scoring the same image twice must yield the same result."""
        img = _checkerboard()
        assert scorer.score(img) == scorer.score(img)

    def test_small_image_no_crash(self, scorer):
        """Very small images should not cause errors."""
        arr = np.full((4, 4, 3), 100, dtype=np.uint8)
        img = Image.fromarray(arr)
        s = scorer.score(img)
        assert 0.0 <= s <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# score_all()
# ─────────────────────────────────────────────────────────────────────────────

class TestScoreAll:
    def test_returns_list(self, scorer):
        result = scorer.score_all([_solid(), _checkerboard()])
        assert isinstance(result, list)

    def test_length_matches_input(self, scorer):
        images = [_solid(), _gradient(), _checkerboard(), _noisy()]
        result = scorer.score_all(images)
        assert len(result) == 4

    def test_empty_list_returns_empty(self, scorer):
        assert scorer.score_all([]) == []

    def test_all_scores_in_range(self, scorer):
        images = [_solid(), _gradient(), _checkerboard(), _noisy()]
        for s in scorer.score_all(images):
            assert 0.0 <= s <= 1.0

    def test_scores_match_individual(self, scorer):
        """score_all results must match individual score() calls."""
        images = [_solid(), _checkerboard(), _gradient()]
        batch = scorer.score_all(images)
        individual = [scorer.score(img) for img in images]
        for a, b in zip(batch, individual):
            assert abs(a - b) < 1e-9

    def test_single_image_list(self, scorer):
        result = scorer.score_all([_solid()])
        assert len(result) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Silhouette clarity metric
# ─────────────────────────────────────────────────────────────────────────────

class TestSilhouetteClarity:
    def test_solid_low_clarity(self, scorer):
        """Solid image → zero edges → below-target density → low score."""
        arr = np.full((64, 64, 3), 128, dtype=np.uint8)
        s = scorer._silhouette_clarity(arr)
        assert s < 0.5  # density ≈ 0, far from target 0.10

    def test_noisy_low_clarity(self, scorer):
        """Noisy image → excessive edges → above-target density → low score."""
        rng = np.random.default_rng(42)
        arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
        s = scorer._silhouette_clarity(arr)
        # Very high edge density penalises score
        assert s < 1.0

    def test_returns_float_in_range(self, scorer):
        arr = np.full((32, 32, 3), 200, dtype=np.uint8)
        s = scorer._silhouette_clarity(arr)
        assert 0.0 <= s <= 1.0

    def test_custom_target_changes_score(self):
        """A scorer whose target matches the image's density scores higher."""
        # Checkerboard has a high edge density (~0.22); target=0.30 is nearby,
        # target=0.01 is far away → nearby target should yield a higher score.
        img = _checkerboard()
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        s_near = StencilScorer(target_edge_density=0.30)
        s_far = StencilScorer(target_edge_density=0.01)
        assert s_near._silhouette_clarity(arr) > s_far._silhouette_clarity(arr)


# ─────────────────────────────────────────────────────────────────────────────
# Connected-component metric
# ─────────────────────────────────────────────────────────────────────────────

class TestComponentScore:
    def test_solid_white_no_small_components(self, scorer):
        """All-white image → one large component → score = 1.0."""
        arr = np.full((64, 64, 3), 255, dtype=np.uint8)
        s = scorer._component_score(arr)
        assert s == 1.0

    def test_solid_black_no_small_components(self, scorer):
        """All-black image → background only → score = 1.0."""
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        s = scorer._component_score(arr)
        assert s == 1.0

    def test_many_tiny_dots_low_score(self, scorer):
        """Image filled with isolated 1-pixel dots → many small components."""
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        # Place single-pixel dots on a grid
        for y in range(2, 62, 4):
            for x in range(2, 62, 4):
                arr[y, x] = 255
        s = scorer._component_score(arr)
        assert s < 1.0

    def test_returns_float_in_range(self, scorer):
        arr = np.full((32, 32, 3), 200, dtype=np.uint8)
        s = scorer._component_score(arr)
        assert 0.0 <= s <= 1.0

    def test_custom_max_small_changes_threshold(self):
        """Lower max_small_components → penalty kicks in sooner."""
        s_strict = StencilScorer(max_small_components=1)
        s_lenient = StencilScorer(max_small_components=100)
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        # A few isolated pixels
        for y in range(4, 60, 8):
            arr[y, 4] = 255
        assert s_strict._component_score(arr) <= s_lenient._component_score(arr)


# ─────────────────────────────────────────────────────────────────────────────
# Contrast metric
# ─────────────────────────────────────────────────────────────────────────────

class TestContrastScore:
    def test_solid_zero_contrast(self, scorer):
        """Uniform image has std=0 → contrast score = 0."""
        arr = np.full((64, 64, 3), 128, dtype=np.uint8)
        s = scorer._contrast_score(arr)
        assert s == 0.0

    def test_high_contrast_near_one(self, scorer):
        """Checkerboard has high std → contrast score close to 1."""
        arr = np.array(_checkerboard())
        s = scorer._contrast_score(arr)
        assert s > 0.5

    def test_returns_capped_at_one(self, scorer):
        """Score must not exceed 1.0 even for extreme contrast."""
        arr = np.array(_checkerboard())
        s = scorer._contrast_score(arr)
        assert s <= 1.0

    def test_gradient_intermediate_contrast(self, scorer):
        """Gradient has non-zero contrast → score is positive."""
        arr = np.array(_gradient())
        s = scorer._contrast_score(arr)
        assert 0.0 < s <= 1.0

    def test_custom_normaliser(self):
        """Lower normaliser → same image scores higher."""
        s_loose = StencilScorer(contrast_normaliser=10.0)
        s_tight = StencilScorer(contrast_normaliser=200.0)
        arr = np.array(_gradient())
        assert s_loose._contrast_score(arr) >= s_tight._contrast_score(arr)

    def test_returns_float_in_range(self, scorer):
        arr = np.array(_gradient())
        s = scorer._contrast_score(arr)
        assert 0.0 <= s <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Weighted combination
# ─────────────────────────────────────────────────────────────────────────────

class TestWeightedCombination:
    def test_all_weight_on_contrast_equals_contrast_score(self):
        """With weight_contrast=1 and others=0, score must equal contrast."""
        scorer = StencilScorer(
            weight_clarity=0.0, weight_component=0.0, weight_contrast=1.0
        )
        img = _gradient()
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        assert abs(scorer.score(img) - scorer._contrast_score(arr)) < 1e-6

    def test_all_weight_on_clarity_equals_clarity_score(self):
        scorer = StencilScorer(
            weight_clarity=1.0, weight_component=0.0, weight_contrast=0.0
        )
        img = _two_region()
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        assert abs(scorer.score(img) - scorer._silhouette_clarity(arr)) < 1e-6

    def test_all_weight_on_component_equals_component_score(self):
        scorer = StencilScorer(
            weight_clarity=0.0, weight_component=1.0, weight_contrast=0.0
        )
        img = _solid()
        arr = np.array(img.convert("RGB"), dtype=np.uint8)
        assert abs(scorer.score(img) - scorer._component_score(arr)) < 1e-6
