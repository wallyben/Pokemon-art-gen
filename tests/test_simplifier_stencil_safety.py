"""
Tests for the stencil safety improvements added to ImageSimplifier:
- Adaptive edge thinning
- Small component merging
- min_island_area threshold

These tests verify the new behaviour added in the v2 upgrade.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pokemon_stencil.config import ProcessingConfig
from pokemon_stencil.image_proc.simplifier import ImageSimplifier, _thin_binary_mask


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def base_cfg() -> ProcessingConfig:
    """Config with stencil safety features enabled (new defaults)."""
    return ProcessingConfig(
        output_size=(64, 64),
        bilateral_d=5,
        bilateral_sigma_color=40.0,
        bilateral_sigma_space=40.0,
        bilateral_passes=1,
        n_colors=4,
        morph_kernel_size=3,
        canny_low=50,
        canny_high=150,
        adaptive_edge_thinning=True,
        merge_small_components=True,
        min_island_area=50,
    )


@pytest.fixture()
def no_safety_cfg() -> ProcessingConfig:
    """Config with stencil safety features disabled."""
    return ProcessingConfig(
        output_size=(64, 64),
        bilateral_d=5,
        bilateral_sigma_color=40.0,
        bilateral_sigma_space=40.0,
        bilateral_passes=1,
        n_colors=4,
        morph_kernel_size=3,
        adaptive_edge_thinning=False,
        merge_small_components=False,
        min_island_area=50,
    )


def _gradient_image(size=(64, 64)) -> Image.Image:
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for x in range(size[0]):
        v = int(x / size[0] * 255)
        arr[:, x] = [v, 255 - v, 128]
    return Image.fromarray(arr)


def _two_region_image(size=(64, 64)) -> Image.Image:
    """Image with a bright left half and dark right half (clean two-region)."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    mid = size[0] // 2
    arr[:, :mid] = [200, 200, 200]
    arr[:, mid:] = [30, 30, 30]
    return Image.fromarray(arr)


# ─────────────────────────────────────────────────────────────────────────────
# _thin_binary_mask
# ─────────────────────────────────────────────────────────────────────────────

class TestThinBinaryMask:
    def test_returns_same_shape(self):
        import cv2
        mask = np.zeros((32, 32), dtype=np.uint8)
        cv2.circle(mask, (16, 16), 10, 255, -1)
        result = _thin_binary_mask(mask)
        assert result.shape == mask.shape

    def test_thinned_mask_is_smaller_or_equal(self):
        import cv2
        mask = np.zeros((32, 32), dtype=np.uint8)
        cv2.circle(mask, (16, 16), 10, 255, -1)
        result = _thin_binary_mask(mask)
        # Thinned mask should have fewer or equal white pixels.
        assert result.sum() <= mask.sum()

    def test_empty_mask_stays_empty(self):
        mask = np.zeros((32, 32), dtype=np.uint8)
        result = _thin_binary_mask(mask)
        assert result.sum() == 0

    def test_full_mask_reduces(self):
        import cv2
        # A 64x64 all-white mask erodes from borders with a cross kernel.
        # Use max_iter=10 to allow enough erosion iterations.
        mask = np.full((64, 64), 255, dtype=np.uint8)
        # Erode once manually to verify the function has any effect.
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        manually_eroded = cv2.erode(mask, kernel, iterations=1)
        result = _thin_binary_mask(mask, max_iter=10)
        # Result should be <= manually eroded (same or more reduced).
        assert result.sum() <= manually_eroded.sum()


# ─────────────────────────────────────────────────────────────────────────────
# ImageSimplifier with stencil safety enabled
# ─────────────────────────────────────────────────────────────────────────────

class TestSimplifierStencilSafety:
    def test_simplify_returns_pil_image(self, base_cfg):
        simplifier = ImageSimplifier(base_cfg)
        result = simplifier.simplify(_gradient_image())
        assert isinstance(result, Image.Image)

    def test_output_mode_is_rgb(self, base_cfg):
        simplifier = ImageSimplifier(base_cfg)
        result = simplifier.simplify(_gradient_image())
        assert result.mode == "RGB"

    def test_output_size_matches_input(self, base_cfg):
        simplifier = ImageSimplifier(base_cfg)
        img = _gradient_image(size=(64, 64))
        result = simplifier.simplify(img)
        assert result.size == img.size

    def test_two_region_image_runs_without_error(self, base_cfg):
        simplifier = ImageSimplifier(base_cfg)
        result = simplifier.simplify(_two_region_image())
        assert result is not None

    def test_edge_thinning_disabled_still_works(self, no_safety_cfg):
        simplifier = ImageSimplifier(no_safety_cfg)
        result = simplifier.simplify(_gradient_image())
        assert isinstance(result, Image.Image)

    def test_component_merging_disabled_still_works(self, no_safety_cfg):
        simplifier = ImageSimplifier(no_safety_cfg)
        result = simplifier.simplify(_gradient_image())
        assert isinstance(result, Image.Image)


# ─────────────────────────────────────────────────────────────────────────────
# ProcessingConfig new fields
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessingConfigNewFields:
    def test_defaults_are_set(self):
        cfg = ProcessingConfig()
        assert cfg.adaptive_edge_thinning is True
        assert cfg.merge_small_components is True
        assert isinstance(cfg.min_island_area, int)
        assert cfg.min_island_area > 0

    def test_can_disable_adaptive_edge_thinning(self):
        cfg = ProcessingConfig(adaptive_edge_thinning=False)
        assert cfg.adaptive_edge_thinning is False

    def test_can_disable_merge_small_components(self):
        cfg = ProcessingConfig(merge_small_components=False)
        assert cfg.merge_small_components is False

    def test_custom_min_island_area(self):
        cfg = ProcessingConfig(min_island_area=1000)
        assert cfg.min_island_area == 1000


# ─────────────────────────────────────────────────────────────────────────────
# _merge_small_components (internal method via simplify)
# ─────────────────────────────────────────────────────────────────────────────

class TestMergeSmallComponents:
    def test_small_island_merged_into_background(self):
        """Test that _merge_small_components absorbs small bright islands.

        Otsu thresholding treats bright pixels as foreground.  A small bright
        dot on a dark background is a small foreground component; after
        merging it should be repainted with the surrounding dark colour.
        """
        simplifier = ImageSimplifier(ProcessingConfig())

        # Black background with a small bright dot (4×4 = 16 px).
        # Otsu will make the bright dot a small foreground component.
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[28:32, 28:32] = [200, 200, 200]  # 16-px bright dot

        # min_area=100 exceeds the dot size (16 px) so it should be merged.
        result = simplifier._merge_small_components(arr, min_area=100)

        dot_region = result[28:32, 28:32]
        # The bright dot should have been repainted with the surrounding dark.
        assert dot_region.mean() < 100
