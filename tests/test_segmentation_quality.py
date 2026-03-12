"""
Tests for segmentation quality improvements in ColourSegmenter.

Verifies:
- Morphological opening and closing use separate kernels from config.
- Connected-component filtering removes regions smaller than min_region_area.
- Output masks are uint8 binary arrays (values 0 or 255 only).
- The _remove_small_components helper works correctly.

All tests use synthetic images; no GPU or SD inference required.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pokemon_stencil.config import ProcessingConfig
from pokemon_stencil.image_proc.segmenter import ColourLayer, ColourSegmenter


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_config(**kwargs) -> ProcessingConfig:
    """Build a ProcessingConfig with sensible test defaults."""
    defaults = dict(
        n_colors=2,
        min_region_area=200,
        morph_open_kernel=3,
        morph_close_kernel=3,
        morph_kernel_size=3,
        bilateral_d=1,
        bilateral_passes=1,
        output_size=(64, 64),
    )
    defaults.update(kwargs)
    return ProcessingConfig(**defaults)


def _two_colour_image(h: int = 64, w: int = 64) -> Image.Image:
    """Image split cleanly into two half-size colour blocks."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, : w // 2] = [50, 50, 200]   # blue half
    arr[:, w // 2 :] = [200, 50, 50]   # red half
    return Image.fromarray(arr)


def _tiny_dot_image(h: int = 64, w: int = 64) -> Image.Image:
    """
    Mostly white image with a single tiny 2×2 dark pixel cluster and a large
    dark region.  The tiny cluster should be removed by the size filter.
    """
    arr = np.full((h, w, 3), 240, dtype=np.uint8)
    # Large dark block (fills left quarter)
    arr[:, : w // 4] = 20
    # Tiny 2×2 dark dot in the centre-right
    arr[30:32, 50:52] = 20
    return Image.fromarray(arr)


# ─────────────────────────────────────────────────────────────────────────────
# ProcessingConfig new fields
# ─────────────────────────────────────────────────────────────────────────────

class TestProcessingConfigNewFields:
    def test_morph_open_kernel_default(self):
        cfg = ProcessingConfig()
        assert cfg.morph_open_kernel == 3

    def test_morph_close_kernel_default(self):
        cfg = ProcessingConfig()
        assert cfg.morph_close_kernel == 3

    def test_min_region_area_default_positive(self):
        cfg = ProcessingConfig()
        assert cfg.min_region_area > 0

    def test_custom_morph_open_kernel(self):
        cfg = ProcessingConfig(morph_open_kernel=5)
        assert cfg.morph_open_kernel == 5

    def test_custom_morph_close_kernel(self):
        cfg = ProcessingConfig(morph_close_kernel=7)
        assert cfg.morph_close_kernel == 7


# ─────────────────────────────────────────────────────────────────────────────
# _remove_small_components
# ─────────────────────────────────────────────────────────────────────────────

class TestRemoveSmallComponents:
    def test_removes_tiny_blobs(self):
        """A 3×3 isolated blob should be removed when min_region_area=200."""
        cfg = _make_config(min_region_area=200)
        seg = ColourSegmenter(cfg)

        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[30:33, 30:33] = 255  # 9-pixel blob → below threshold

        result = seg._remove_small_components(mask)
        assert np.all(result == 0), "Tiny blob should be removed."

    def test_preserves_large_regions(self):
        """A 20×20 block (400 px) must survive a min_region_area=200 filter."""
        cfg = _make_config(min_region_area=200)
        seg = ColourSegmenter(cfg)

        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:30, 10:30] = 255  # 400-pixel block

        result = seg._remove_small_components(mask)
        assert np.any(result == 255), "Large region must be preserved."

    def test_output_dtype_is_uint8(self):
        cfg = _make_config(min_region_area=10)
        seg = ColourSegmenter(cfg)

        mask = np.zeros((32, 32), dtype=np.uint8)
        mask[5:15, 5:15] = 255

        result = seg._remove_small_components(mask)
        assert result.dtype == np.uint8

    def test_empty_mask_returns_zeros(self):
        cfg = _make_config(min_region_area=1)
        seg = ColourSegmenter(cfg)

        mask = np.zeros((32, 32), dtype=np.uint8)
        result = seg._remove_small_components(mask)
        assert np.all(result == 0)

    def test_large_only_mask_preserved(self):
        """A full-image mask should always survive regardless of threshold."""
        cfg = _make_config(min_region_area=100)
        seg = ColourSegmenter(cfg)

        mask = np.full((64, 64), 255, dtype=np.uint8)
        result = seg._remove_small_components(mask)
        assert np.any(result == 255)


# ─────────────────────────────────────────────────────────────────────────────
# _clean_mask
# ─────────────────────────────────────────────────────────────────────────────

class TestCleanMask:
    def test_output_only_zero_or_255(self):
        """Cleaned mask must be binary (0 or 255 only)."""
        cfg = _make_config(min_region_area=10)
        seg = ColourSegmenter(cfg)

        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        result = seg._clean_mask(mask)

        unique = set(np.unique(result).tolist())
        assert unique.issubset({0, 255}), f"Non-binary values found: {unique}"

    def test_output_dtype_is_uint8(self):
        cfg = _make_config(min_region_area=10)
        seg = ColourSegmenter(cfg)

        mask = np.full((32, 32), 255, dtype=np.uint8)
        result = seg._clean_mask(mask)
        assert result.dtype == np.uint8

    def test_tiny_component_removed_after_clean(self):
        """A small isolated cluster should be gone after _clean_mask."""
        cfg = _make_config(min_region_area=200, morph_open_kernel=3, morph_close_kernel=3)
        seg = ColourSegmenter(cfg)

        mask = np.zeros((64, 64), dtype=np.uint8)
        # Tiny 4×4 blob only
        mask[10:14, 10:14] = 255  # 16 px < 200 threshold

        result = seg._clean_mask(mask)
        assert np.all(result == 0), "Sub-threshold component must be removed."

    def test_large_region_survives(self):
        cfg = _make_config(min_region_area=50, morph_open_kernel=3, morph_close_kernel=3)
        seg = ColourSegmenter(cfg)

        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[5:30, 5:30] = 255  # 625 px > 50 threshold

        result = seg._clean_mask(mask)
        assert np.any(result == 255)


# ─────────────────────────────────────────────────────────────────────────────
# segment() – end-to-end
# ─────────────────────────────────────────────────────────────────────────────

class TestSegmentEndToEnd:
    def test_returns_list_of_colour_layers(self):
        cfg = _make_config(n_colors=2)
        seg = ColourSegmenter(cfg)
        layers = seg.segment(_two_colour_image())
        assert isinstance(layers, list)
        assert all(isinstance(l, ColourLayer) for l in layers)

    def test_masks_are_binary_uint8(self):
        cfg = _make_config(n_colors=2, min_region_area=10)
        seg = ColourSegmenter(cfg)
        layers = seg.segment(_two_colour_image())
        for layer in layers:
            assert layer.mask.dtype == np.uint8
            unique = set(np.unique(layer.mask).tolist())
            assert unique.issubset({0, 255}), (
                f"Layer {layer.index} mask has non-binary values: {unique}"
            )

    def test_layers_sorted_by_pixel_count_descending(self):
        cfg = _make_config(n_colors=2, min_region_area=10)
        seg = ColourSegmenter(cfg)
        layers = seg.segment(_two_colour_image())
        counts = [l.pixel_count for l in layers]
        assert counts == sorted(counts, reverse=True)

    def test_morph_open_kernel_respected(self):
        """Segmenter constructed with different kernels should not crash."""
        cfg_small = _make_config(morph_open_kernel=1, morph_close_kernel=1, min_region_area=10)
        cfg_large = _make_config(morph_open_kernel=5, morph_close_kernel=5, min_region_area=10)
        img = _two_colour_image()
        seg_small = ColourSegmenter(cfg_small)
        seg_large = ColourSegmenter(cfg_large)
        # Both should run without error
        seg_small.segment(img)
        seg_large.segment(img)
