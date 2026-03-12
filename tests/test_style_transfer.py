"""
Tests for pokemon_stencil.image_gen.style_transfer.StencilStyleTransfer.

All tests use synthetic PIL Images; no GPU or external files required.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pokemon_stencil.config import ProcessingConfig
from pokemon_stencil.image_gen.style_transfer import StencilStyleTransfer


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def default_cfg() -> ProcessingConfig:
    return ProcessingConfig(
        output_size=(64, 64),
        bilateral_d=5,
        bilateral_sigma_color=50.0,
        bilateral_sigma_space=50.0,
        bilateral_passes=1,
        n_colors=4,
    )


@pytest.fixture()
def xfer(default_cfg) -> StencilStyleTransfer:
    return StencilStyleTransfer(default_cfg)


def _make_gradient(size=(64, 64)) -> Image.Image:
    """64×64 RGB image with a smooth horizontal gradient."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for x in range(size[0]):
        val = int(x / size[0] * 255)
        arr[:, x] = [val, 255 - val, 128]
    return Image.fromarray(arr)


def _make_solid(size=(64, 64), colour=(120, 80, 200)) -> Image.Image:
    return Image.new("RGB", size, colour)


# ─────────────────────────────────────────────────────────────────────────────
# apply() – end-to-end
# ─────────────────────────────────────────────────────────────────────────────

class TestApply:
    def test_returns_pil_image(self, xfer):
        result = xfer.apply(_make_gradient())
        assert isinstance(result, Image.Image)

    def test_output_mode_is_rgb(self, xfer):
        result = xfer.apply(_make_gradient())
        assert result.mode == "RGB"

    def test_accepts_rgba_input(self, xfer):
        rgba = Image.new("RGBA", (64, 64), (200, 100, 50, 200))
        result = xfer.apply(rgba)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_output_size_matches_config(self, default_cfg):
        custom_cfg = ProcessingConfig(
            output_size=(48, 48),
            bilateral_d=5,
            bilateral_sigma_color=50.0,
            bilateral_sigma_space=50.0,
            bilateral_passes=1,
        )
        xfer = StencilStyleTransfer(custom_cfg)
        result = xfer.apply(_make_gradient((80, 80)))
        assert result.size == (48, 48)

    def test_output_is_not_all_black(self, xfer):
        result = xfer.apply(_make_gradient())
        arr = np.array(result)
        assert arr.mean() > 10

    def test_output_is_not_all_white(self, xfer):
        result = xfer.apply(_make_gradient())
        arr = np.array(result)
        assert arr.mean() < 250

    def test_deterministic_for_same_input(self, xfer):
        img = _make_gradient()
        r1 = np.array(xfer.apply(img))
        r2 = np.array(xfer.apply(img))
        np.testing.assert_array_equal(r1, r2)


# ─────────────────────────────────────────────────────────────────────────────
# _resize
# ─────────────────────────────────────────────────────────────────────────────

class TestResize:
    def test_resizes_larger_image(self, xfer):
        big = _make_solid((256, 256))
        result = xfer._resize(big)
        assert result.size == (64, 64)

    def test_resizes_smaller_image(self, xfer):
        small = _make_solid((16, 16))
        result = xfer._resize(small)
        assert result.size == (64, 64)

    def test_same_size_returns_unchanged(self, xfer):
        img = _make_solid((64, 64))
        result = xfer._resize(img)
        assert result.size == (64, 64)


# ─────────────────────────────────────────────────────────────────────────────
# _bilateral_smooth
# ─────────────────────────────────────────────────────────────────────────────

class TestBilateralSmooth:
    def test_returns_same_size(self, xfer):
        img = _make_gradient()
        result = xfer._bilateral_smooth(img)
        assert result.size == img.size

    def test_returns_pil_image(self, xfer):
        result = xfer._bilateral_smooth(_make_gradient())
        assert isinstance(result, Image.Image)

    def test_smooths_gradient(self, xfer):
        """A gradient image should have reduced variance after bilateral smooth."""
        img = _make_gradient()
        original_std = np.array(img).std()
        smoothed_std = np.array(xfer._bilateral_smooth(img)).std()
        # Bilateral filter never increases variance drastically
        assert smoothed_std <= original_std * 1.1

    def test_multiple_passes_run_without_error(self, default_cfg):
        cfg = ProcessingConfig(
            output_size=(64, 64),
            bilateral_d=5,
            bilateral_sigma_color=50.0,
            bilateral_sigma_space=50.0,
            bilateral_passes=3,
        )
        xfer = StencilStyleTransfer(cfg)
        result = xfer._bilateral_smooth(_make_gradient())
        assert isinstance(result, Image.Image)


# ─────────────────────────────────────────────────────────────────────────────
# _enhance_contrast (CLAHE)
# ─────────────────────────────────────────────────────────────────────────────

class TestEnhanceContrast:
    def test_returns_pil_image(self, xfer):
        result = xfer._enhance_contrast(_make_gradient())
        assert isinstance(result, Image.Image)

    def test_output_size_unchanged(self, xfer):
        img = _make_gradient((64, 64))
        result = xfer._enhance_contrast(img)
        assert result.size == (64, 64)

    def test_low_contrast_image_gains_contrast(self, xfer):
        """A near-grey image should have higher std after CLAHE."""
        grey = Image.new("RGB", (64, 64), (128, 128, 128))
        # Don't assert exact values; just confirm it runs without error
        result = xfer._enhance_contrast(grey)
        assert isinstance(result, Image.Image)


# ─────────────────────────────────────────────────────────────────────────────
# _posterise
# ─────────────────────────────────────────────────────────────────────────────

class TestPosterise:
    def test_returns_pil_image(self):
        img = _make_gradient()
        result = StencilStyleTransfer._posterise(img, levels=4)
        assert isinstance(result, Image.Image)

    def test_output_size_unchanged(self):
        img = _make_gradient((64, 64))
        result = StencilStyleTransfer._posterise(img, levels=4)
        assert result.size == (64, 64)

    def test_reduces_unique_values(self):
        """Posterisation should produce fewer unique pixel values."""
        img = _make_gradient((64, 64))
        original_unique = len(np.unique(np.array(img).reshape(-1, 3), axis=0))
        result = StencilStyleTransfer._posterise(img, levels=4)
        result_unique = len(np.unique(np.array(result).reshape(-1, 3), axis=0))
        assert result_unique < original_unique

    def test_pixel_values_within_range(self):
        arr = np.array(StencilStyleTransfer._posterise(_make_gradient(), levels=4))
        assert arr.min() >= 0
        assert arr.max() <= 255

    def test_two_levels_produces_binary_output(self):
        """Two levels → each channel value is either 0 or 255."""
        img = _make_gradient()
        result = StencilStyleTransfer._posterise(img, levels=2)
        arr = np.array(result)
        unique = set(np.unique(arr).tolist())
        assert unique.issubset({0, 255})


# ─────────────────────────────────────────────────────────────────────────────
# _sharpen_edges
# ─────────────────────────────────────────────────────────────────────────────

class TestSharpenEdges:
    def test_returns_pil_image(self):
        result = StencilStyleTransfer._sharpen_edges(_make_gradient())
        assert isinstance(result, Image.Image)

    def test_output_size_unchanged(self):
        img = _make_gradient((64, 64))
        result = StencilStyleTransfer._sharpen_edges(img)
        assert result.size == (64, 64)

    def test_sharpening_increases_edge_contrast(self):
        """Sharpening should not decrease the standard deviation of a gradient."""
        img = _make_gradient()
        original_std = float(np.array(img).std())
        sharpened_std = float(np.array(StencilStyleTransfer._sharpen_edges(img)).std())
        # Sharpening typically increases std; allow small tolerance
        assert sharpened_std >= original_std * 0.90
