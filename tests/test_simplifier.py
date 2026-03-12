"""
Tests for pokemon_stencil.image_proc.simplifier.ImageSimplifier.

All tests use synthetic PIL Images; no external files or GPU required.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pokemon_stencil.config import ProcessingConfig
from pokemon_stencil.image_proc.simplifier import ImageSimplifier


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def fast_cfg() -> ProcessingConfig:
    """ProcessingConfig tuned for fast test execution."""
    return ProcessingConfig(
        output_size=(64, 64),
        bilateral_d=5,
        bilateral_sigma_color=50.0,
        bilateral_sigma_space=50.0,
        bilateral_passes=1,
        n_colors=4,
        morph_kernel_size=3,
        canny_low=50,
        canny_high=150,
    )


@pytest.fixture()
def simplifier(fast_cfg) -> ImageSimplifier:
    return ImageSimplifier(fast_cfg)


def _gradient(size=(64, 64)) -> Image.Image:
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for x in range(size[0]):
        v = int(x / size[0] * 255)
        arr[:, x] = [v, 255 - v, 128]
    return Image.fromarray(arr)


def _solid(size=(64, 64), colour=(100, 150, 200)) -> Image.Image:
    return Image.new("RGB", size, colour)


def _checkerboard(size=(64, 64)) -> Image.Image:
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for y in range(size[1]):
        for x in range(size[0]):
            if (x // 8 + y // 8) % 2 == 0:
                arr[y, x] = [255, 255, 255]
            else:
                arr[y, x] = [0, 0, 0]
    return Image.fromarray(arr)


# ─────────────────────────────────────────────────────────────────────────────
# simplify() – end-to-end
# ─────────────────────────────────────────────────────────────────────────────

class TestSimplify:
    def test_returns_pil_image(self, simplifier):
        result = simplifier.simplify(_gradient())
        assert isinstance(result, Image.Image)

    def test_output_mode_is_rgb(self, simplifier):
        result = simplifier.simplify(_gradient())
        assert result.mode == "RGB"

    def test_output_size_preserved(self, simplifier):
        """Output size should match the input (no resizing in simplify)."""
        img = _gradient((64, 64))
        result = simplifier.simplify(img)
        assert result.size == img.size

    def test_accepts_rgba_input(self, simplifier):
        rgba = Image.new("RGBA", (64, 64), (200, 100, 50, 255))
        result = simplifier.simplify(rgba)
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_result_not_all_black(self, simplifier):
        result = simplifier.simplify(_gradient())
        assert np.array(result).mean() > 5

    def test_result_not_all_white(self, simplifier):
        result = simplifier.simplify(_gradient())
        assert np.array(result).mean() < 250

    def test_deterministic_for_same_input(self, simplifier):
        img = _gradient()
        r1 = np.array(simplifier.simplify(img))
        r2 = np.array(simplifier.simplify(img))
        np.testing.assert_array_equal(r1, r2)

    def test_solid_colour_survives(self, simplifier):
        """A uniform image should remain roughly uniform after simplification."""
        img = _solid(colour=(200, 200, 200))
        result = simplifier.simplify(img)
        arr = np.array(result)
        # Mean should stay near the input value
        assert 150 < float(arr.mean()) < 250

    def test_checkerboard_noise_reduced(self, simplifier):
        """Simplification should reduce high-frequency noise."""
        img = _checkerboard()
        original_std = float(np.array(img).std())
        result_std = float(np.array(simplifier.simplify(img)).std())
        # After bilateral + median blur the std should be lower
        assert result_std < original_std * 1.1  # at most equal with tolerance


# ─────────────────────────────────────────────────────────────────────────────
# extract_edges()
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractEdges:
    def test_returns_ndarray(self, simplifier):
        result = simplifier.extract_edges(_gradient())
        assert isinstance(result, np.ndarray)

    def test_output_shape_matches_input(self, simplifier):
        img = _gradient((64, 64))
        result = simplifier.extract_edges(img)
        assert result.shape == (64, 64)

    def test_output_dtype_uint8(self, simplifier):
        result = simplifier.extract_edges(_gradient())
        assert result.dtype == np.uint8

    def test_values_are_binary(self, simplifier):
        """Edge map must contain only 0 and 255."""
        result = simplifier.extract_edges(_gradient())
        unique = set(np.unique(result).tolist())
        assert unique.issubset({0, 255})

    def test_uniform_image_produces_no_edges(self, simplifier):
        """A perfectly uniform image has no edges."""
        img = _solid(colour=(128, 128, 128))
        result = simplifier.extract_edges(img)
        assert result.sum() == 0

    def test_gradient_produces_edge_on_boundary(self, simplifier):
        """A sharp colour boundary should produce detected edges."""
        # Create an image with a hard black/white split
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        arr[:, :32] = 0
        arr[:, 32:] = 255
        img = Image.fromarray(arr)
        result = simplifier.extract_edges(img)
        # At least some edge pixels should be non-zero
        assert result.sum() > 0


# ─────────────────────────────────────────────────────────────────────────────
# _bilateral_smooth (internal, but testable)
# ─────────────────────────────────────────────────────────────────────────────

class TestBilateralSmooth:
    def test_returns_ndarray(self, simplifier):
        import cv2
        arr = cv2.cvtColor(np.array(_gradient(), dtype=np.uint8), cv2.COLOR_RGB2BGR)
        result = simplifier._bilateral_smooth(arr)
        assert isinstance(result, np.ndarray)

    def test_shape_preserved(self, simplifier):
        import cv2
        arr = cv2.cvtColor(np.array(_gradient(), dtype=np.uint8), cv2.COLOR_RGB2BGR)
        result = simplifier._bilateral_smooth(arr)
        assert result.shape == arr.shape


# ─────────────────────────────────────────────────────────────────────────────
# _posterise (static, internal)
# ─────────────────────────────────────────────────────────────────────────────

class TestPosterise:
    def test_uint8_output(self):
        import cv2
        arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = ImageSimplifier._posterise(arr, levels=4)
        assert result.dtype == np.uint8

    def test_shape_preserved(self):
        arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = ImageSimplifier._posterise(arr, levels=4)
        assert result.shape == arr.shape

    def test_values_in_valid_range(self):
        arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = ImageSimplifier._posterise(arr, levels=6)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_reduces_unique_values(self):
        arr = np.arange(256, dtype=np.uint8).reshape(1, 256, 1)
        arr = np.repeat(arr, 3, axis=2)
        original_unique = len(np.unique(arr))
        result = ImageSimplifier._posterise(arr, levels=4)
        result_unique = len(np.unique(result))
        assert result_unique <= original_unique


# ─────────────────────────────────────────────────────────────────────────────
# _morphological_clean (internal)
# ─────────────────────────────────────────────────────────────────────────────

class TestMorphologicalClean:
    def test_returns_ndarray(self, simplifier):
        import cv2
        arr = cv2.cvtColor(np.array(_checkerboard(), dtype=np.uint8), cv2.COLOR_RGB2BGR)
        result = simplifier._morphological_clean(arr)
        assert isinstance(result, np.ndarray)

    def test_shape_preserved(self, simplifier):
        import cv2
        arr = cv2.cvtColor(np.array(_checkerboard(), dtype=np.uint8), cv2.COLOR_RGB2BGR)
        result = simplifier._morphological_clean(arr)
        assert result.shape == arr.shape

    def test_removes_small_noise(self, simplifier):
        """A single-pixel noise spike surrounded by white should be removed."""
        import cv2
        arr = np.full((64, 64, 3), 255, dtype=np.uint8)
        arr[32, 32] = [0, 0, 0]  # single black pixel
        # Convert to BGR for OpenCV
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        result = simplifier._morphological_clean(bgr)
        result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        # The noise pixel should be gone (morphological open removes it)
        assert result_rgb[32, 32].mean() > 200


# ─────────────────────────────────────────────────────────────────────────────
# _median_blur (static, internal)
# ─────────────────────────────────────────────────────────────────────────────

class TestMedianBlur:
    def test_returns_ndarray(self):
        arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = ImageSimplifier._median_blur(arr)
        assert isinstance(result, np.ndarray)

    def test_shape_preserved(self):
        arr = np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8)
        result = ImageSimplifier._median_blur(arr)
        assert result.shape == arr.shape

    def test_salt_pepper_noise_removed(self):
        """Median blur is the classic salt-and-pepper denoiser."""
        arr = np.full((64, 64, 3), 128, dtype=np.uint8)
        # Inject salt-and-pepper
        np.random.seed(0)
        noise_pixels = np.random.choice([0, 255], size=(20, 3)).astype(np.uint8)
        for i, (r, c) in enumerate(
            zip(
                np.random.randint(5, 59, 20),
                np.random.randint(5, 59, 20),
            )
        ):
            arr[r, c] = noise_pixels[i]

        result = ImageSimplifier._median_blur(arr)
        # After median blur most pixels should be closer to 128
        assert abs(float(result.mean()) - 128) < 20
