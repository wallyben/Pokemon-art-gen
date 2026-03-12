"""
Tests for pokemon_stencil.models.reference_encoder.ReferenceEncoder.

All tests use synthetic PIL Images; no Stable Diffusion inference or
GPU access is required.  VAE behaviour is mocked where needed.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from pokemon_stencil.models.reference_encoder import ReferenceEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _solid_rgb(r: int, g: int, b: int, size: int = 64) -> Image.Image:
    """Create a solid-colour RGB image."""
    arr = np.full((size, size, 3), [r, g, b], dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _random_rgb(seed: int = 0, size: int = 64) -> Image.Image:
    """Create a random-noise RGB image."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


# ─────────────────────────────────────────────────────────────────────────────
# Construction
# ─────────────────────────────────────────────────────────────────────────────

class TestReferenceEncoderInit:
    def test_default_target_size(self):
        enc = ReferenceEncoder()
        assert enc.target_size == (512, 512)

    def test_custom_target_size(self):
        enc = ReferenceEncoder(target_size=(256, 256))
        assert enc.target_size == (256, 256)


# ─────────────────────────────────────────────────────────────────────────────
# _resize_all
# ─────────────────────────────────────────────────────────────────────────────

class TestResizeAll:
    def test_output_size_matches_target(self):
        enc = ReferenceEncoder(target_size=(32, 32))
        imgs = [_solid_rgb(255, 0, 0, size=64), _solid_rgb(0, 255, 0, size=128)]
        resized = enc._resize_all(imgs)
        for img in resized:
            assert img.size == (32, 32)

    def test_converts_to_rgb(self):
        enc = ReferenceEncoder(target_size=(32, 32))
        rgba = Image.new("RGBA", (32, 32), (100, 100, 100, 255))
        resized = enc._resize_all([rgba])
        assert resized[0].mode == "RGB"

    def test_grayscale_converted(self):
        enc = ReferenceEncoder(target_size=(32, 32))
        gray = Image.new("L", (32, 32), 128)
        resized = enc._resize_all([gray])
        assert resized[0].mode == "RGB"


# ─────────────────────────────────────────────────────────────────────────────
# _pixel_blend
# ─────────────────────────────────────────────────────────────────────────────

class TestPixelBlend:
    def test_single_image_returns_same_content(self):
        img = _solid_rgb(100, 150, 200, size=32)
        result = ReferenceEncoder._pixel_blend([img])
        arr_in = np.array(img)
        arr_out = np.array(result)
        np.testing.assert_array_equal(arr_in, arr_out)

    def test_two_images_averages_correctly(self):
        """Average of black and white should produce mid-grey."""
        black = _solid_rgb(0, 0, 0, size=32)
        white = _solid_rgb(255, 255, 255, size=32)
        result = ReferenceEncoder._pixel_blend([black, white])
        arr = np.array(result)
        # uint8 arithmetic: (0+255)//2 = 127
        assert np.all(arr == 127)

    def test_output_is_pil_image(self):
        result = ReferenceEncoder._pixel_blend([_solid_rgb(0, 0, 0, size=16)])
        assert isinstance(result, Image.Image)

    def test_output_mode_is_rgb(self):
        result = ReferenceEncoder._pixel_blend([_solid_rgb(10, 20, 30, size=16)])
        assert result.mode == "RGB"

    def test_three_images_average(self):
        imgs = [
            _solid_rgb(0, 0, 0, size=16),
            _solid_rgb(150, 150, 150, size=16),
            _solid_rgb(255, 255, 255, size=16),
        ]
        result = ReferenceEncoder._pixel_blend(imgs)
        arr = np.array(result, dtype=np.float32)
        expected = (0 + 150 + 255) / 3
        assert abs(float(arr[0, 0, 0]) - expected) < 2  # tolerate uint8 rounding


# ─────────────────────────────────────────────────────────────────────────────
# blend() – public convenience method
# ─────────────────────────────────────────────────────────────────────────────

class TestBlend:
    def test_returns_pil_image(self):
        enc = ReferenceEncoder(target_size=(32, 32))
        result = enc.blend([_solid_rgb(255, 0, 0)])
        assert isinstance(result, Image.Image)

    def test_output_size_matches_target(self):
        enc = ReferenceEncoder(target_size=(48, 48))
        result = enc.blend([_random_rgb(), _random_rgb(seed=1)])
        assert result.size == (48, 48)

    def test_empty_list_raises(self):
        enc = ReferenceEncoder()
        with pytest.raises(ValueError):
            enc.blend([])

    def test_rgba_input_accepted(self):
        enc = ReferenceEncoder(target_size=(32, 32))
        rgba = Image.new("RGBA", (64, 64), (200, 100, 50, 255))
        result = enc.blend([rgba])
        assert result.mode == "RGB"


# ─────────────────────────────────────────────────────────────────────────────
# encode() – fallback path (no VAE)
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodeFallback:
    def test_returns_pil_when_no_vae(self):
        enc = ReferenceEncoder(target_size=(32, 32))
        result = enc.encode([_solid_rgb(0, 128, 255)], vae=None)
        assert isinstance(result, Image.Image)

    def test_empty_list_raises(self):
        enc = ReferenceEncoder()
        with pytest.raises(ValueError):
            enc.encode([], vae=None)

    def test_multiple_references_blended(self):
        enc = ReferenceEncoder(target_size=(32, 32))
        imgs = [_solid_rgb(0, 0, 0), _solid_rgb(255, 255, 255)]
        result = enc.encode(imgs, vae=None)
        assert isinstance(result, Image.Image)
        arr = np.array(result)
        # Should be mid-grey
        assert abs(int(arr[0, 0, 0]) - 127) <= 2


# ─────────────────────────────────────────────────────────────────────────────
# encode() – VAE path (mocked)
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodeVAE:
    def _make_mock_vae(self, latent_shape=(1, 4, 4, 4)):
        """Build a mock VAE that returns a fixed latent tensor."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        mock_vae = MagicMock()
        fake_latent = torch.zeros(latent_shape)
        mock_dist = MagicMock()
        mock_dist.mean = fake_latent
        mock_vae_output = MagicMock()
        mock_vae_output.latent_dist = mock_dist
        mock_vae.encode.return_value = mock_vae_output
        return mock_vae

    def test_vae_path_returns_tensor(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        enc = ReferenceEncoder(target_size=(32, 32))
        mock_vae = self._make_mock_vae()
        result = enc.encode([_solid_rgb(100, 100, 100)], vae=mock_vae)
        assert isinstance(result, torch.Tensor)

    def test_vae_called_once_per_image(self):
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        enc = ReferenceEncoder(target_size=(32, 32))
        mock_vae = self._make_mock_vae()
        imgs = [_solid_rgb(0, 0, 0), _solid_rgb(128, 128, 128)]
        enc.encode(imgs, vae=mock_vae)
        assert mock_vae.encode.call_count == 2

    def test_vae_failure_falls_back_to_blend(self):
        """If VAE raises, encode() should return a PIL Image blend."""
        enc = ReferenceEncoder(target_size=(32, 32))
        mock_vae = MagicMock()
        mock_vae.encode.side_effect = RuntimeError("VAE broken")
        result = enc.encode([_solid_rgb(50, 50, 50)], vae=mock_vae)
        assert isinstance(result, Image.Image)
