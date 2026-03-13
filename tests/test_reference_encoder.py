"""
Tests for pokemon_stencil.image_gen.reference_encoder.
"""

from __future__ import annotations

from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from pokemon_stencil.image_gen.reference_encoder import (
    CharacterReferenceEncoder,
    _load_images_from_dir,
    _pil_to_tensor,
)

# Skip torch-dependent tests when the library is not installed.
try:
    import torch as _torch_check  # noqa: F401
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

requires_torch = pytest.mark.skipif(
    not TORCH_AVAILABLE, reason="torch not installed"
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _solid_image(color=(128, 64, 32), size=(64, 64)) -> Image.Image:
    """Create a solid-colour PIL Image."""
    arr = np.full((*size[::-1], 3), color, dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _save_image(directory: Path, name: str, color=(100, 100, 100)) -> Path:
    """Save a small solid PNG to *directory*."""
    path = directory / name
    _solid_image(color=color, size=(32, 32)).save(path)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# _pil_to_tensor
# ─────────────────────────────────────────────────────────────────────────────

class TestPilToTensor:
    @requires_torch
    def test_shape(self):
        import torch
        img = _solid_image(size=(32, 32))
        tensor = _pil_to_tensor(img)
        assert tensor.shape == (3, 32, 32)

    @requires_torch
    def test_range(self):
        import torch
        img = _solid_image(size=(8, 8))
        tensor = _pil_to_tensor(img)
        # Normalised to [-1, 1]
        assert float(tensor.min()) >= -1.0
        assert float(tensor.max()) <= 1.0

    @requires_torch
    def test_dtype_float32(self):
        import torch
        img = _solid_image(size=(8, 8))
        tensor = _pil_to_tensor(img)
        assert tensor.dtype == torch.float32


# ─────────────────────────────────────────────────────────────────────────────
# _load_images_from_dir
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadImagesFromDir:
    def test_empty_directory_returns_empty(self, tmp_path):
        assert _load_images_from_dir(tmp_path, max_images=10) == []

    def test_nonexistent_directory_returns_empty(self, tmp_path):
        assert _load_images_from_dir(tmp_path / "nope", max_images=10) == []

    def test_loads_png_files(self, tmp_path):
        _save_image(tmp_path, "a.png")
        _save_image(tmp_path, "b.png")
        images = _load_images_from_dir(tmp_path, max_images=10)
        assert len(images) == 2
        assert all(isinstance(im, Image.Image) for im in images)

    def test_respects_max_images(self, tmp_path):
        for i in range(5):
            _save_image(tmp_path, f"img{i}.png")
        images = _load_images_from_dir(tmp_path, max_images=3)
        assert len(images) == 3

    def test_skips_non_image_files(self, tmp_path):
        _save_image(tmp_path, "real.png")
        (tmp_path / "notes.txt").write_text("text", encoding="utf-8")
        images = _load_images_from_dir(tmp_path, max_images=10)
        assert len(images) == 1


# ─────────────────────────────────────────────────────────────────────────────
# CharacterReferenceEncoder.encode_references – pixel fallback (no VAE)
# ─────────────────────────────────────────────────────────────────────────────

class TestCharacterReferenceEncoderPixelFallback:
    """Tests that run without a real VAE, using pixel-space blending."""

    def _make_encoder(self, size=(64, 64)) -> CharacterReferenceEncoder:
        return CharacterReferenceEncoder(vae=None, image_size=size, max_images=30)

    def test_returns_none_for_empty_list(self):
        encoder = self._make_encoder()
        assert encoder.encode_references([]) is None

    def test_single_image_returns_image(self):
        encoder = self._make_encoder(size=(32, 32))
        img = _solid_image(size=(64, 64))
        result = encoder.encode_references([img])
        assert isinstance(result, Image.Image)
        assert result.size == (32, 32)

    def test_multiple_images_returns_image(self):
        encoder = self._make_encoder(size=(32, 32))
        images = [_solid_image(color=(c, c, c)) for c in [50, 100, 150]]
        result = encoder.encode_references(images)
        assert isinstance(result, Image.Image)

    def test_averaged_pixel_values(self):
        """Averaged image should have intermediate pixel values."""
        encoder = self._make_encoder(size=(8, 8))
        black = _solid_image(color=(0, 0, 0), size=(8, 8))
        white = _solid_image(color=(255, 255, 255), size=(8, 8))
        result = encoder.encode_references([black, white])
        arr = np.array(result)
        # Mean of 0 and 255 = 127 (±1 for rounding)
        assert abs(int(arr.mean()) - 127) <= 2

    def test_respects_max_images(self):
        encoder = CharacterReferenceEncoder(vae=None, image_size=(8, 8), max_images=2)
        images = [_solid_image(color=(i * 10, i * 10, i * 10)) for i in range(5)]
        result = encoder.encode_references(images)
        assert result is not None  # Should still return a result from 2 images

    def test_mode_is_rgb(self):
        encoder = self._make_encoder(size=(16, 16))
        result = encoder.encode_references([_solid_image()])
        assert result.mode == "RGB"


# ─────────────────────────────────────────────────────────────────────────────
# CharacterReferenceEncoder.encode_from_directory
# ─────────────────────────────────────────────────────────────────────────────

class TestEncodeFromDirectory:
    def test_returns_none_for_empty_directory(self, tmp_path):
        encoder = CharacterReferenceEncoder(vae=None, image_size=(16, 16))
        result = encoder.encode_from_directory(tmp_path)
        assert result is None

    def test_returns_image_when_directory_has_images(self, tmp_path):
        _save_image(tmp_path, "ref.png", color=(200, 100, 50))
        encoder = CharacterReferenceEncoder(vae=None, image_size=(16, 16))
        result = encoder.encode_from_directory(tmp_path)
        assert isinstance(result, Image.Image)

    def test_returns_none_for_nonexistent_directory(self, tmp_path):
        encoder = CharacterReferenceEncoder(vae=None, image_size=(16, 16))
        result = encoder.encode_from_directory(tmp_path / "nope")
        assert result is None


# ─────────────────────────────────────────────────────────────────────────────
# VAE failure graceful fallback
# ─────────────────────────────────────────────────────────────────────────────

class TestVAEFallback:
    def test_falls_back_to_pixel_blend_on_vae_error(self):
        """If the VAE raises, encode_references falls back to pixel blending."""
        broken_vae = MagicMock()
        broken_vae.encode.side_effect = RuntimeError("simulated VAE failure")

        encoder = CharacterReferenceEncoder(vae=broken_vae, image_size=(16, 16))
        images = [_solid_image(size=(16, 16)) for _ in range(2)]
        result = encoder.encode_references(images)
        assert isinstance(result, Image.Image)
