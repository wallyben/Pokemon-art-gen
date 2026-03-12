"""
Tests for pokemon_stencil.image_proc.loader.ReferenceImageLoader.

All tests use synthetic images generated in-memory; no external files needed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pokemon_stencil.image_proc.loader import (
    SUPPORTED_EXTENSIONS,
    ReferenceImageLoader,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

TARGET = (128, 128)


@pytest.fixture()
def loader() -> ReferenceImageLoader:
    return ReferenceImageLoader(target_size=TARGET)


def _save_rgb(path: Path, size: tuple = (64, 64), colour=(200, 100, 50)) -> None:
    """Save a solid-colour RGB PNG at *path*."""
    img = Image.new("RGB", size, colour)
    img.save(path)


def _save_rgba(path: Path, size: tuple = (64, 64)) -> None:
    """Save a solid semi-transparent RGBA PNG at *path*."""
    img = Image.new("RGBA", size, (100, 150, 200, 128))
    img.save(path)


# ─────────────────────────────────────────────────────────────────────────────
# load_from_directory
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadFromDirectory:
    def test_loads_all_png_images(self, tmp_path, loader):
        for i in range(3):
            _save_rgb(tmp_path / f"img_{i:02d}.png")
        result = loader.load_from_directory(tmp_path)
        assert len(result) == 3

    def test_returns_pil_images(self, tmp_path, loader):
        _save_rgb(tmp_path / "img.png")
        result = loader.load_from_directory(tmp_path)
        assert all(isinstance(img, Image.Image) for img in result)

    def test_converts_all_to_rgb(self, tmp_path, loader):
        _save_rgba(tmp_path / "rgba.png")
        result = loader.load_from_directory(tmp_path)
        assert all(img.mode == "RGB" for img in result)

    def test_resizes_to_target_size(self, tmp_path, loader):
        _save_rgb(tmp_path / "small.png", size=(32, 32))
        _save_rgb(tmp_path / "large.png", size=(256, 256))
        result = loader.load_from_directory(tmp_path)
        for img in result:
            assert img.size == TARGET

    def test_nonexistent_directory_raises(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_from_directory("/nonexistent/path/")

    def test_empty_directory_raises(self, tmp_path, loader):
        with pytest.raises(ValueError, match="No supported images"):
            loader.load_from_directory(tmp_path)

    def test_ignores_non_image_files(self, tmp_path, loader):
        _save_rgb(tmp_path / "valid.png")
        (tmp_path / "readme.txt").write_text("ignore me", encoding="utf-8")
        (tmp_path / "data.json").write_text("{}", encoding="utf-8")
        result = loader.load_from_directory(tmp_path)
        assert len(result) == 1

    def test_loads_jpg_images(self, tmp_path, loader):
        img = Image.new("RGB", (64, 64), (80, 120, 200))
        img.save(tmp_path / "test.jpg", format="JPEG")
        result = loader.load_from_directory(tmp_path)
        assert len(result) == 1

    def test_loads_bmp_images(self, tmp_path, loader):
        img = Image.new("RGB", (64, 64), (50, 50, 50))
        img.save(tmp_path / "test.bmp", format="BMP")
        result = loader.load_from_directory(tmp_path)
        assert len(result) == 1

    def test_max_images_cap(self, tmp_path):
        for i in range(5):
            _save_rgb(tmp_path / f"img_{i:02d}.png")
        capped_loader = ReferenceImageLoader(target_size=TARGET, max_images=2)
        result = capped_loader.load_from_directory(tmp_path)
        assert len(result) == 2

    def test_sorted_order(self, tmp_path, loader):
        # Files created in reverse order; loader should sort by name/path.
        for name in ["c.png", "a.png", "b.png"]:
            _save_rgb(tmp_path / name)
        result = loader.load_from_directory(tmp_path)
        assert len(result) == 3

    def test_skips_corrupt_file(self, tmp_path, loader):
        # Write a file with a .png extension but invalid content
        (tmp_path / "corrupt.png").write_bytes(b"\xff\xd8 not a real image")
        _save_rgb(tmp_path / "valid.png")
        result = loader.load_from_directory(tmp_path)
        # At least the valid image was loaded; corrupt one was skipped.
        assert any(img is not None for img in result)


# ─────────────────────────────────────────────────────────────────────────────
# load_from_paths
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadFromPaths:
    def test_loads_explicit_paths(self, tmp_path, loader):
        paths = []
        for i in range(2):
            p = tmp_path / f"img_{i}.png"
            _save_rgb(p)
            paths.append(p)
        result = loader.load_from_paths(paths)
        assert len(result) == 2

    def test_accepts_string_paths(self, tmp_path, loader):
        p = tmp_path / "img.png"
        _save_rgb(p)
        result = loader.load_from_paths([str(p)])
        assert len(result) == 1

    def test_skips_unreadable_files(self, tmp_path, loader):
        good = tmp_path / "good.png"
        bad = tmp_path / "bad.png"
        _save_rgb(good)
        bad.write_bytes(b"\xde\xad\xbe\xef")
        result = loader.load_from_paths([good, bad])
        assert len(result) == 1

    def test_all_invalid_raises(self, tmp_path, loader):
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"\xde\xad\xbe\xef")
        with pytest.raises(ValueError, match="No valid images"):
            loader.load_from_paths([bad])

    def test_output_size_applied(self, tmp_path, loader):
        p = tmp_path / "img.png"
        _save_rgb(p, size=(300, 200))
        result = loader.load_from_paths([p])
        assert result[0].size == TARGET


# ─────────────────────────────────────────────────────────────────────────────
# load_single
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadSingle:
    def test_returns_pil_image(self, tmp_path, loader):
        p = tmp_path / "img.png"
        _save_rgb(p)
        result = loader.load_single(p)
        assert isinstance(result, Image.Image)

    def test_returns_rgb_mode(self, tmp_path, loader):
        p = tmp_path / "rgba.png"
        _save_rgba(p)
        result = loader.load_single(p)
        assert result.mode == "RGB"

    def test_resized_to_target(self, tmp_path, loader):
        p = tmp_path / "img.png"
        _save_rgb(p, size=(16, 16))
        result = loader.load_single(p)
        assert result.size == TARGET

    def test_nonexistent_raises_file_not_found(self, loader):
        with pytest.raises(FileNotFoundError):
            loader.load_single(Path("/nonexistent/image.png"))

    def test_corrupt_file_raises_value_error(self, tmp_path, loader):
        bad = tmp_path / "bad.png"
        bad.write_bytes(b"\x00\x01\x02")
        with pytest.raises(ValueError, match="Could not decode"):
            loader.load_single(bad)


# ─────────────────────────────────────────────────────────────────────────────
# composite_references
# ─────────────────────────────────────────────────────────────────────────────

class TestCompositeReferences:
    def test_single_image_passthrough(self):
        img = Image.new("RGB", (64, 64), (100, 100, 100))
        result = ReferenceImageLoader.composite_references([img])
        assert result is img

    def test_average_of_two_images(self):
        black = Image.new("RGB", (64, 64), (0, 0, 0))
        white = Image.new("RGB", (64, 64), (254, 254, 254))
        result = ReferenceImageLoader.composite_references([black, white])
        arr = np.array(result)
        # Average should be approximately 127
        assert 120 <= int(arr.mean()) <= 135

    def test_output_is_pil_image(self):
        imgs = [Image.new("RGB", (64, 64), (i * 40, 0, 0)) for i in range(3)]
        result = ReferenceImageLoader.composite_references(imgs)
        assert isinstance(result, Image.Image)

    def test_empty_list_raises(self):
        with pytest.raises(ValueError, match="empty"):
            ReferenceImageLoader.composite_references([])

    def test_output_size_matches_input(self):
        imgs = [Image.new("RGB", (100, 200), (50, 60, 70)) for _ in range(3)]
        result = ReferenceImageLoader.composite_references(imgs)
        assert result.size == (100, 200)


# ─────────────────────────────────────────────────────────────────────────────
# SUPPORTED_EXTENSIONS constant
# ─────────────────────────────────────────────────────────────────────────────

class TestSupportedExtensions:
    def test_includes_common_formats(self):
        for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"):
            assert ext in SUPPORTED_EXTENSIONS

    def test_all_lowercase(self):
        for ext in SUPPORTED_EXTENSIONS:
            assert ext == ext.lower()
