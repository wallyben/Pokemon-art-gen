"""
Tests for pokemon_stencil.data.reference_fetcher.ReferenceImageFetcher.

All HTTP requests are mocked via ``unittest.mock``.
No network access or Stable Diffusion execution is required.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from pokemon_stencil.data.reference_fetcher import (
    ReferenceImageFetcher,
    _extract_sprite_urls,
    _normalise_size,
    _remove_transparent_background,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Serialise a PIL Image to raw bytes."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_response(status: int = 200, content: bytes = b"", json_data: dict = None):
    """Build a mock ``requests.Response``."""
    resp = MagicMock()
    resp.status_code = status
    resp.content = content
    resp.json.return_value = json_data or {}
    if status >= 400:
        resp.raise_for_status.side_effect = Exception(f"HTTP {status}")
    else:
        resp.raise_for_status.return_value = None
    return resp


def _make_api_response(pokemon_id: int = 25) -> dict:
    """Minimal PokéAPI JSON payload."""
    return {
        "id": pokemon_id,
        "name": "pikachu",
        "sprites": {
            "front_default": (
                f"https://raw.githubusercontent.com/PokeAPI/sprites/master/"
                f"sprites/pokemon/{pokemon_id}.png"
            ),
            "front_shiny": None,
            "other": {
                "official-artwork": {
                    "front_default": (
                        f"https://raw.githubusercontent.com/PokeAPI/sprites/master/"
                        f"sprites/pokemon/other/official-artwork/{pokemon_id}.png"
                    )
                }
            },
        },
    }


def _big_rgb_image(size: tuple = (512, 512)) -> Image.Image:
    return Image.new("RGB", size, (200, 150, 100))


def _small_rgb_image(size: tuple = (64, 64)) -> Image.Image:
    return Image.new("RGB", size, (100, 200, 50))


# ─────────────────────────────────────────────────────────────────────────────
# fetch_references – mocked HTTP
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchReferences:
    def _patch_requests(self, responses: list):
        """
        Return a context manager that patches ``requests.get`` with
        successive responses from *responses*.
        """
        mock_get = MagicMock(side_effect=responses)
        return patch(
            "pokemon_stencil.data.reference_fetcher.requests",
            get=mock_get,
        )

    def test_returns_list_of_paths(self, tmp_path):
        big_img = _big_rgb_image()
        api_resp = _make_response(json_data=_make_api_response())
        img_resp = _make_response(content=_pil_to_bytes(big_img))

        with patch(
            "pokemon_stencil.data.reference_fetcher._import_requests"
        ) as mock_import:
            requests_mock = MagicMock()
            requests_mock.get.side_effect = [
                api_resp,
                img_resp,
                _make_response(status=404, content=b""),
                _make_response(status=404, content=b""),
                _make_response(status=404, content=b""),
                _make_response(status=404, content=b""),
            ]
            mock_import.return_value = requests_mock

            fetcher = ReferenceImageFetcher(max_images=5)
            saved = fetcher.fetch_references("pikachu", tmp_path / "refs")

        assert isinstance(saved, list)
        assert all(isinstance(p, Path) for p in saved)

    def test_saves_png_files(self, tmp_path):
        big_img = _big_rgb_image()
        api_resp = _make_response(json_data=_make_api_response())
        img_resp = _make_response(content=_pil_to_bytes(big_img))
        fail_resp = _make_response(status=404, content=b"")

        with patch(
            "pokemon_stencil.data.reference_fetcher._import_requests"
        ) as mock_import:
            requests_mock = MagicMock()
            requests_mock.get.side_effect = [api_resp, img_resp] + [fail_resp] * 10
            mock_import.return_value = requests_mock

            fetcher = ReferenceImageFetcher(max_images=5)
            output_dir = tmp_path / "refs_png"
            saved = fetcher.fetch_references("pikachu", output_dir)

        for path in saved:
            assert path.suffix == ".png"
            assert path.exists()

    def test_creates_output_directory(self, tmp_path):
        dest = tmp_path / "nested" / "refs" / "pikachu"
        assert not dest.exists()

        api_resp = _make_response(json_data=_make_api_response())
        fail_resp = _make_response(status=404, content=b"")

        with patch(
            "pokemon_stencil.data.reference_fetcher._import_requests"
        ) as mock_import:
            requests_mock = MagicMock()
            requests_mock.get.side_effect = [api_resp] + [fail_resp] * 20
            mock_import.return_value = requests_mock

            fetcher = ReferenceImageFetcher(max_images=5)
            fetcher.fetch_references("pikachu", dest)

        assert dest.is_dir()

    def test_discards_low_resolution_images(self, tmp_path):
        small_img = _small_rgb_image(size=(64, 64))  # below 256 px
        api_resp = _make_response(json_data=_make_api_response())
        img_resp = _make_response(content=_pil_to_bytes(small_img))
        fail_resp = _make_response(status=404, content=b"")

        with patch(
            "pokemon_stencil.data.reference_fetcher._import_requests"
        ) as mock_import:
            requests_mock = MagicMock()
            requests_mock.get.side_effect = [api_resp, img_resp] + [fail_resp] * 10
            mock_import.return_value = requests_mock

            fetcher = ReferenceImageFetcher(max_images=5, min_resolution=256)
            saved = fetcher.fetch_references("pikachu", tmp_path / "refs")

        # The small image should have been discarded.
        assert len(saved) == 0

    def test_api_failure_returns_empty_list(self, tmp_path):
        api_resp = _make_response(status=404, content=b"")

        with patch(
            "pokemon_stencil.data.reference_fetcher._import_requests"
        ) as mock_import:
            requests_mock = MagicMock()
            requests_mock.get.side_effect = [api_resp]
            mock_import.return_value = requests_mock

            fetcher = ReferenceImageFetcher(max_images=5)
            saved = fetcher.fetch_references("unknown_mon", tmp_path / "refs")

        assert saved == []

    def test_respects_max_images_limit(self, tmp_path):
        big_img = _big_rgb_image()
        api_resp = _make_response(json_data=_make_api_response())
        img_resps = [
            _make_response(content=_pil_to_bytes(big_img))
            for _ in range(20)
        ]

        with patch(
            "pokemon_stencil.data.reference_fetcher._import_requests"
        ) as mock_import:
            requests_mock = MagicMock()
            requests_mock.get.side_effect = [api_resp] + img_resps
            mock_import.return_value = requests_mock

            max_imgs = 3
            fetcher = ReferenceImageFetcher(max_images=max_imgs)
            saved = fetcher.fetch_references("pikachu", tmp_path / "refs")

        assert len(saved) <= max_imgs

    def test_saved_images_are_valid_pngs(self, tmp_path):
        big_img = _big_rgb_image(size=(300, 300))
        api_resp = _make_response(json_data=_make_api_response())
        img_resp = _make_response(content=_pil_to_bytes(big_img))
        fail_resp = _make_response(status=404, content=b"")

        with patch(
            "pokemon_stencil.data.reference_fetcher._import_requests"
        ) as mock_import:
            requests_mock = MagicMock()
            requests_mock.get.side_effect = [api_resp, img_resp] + [fail_resp] * 10
            mock_import.return_value = requests_mock

            fetcher = ReferenceImageFetcher(max_images=5)
            saved = fetcher.fetch_references("pikachu", tmp_path / "refs")

        for path in saved:
            loaded = Image.open(path)
            assert loaded.mode == "RGB"

    def test_transparent_images_converted_to_rgb(self, tmp_path):
        rgba_img = Image.new("RGBA", (300, 300), (255, 0, 0, 128))
        api_resp = _make_response(json_data=_make_api_response())
        img_resp = _make_response(content=_pil_to_bytes(rgba_img))
        fail_resp = _make_response(status=404, content=b"")

        with patch(
            "pokemon_stencil.data.reference_fetcher._import_requests"
        ) as mock_import:
            requests_mock = MagicMock()
            requests_mock.get.side_effect = [api_resp, img_resp] + [fail_resp] * 10
            mock_import.return_value = requests_mock

            fetcher = ReferenceImageFetcher(max_images=5)
            saved = fetcher.fetch_references("pikachu", tmp_path / "refs")

        for path in saved:
            assert Image.open(path).mode == "RGB"


# ─────────────────────────────────────────────────────────────────────────────
# prepare_reference_images
# ─────────────────────────────────────────────────────────────────────────────

class TestPrepareReferenceImages:
    def test_returns_list_of_paths(self, tmp_path):
        img = _big_rgb_image()
        p = tmp_path / "ref_000.png"
        img.save(p)

        fetcher = ReferenceImageFetcher()
        kept = fetcher.prepare_reference_images(tmp_path)
        assert isinstance(kept, list)

    def test_empty_directory_returns_empty(self, tmp_path):
        fetcher = ReferenceImageFetcher()
        kept = fetcher.prepare_reference_images(tmp_path)
        assert kept == []

    def test_removes_duplicate_files(self, tmp_path):
        """Two copies of the same image should be reduced to one."""
        img = _big_rgb_image()
        img.save(tmp_path / "a.png")
        img.save(tmp_path / "b.png")

        fetcher = ReferenceImageFetcher()
        kept = fetcher.prepare_reference_images(tmp_path)
        assert len(kept) == 1

    def test_keeps_distinct_images(self, tmp_path):
        """Structurally distinct images should both be kept."""
        # Use a checkerboard and a gradient – these have very different
        # perceptual hashes regardless of the imagehash backend.
        arr_check = np.zeros((300, 300, 3), dtype=np.uint8)
        for y in range(300):
            for x in range(300):
                if (x // 30 + y // 30) % 2 == 0:
                    arr_check[y, x, :] = 255
        img_a = Image.fromarray(arr_check)

        arr_grad = np.zeros((300, 300, 3), dtype=np.uint8)
        for x in range(300):
            val = int(255 * x / 299)
            arr_grad[:, x, :] = val
        img_b = Image.fromarray(arr_grad)

        img_a.save(tmp_path / "a.png")
        img_b.save(tmp_path / "b.png")

        fetcher = ReferenceImageFetcher()
        kept = fetcher.prepare_reference_images(tmp_path)
        assert len(kept) == 2

    def test_transparent_background_removed(self, tmp_path):
        rgba_img = Image.new("RGBA", (300, 300), (100, 200, 50, 128))
        p = tmp_path / "rgba.png"
        rgba_img.save(p)

        fetcher = ReferenceImageFetcher()
        fetcher.prepare_reference_images(tmp_path)

        result = Image.open(p)
        assert result.mode == "RGB"

    def test_large_images_downscaled(self, tmp_path):
        large_img = Image.new("RGB", (1024, 1024), (200, 150, 100))
        p = tmp_path / "large.png"
        large_img.save(p)

        fetcher = ReferenceImageFetcher()
        fetcher.prepare_reference_images(tmp_path)

        result = Image.open(p)
        assert max(result.size) <= 512

    def test_small_images_not_upscaled(self, tmp_path):
        small_img = Image.new("RGB", (100, 100), (50, 100, 150))
        p = tmp_path / "small.png"
        small_img.save(p)

        fetcher = ReferenceImageFetcher()
        fetcher.prepare_reference_images(tmp_path)

        result = Image.open(p)
        assert result.size == (100, 100)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestHelpers:
    def test_remove_transparent_background_rgba(self):
        rgba = Image.new("RGBA", (32, 32), (0, 0, 255, 0))  # fully transparent blue
        result = _remove_transparent_background(rgba)
        assert result.mode == "RGB"
        arr = np.array(result)
        assert arr.mean() > 200  # mostly white

    def test_remove_transparent_background_rgb_passthrough(self):
        rgb = Image.new("RGB", (32, 32), (200, 100, 50))
        result = _remove_transparent_background(rgb)
        assert result.mode == "RGB"

    def test_normalise_size_downscales_large(self):
        img = Image.new("RGB", (1024, 768), (0, 0, 0))
        result = _normalise_size(img, max_size=512)
        assert max(result.size) <= 512
        # Aspect ratio preserved.
        w, h = result.size
        assert abs(w / h - 1024 / 768) < 0.01

    def test_normalise_size_no_upscale(self):
        img = Image.new("RGB", (100, 100), (0, 0, 0))
        result = _normalise_size(img, max_size=512)
        assert result.size == (100, 100)

    def test_extract_sprite_urls_flat(self):
        sprites = {
            "front_default": "https://example.com/a.png",
            "back_default": "https://example.com/b.png",
            "front_shiny": None,
        }
        urls = _extract_sprite_urls(sprites)
        assert "https://example.com/a.png" in urls
        assert "https://example.com/b.png" in urls
        assert None not in urls

    def test_extract_sprite_urls_nested(self):
        sprites = {
            "other": {
                "official-artwork": {
                    "front_default": "https://example.com/art.png"
                }
            }
        }
        urls = _extract_sprite_urls(sprites)
        assert "https://example.com/art.png" in urls

    def test_extract_sprite_urls_excludes_non_http(self):
        sprites = {
            "front_default": "ftp://example.com/a.png",
            "other": "https://example.com/b.png",
        }
        urls = _extract_sprite_urls(sprites)
        assert "ftp://example.com/a.png" not in urls
        assert "https://example.com/b.png" in urls


# ─────────────────────────────────────────────────────────────────────────────
# Configuration integration
# ─────────────────────────────────────────────────────────────────────────────

class TestConfigIntegration:
    def test_generation_config_has_auto_fetch_references(self):
        from pokemon_stencil.config import GenerationConfig
        cfg = GenerationConfig()
        assert hasattr(cfg, "auto_fetch_references")
        assert cfg.auto_fetch_references is False

    def test_generation_config_has_max_reference_images(self):
        from pokemon_stencil.config import GenerationConfig
        cfg = GenerationConfig()
        assert hasattr(cfg, "max_reference_images")
        assert cfg.max_reference_images == 25

    def test_auto_fetch_disabled_by_default(self):
        """auto_fetch_references must default to False for backward compat."""
        from pokemon_stencil.config import GenerationConfig
        cfg = GenerationConfig()
        assert cfg.auto_fetch_references is False

    def test_factory_runner_auto_fetch_skipped_when_disabled(self, tmp_path):
        """When auto_fetch_references=False, the fetcher should not be called.

        ReferenceImageFetcher is a local import inside factory_runner.run(),
        so we patch it at its definition site.
        """
        from pokemon_stencil.config import PipelineConfig
        from pokemon_stencil.factory.factory_runner import FactoryRunner

        config = PipelineConfig(skip_generation=True)
        config.generation.auto_fetch_references = False
        config.factory.count = 1
        config.factory.top_k = 1
        config.output.base_dir = tmp_path

        # Create a dummy reference image so the run doesn't fail on load.
        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        Image.new("RGB", (64, 64), (100, 150, 200)).save(ref_dir / "ref.png")

        runner = FactoryRunner(config)

        with patch(
            "pokemon_stencil.data.reference_fetcher.ReferenceImageFetcher"
        ) as mock_fetcher_cls:
            try:
                runner.run("Pikachu", reference_dir=ref_dir)
            except Exception:
                pass  # Pipeline errors are fine; we just check fetch call.
            mock_fetcher_cls.assert_not_called()

    def test_factory_runner_auto_fetch_triggered_on_empty_dir(self, tmp_path):
        """When auto_fetch_references=True and refs empty, fetcher is called.

        ReferenceImageFetcher is a local import inside factory_runner.run(),
        so we patch at the module level where it is defined.
        """
        from pokemon_stencil.config import PipelineConfig
        from pokemon_stencil.factory.factory_runner import FactoryRunner

        config = PipelineConfig(skip_generation=True)
        config.generation.auto_fetch_references = True
        config.generation.max_reference_images = 5
        config.factory.count = 1
        config.factory.top_k = 1
        config.output.base_dir = tmp_path

        ref_dir = tmp_path / "refs_empty"
        ref_dir.mkdir()  # empty directory

        runner = FactoryRunner(config)

        mock_fetcher_instance = MagicMock()
        mock_fetcher_instance.fetch_references.return_value = []

        with patch(
            "pokemon_stencil.data.reference_fetcher.ReferenceImageFetcher",
            return_value=mock_fetcher_instance,
        ) as mock_fetcher_cls:
            try:
                runner.run("Pikachu", reference_dir=ref_dir)
            except Exception:
                pass
            # The local import creates the class; verify it was instantiated.
            mock_fetcher_cls.assert_called_once()
            mock_fetcher_instance.fetch_references.assert_called_once()
            call_args = mock_fetcher_instance.fetch_references.call_args
            assert call_args[0][0] == "Pikachu"
