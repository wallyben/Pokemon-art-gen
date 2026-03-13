"""
Tests for pokemon_stencil.models.model_manager.

No network access or real model downloads are performed; huggingface_hub
is mocked throughout.
"""

from __future__ import annotations

from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from pokemon_stencil.models.model_manager import ModelManager, _is_populated


# ─────────────────────────────────────────────────────────────────────────────
# _is_populated
# ─────────────────────────────────────────────────────────────────────────────

class TestIsPopulated:
    def test_false_when_path_missing(self, tmp_path):
        assert _is_populated(tmp_path / "nonexistent") is False

    def test_false_when_directory_empty(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        assert _is_populated(empty) is False

    def test_true_when_directory_has_files(self, tmp_path):
        d = tmp_path / "model"
        d.mkdir()
        (d / "config.json").write_text("{}", encoding="utf-8")
        assert _is_populated(d) is True


# ─────────────────────────────────────────────────────────────────────────────
# ModelManager.status
# ─────────────────────────────────────────────────────────────────────────────

class TestModelManagerStatus:
    def test_all_missing_returns_false_for_all(self, tmp_path):
        manager = ModelManager(models_root=tmp_path)
        status = manager.status()
        assert all(v is False for v in status.values())
        assert len(status) == 4  # sdxl + openpose + canny + ip_adapter

    def test_populated_directory_returns_true(self, tmp_path):
        # Populate the sdxl directory.
        sdxl_dir = tmp_path / "sdxl"
        sdxl_dir.mkdir()
        (sdxl_dir / "model.safetensors").write_bytes(b"\x00")

        manager = ModelManager(models_root=tmp_path)
        status = manager.status()

        from pokemon_stencil.config import DEFAULT_SDXL_HUB_ID
        assert status[DEFAULT_SDXL_HUB_ID] is True


# ─────────────────────────────────────────────────────────────────────────────
# ModelManager.ensure_directories
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsureDirectories:
    def test_creates_model_directories(self, tmp_path):
        manager = ModelManager(models_root=tmp_path)
        manager.ensure_directories()

        expected_names = {"sdxl", "controlnet_pose", "controlnet_canny", "ip_adapter", "lora"}
        created = {d.name for d in tmp_path.iterdir() if d.is_dir()}
        assert expected_names == created


# ─────────────────────────────────────────────────────────────────────────────
# ModelManager.ensure_models – mocked download
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsureModels:
    def _make_mock_hf_hub(self, tmp_path: Path) -> ModuleType:
        """Return a fake huggingface_hub that creates a dummy file on download."""

        def fake_snapshot_download(repo_id, local_dir, **kwargs):
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "model.safetensors").write_bytes(b"\x00")

        fake_hub = ModuleType("huggingface_hub")
        fake_hub.snapshot_download = fake_snapshot_download
        return fake_hub

    def test_downloads_missing_models(self, tmp_path):
        fake_hub = self._make_mock_hf_hub(tmp_path)
        manager = ModelManager(models_root=tmp_path)

        with patch.dict(__import__("sys").modules, {"huggingface_hub": fake_hub}):
            availability = manager.ensure_models(download_missing=True)

        assert all(v is True for v in availability.values())

    def test_skip_download_when_disabled(self, tmp_path):
        manager = ModelManager(models_root=tmp_path)
        availability = manager.ensure_models(download_missing=False)
        # Nothing was downloaded; all should still be False.
        assert all(v is False for v in availability.values())

    def test_skips_already_present_models(self, tmp_path):
        # Pre-populate the SDXL base model directory.
        sdxl_dir = tmp_path / "sdxl"
        sdxl_dir.mkdir()
        (sdxl_dir / "config.json").write_bytes(b"\x00")

        download_calls = []

        def fake_snapshot_download(repo_id, local_dir, **kwargs):
            download_calls.append(repo_id)
            Path(local_dir).mkdir(parents=True, exist_ok=True)
            (Path(local_dir) / "model.safetensors").write_bytes(b"\x00")

        fake_hub = ModuleType("huggingface_hub")
        fake_hub.snapshot_download = fake_snapshot_download

        manager = ModelManager(models_root=tmp_path)
        with patch.dict(__import__("sys").modules, {"huggingface_hub": fake_hub}):
            manager.ensure_models(download_missing=True)

        # SDXL was already present; only the three remaining models should be downloaded.
        from pokemon_stencil.config import DEFAULT_SDXL_HUB_ID
        assert DEFAULT_SDXL_HUB_ID not in download_calls
        assert len(download_calls) == 3

    def test_raises_import_error_when_hf_hub_missing(self, tmp_path):
        manager = ModelManager(models_root=tmp_path)
        with patch.dict(__import__("sys").modules, {"huggingface_hub": None}):
            with pytest.raises(ImportError, match="huggingface_hub"):
                manager.ensure_models(download_missing=True)
