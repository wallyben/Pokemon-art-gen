"""
Tests for pokemon_stencil.models.model_loader.

All tests mock ``diffusers.StableDiffusionPipeline.from_pretrained`` so
no network access or GPU is required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, call, patch

import pytest

from pokemon_stencil.config import GenerationConfig
from pokemon_stencil.models import model_loader
from pokemon_stencil.models.model_loader import (
    _PIPELINE_CACHE,
    _resolve_model_source,
    clear_pipeline_cache,
    is_cached,
    load_sd_pipeline,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_pipe() -> MagicMock:
    """Return a mock StableDiffusionPipeline with the required API surface."""
    pipe = MagicMock()
    pipe.to.return_value = pipe          # pipe.to(device) returns self
    pipe.enable_attention_slicing = MagicMock()
    return pipe


def _make_diffusers_mock(pipe: MagicMock) -> ModuleType:
    """
    Build a fake ``diffusers`` module whose ``StableDiffusionPipeline``
    ``from_pretrained`` returns *pipe*.
    """
    fake_diffusers = ModuleType("diffusers")
    fake_sdp = MagicMock()
    fake_sdp.from_pretrained.return_value = pipe
    fake_diffusers.StableDiffusionPipeline = fake_sdp
    return fake_diffusers


def _make_torch_mock() -> ModuleType:
    """Build a minimal fake ``torch`` module."""
    fake_torch = ModuleType("torch")
    fake_torch.float32 = "float32_sentinel"
    fake_torch.float16 = "float16_sentinel"
    fake_torch.bfloat16 = "bfloat16_sentinel"
    return fake_torch


@pytest.fixture(autouse=True)
def clear_cache():
    """Ensure cache is empty before and after every test."""
    clear_pipeline_cache()
    yield
    clear_pipeline_cache()


# ─────────────────────────────────────────────────────────────────────────────
# _resolve_model_source
# ─────────────────────────────────────────────────────────────────────────────

class TestResolveModelSource:
    def test_returns_local_path_when_exists_and_nonempty(self, tmp_path):
        # Create a non-empty local model directory
        local = tmp_path / "sd15"
        local.mkdir()
        (local / "model.safetensors").write_bytes(b"\x00")  # non-empty dir

        cfg = GenerationConfig(model_local_path=local)
        assert _resolve_model_source(cfg) == local

    def test_returns_hub_id_when_local_path_absent(self, tmp_path):
        cfg = GenerationConfig(
            model_local_path=tmp_path / "nonexistent",
            model_hub_id="runwayml/stable-diffusion-v1-5",
        )
        result = _resolve_model_source(cfg)
        assert result == "runwayml/stable-diffusion-v1-5"

    def test_returns_hub_id_when_local_path_empty_dir(self, tmp_path):
        empty_local = tmp_path / "empty_sd"
        empty_local.mkdir()
        # Directory exists but is empty → fall through to Hub

        cfg = GenerationConfig(
            model_local_path=empty_local,
            model_hub_id="runwayml/stable-diffusion-v1-5",
        )
        result = _resolve_model_source(cfg)
        assert result == "runwayml/stable-diffusion-v1-5"

    def test_custom_hub_id_used(self, tmp_path):
        cfg = GenerationConfig(
            model_local_path=tmp_path / "none",
            model_hub_id="custom-org/custom-model",
        )
        result = _resolve_model_source(cfg)
        assert result == "custom-org/custom-model"


# ─────────────────────────────────────────────────────────────────────────────
# load_sd_pipeline – mocked diffusers / torch
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadSdPipeline:
    def _patch_imports(self, pipe: MagicMock):
        """Context manager that patches torch and diffusers imports."""
        fake_torch = _make_torch_mock()
        fake_diffusers = _make_diffusers_mock(pipe)
        return patch.dict(
            sys.modules,
            {"torch": fake_torch, "diffusers": fake_diffusers},
        )

    def test_returns_pipeline_instance(self):
        pipe = _make_mock_pipe()
        cfg = GenerationConfig()

        with self._patch_imports(pipe):
            result = load_sd_pipeline(cfg)

        assert result is pipe

    def test_attention_slicing_enabled(self):
        pipe = _make_mock_pipe()
        cfg = GenerationConfig()

        with self._patch_imports(pipe):
            load_sd_pipeline(cfg)

        pipe.enable_attention_slicing.assert_called_once()

    def test_moved_to_cpu(self):
        pipe = _make_mock_pipe()
        cfg = GenerationConfig(device="cpu")

        with self._patch_imports(pipe):
            load_sd_pipeline(cfg)

        pipe.to.assert_called_once_with("cpu")

    def test_safety_checker_disabled(self):
        pipe = _make_mock_pipe()
        fake_torch = _make_torch_mock()
        fake_diffusers = _make_diffusers_mock(pipe)

        cfg = GenerationConfig(model_hub_id="runwayml/stable-diffusion-v1-5")
        with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}):
            load_sd_pipeline(cfg)

        _, kwargs = fake_diffusers.StableDiffusionPipeline.from_pretrained.call_args
        assert kwargs.get("safety_checker") is None
        assert kwargs.get("requires_safety_checker") is False

    def test_cached_on_second_call(self):
        pipe = _make_mock_pipe()
        cfg = GenerationConfig()
        fake_torch = _make_torch_mock()
        fake_diffusers = _make_diffusers_mock(pipe)

        with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}):
            first = load_sd_pipeline(cfg)
            second = load_sd_pipeline(cfg)

        assert first is second
        # from_pretrained called only once despite two load calls
        assert fake_diffusers.StableDiffusionPipeline.from_pretrained.call_count == 1

    def test_different_configs_load_independently(self, tmp_path):
        pipe_a = _make_mock_pipe()
        pipe_b = _make_mock_pipe()

        fake_torch = _make_torch_mock()
        fake_diffusers = ModuleType("diffusers")
        fake_sdp = MagicMock()
        fake_sdp.from_pretrained = MagicMock(side_effect=[pipe_a, pipe_b])
        fake_diffusers.StableDiffusionPipeline = fake_sdp

        # Use a nonexistent local path so resolution falls through to hub IDs.
        absent = tmp_path / "no_local"
        cfg_a = GenerationConfig(model_local_path=absent, model_hub_id="org/model-a")
        cfg_b = GenerationConfig(model_local_path=absent, model_hub_id="org/model-b")

        with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}):
            result_a = load_sd_pipeline(cfg_a)
            result_b = load_sd_pipeline(cfg_b)

        assert result_a is pipe_a
        assert result_b is pipe_b

    def test_missing_torch_raises_import_error(self):
        cfg = GenerationConfig()
        # Remove torch and diffusers from sys.modules so the import inside
        # _load_from_source fails naturally.
        patched = {k: None for k in ("torch", "diffusers") if k in sys.modules}
        with patch.dict(sys.modules, {"torch": None, "diffusers": None}):
            with pytest.raises(ImportError, match="torch and diffusers"):
                load_sd_pipeline(cfg)

    def test_loads_from_local_path_when_present(self, tmp_path):
        local = tmp_path / "sd15"
        local.mkdir()
        (local / "config.json").write_text("{}", encoding="utf-8")  # non-empty

        pipe = _make_mock_pipe()
        fake_torch = _make_torch_mock()
        fake_diffusers = _make_diffusers_mock(pipe)

        cfg = GenerationConfig(model_local_path=local)
        with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}):
            load_sd_pipeline(cfg)

        # Verify from_pretrained was called with the local Path, not Hub ID
        args, _ = fake_diffusers.StableDiffusionPipeline.from_pretrained.call_args
        assert args[0] == local


# ─────────────────────────────────────────────────────────────────────────────
# clear_pipeline_cache / is_cached
# ─────────────────────────────────────────────────────────────────────────────

class TestCacheHelpers:
    def _patch_imports(self, pipe: MagicMock):
        fake_torch = _make_torch_mock()
        fake_diffusers = _make_diffusers_mock(pipe)
        return patch.dict(
            sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}
        )

    def test_is_cached_false_before_load(self):
        cfg = GenerationConfig()
        assert is_cached(cfg) is False

    def test_is_cached_true_after_load(self):
        pipe = _make_mock_pipe()
        cfg = GenerationConfig()
        with self._patch_imports(pipe):
            load_sd_pipeline(cfg)
        assert is_cached(cfg) is True

    def test_is_cached_false_after_clear(self):
        pipe = _make_mock_pipe()
        cfg = GenerationConfig()
        with self._patch_imports(pipe):
            load_sd_pipeline(cfg)
        clear_pipeline_cache()
        assert is_cached(cfg) is False

    def test_clear_removes_all_entries(self, tmp_path):
        pipe_a = _make_mock_pipe()
        pipe_b = _make_mock_pipe()

        # Use a nonexistent local path so each config resolves to its hub ID.
        absent = tmp_path / "no_local"
        cfg_a = GenerationConfig(model_local_path=absent, model_hub_id="org/model-x")
        cfg_b = GenerationConfig(model_local_path=absent, model_hub_id="org/model-y")

        fake_torch = _make_torch_mock()
        fake_diffusers = ModuleType("diffusers")
        fake_sdp = MagicMock()
        fake_sdp.from_pretrained = MagicMock(side_effect=[pipe_a, pipe_b])
        fake_diffusers.StableDiffusionPipeline = fake_sdp

        with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}):
            load_sd_pipeline(cfg_a)
            load_sd_pipeline(cfg_b)

        assert len(_PIPELINE_CACHE) == 2
        clear_pipeline_cache()
        assert len(_PIPELINE_CACHE) == 0
