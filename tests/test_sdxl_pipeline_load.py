"""
tests/test_sdxl_pipeline_load.py — SDXL pipeline loading tests.

Verifies the pipeline loader:
- constructs the correct cache key
- calls from_pretrained with the right arguments
- applies memory optimizations
- routes to the cache on repeated calls
- raises ImportError cleanly when torch/diffusers absent

No real model downloads or GPU are required — all heavy libs are mocked.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from pokemon_stencil.config import GenerationConfig
from pokemon_stencil.models.model_loader import (
    _SDXL_CACHE,
    clear_pipeline_cache,
    is_cached,
    load_sdxl_controlnet_pipeline,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_torch_mock() -> ModuleType:
    m = ModuleType("torch")
    m.float32 = "float32_sentinel"
    m.float16 = "float16_sentinel"
    m.bfloat16 = "bfloat16_sentinel"
    return m


def _make_pipe_mock() -> MagicMock:
    pipe = MagicMock()
    pipe.to.return_value = pipe
    return pipe


def _make_controlnet_mock(pipe_mock: MagicMock) -> ModuleType:
    """Return a fake diffusers module with ControlNetModel + SDXL pipeline."""
    fake = ModuleType("diffusers")

    cn = MagicMock()
    cn.from_pretrained.return_value = MagicMock()
    fake.ControlNetModel = cn

    sdxl = MagicMock()
    sdxl.from_pretrained.return_value = pipe_mock
    fake.StableDiffusionXLControlNetPipeline = sdxl

    # Also expose SD 1.5 classes for legacy paths
    sdp = MagicMock()
    sdp.from_pretrained.return_value = MagicMock()
    fake.StableDiffusionPipeline = sdp
    cn2 = MagicMock()
    cn2.from_pretrained.return_value = MagicMock()
    fake.StableDiffusionControlNetPipeline = cn2

    return fake


@pytest.fixture(autouse=True)
def clear_cache():
    """Ensure pipeline cache is empty before and after every test."""
    clear_pipeline_cache()
    yield
    clear_pipeline_cache()


# ─────────────────────────────────────────────────────────────────────────────
# Basic pipeline loading
# ─────────────────────────────────────────────────────────────────────────────

class TestLoadSdxlPipeline:
    def test_returns_pipeline_object(self) -> None:
        pipe = _make_pipe_mock()
        cfg = GenerationConfig(use_ip_adapter=False, use_controlnet=True)
        fake_torch = _make_torch_mock()
        fake_diffusers = _make_controlnet_mock(pipe)

        with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}):
            result = load_sdxl_controlnet_pipeline(cfg)

        assert result is pipe

    def test_pipeline_moved_to_device(self) -> None:
        pipe = _make_pipe_mock()
        cfg = GenerationConfig(device="cpu", use_ip_adapter=False)
        fake_torch = _make_torch_mock()
        fake_diffusers = _make_controlnet_mock(pipe)

        with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}):
            load_sdxl_controlnet_pipeline(cfg)

        pipe.to.assert_called_once_with("cpu")

    def test_attention_slicing_applied_on_cpu(self) -> None:
        pipe = _make_pipe_mock()
        cfg = GenerationConfig(device="cpu", use_ip_adapter=False)
        fake_torch = _make_torch_mock()
        fake_diffusers = _make_controlnet_mock(pipe)

        with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}):
            load_sdxl_controlnet_pipeline(cfg)

        pipe.enable_attention_slicing.assert_called_once()

    def test_uses_safetensors_kwarg(self) -> None:
        pipe = _make_pipe_mock()
        cfg = GenerationConfig(use_ip_adapter=False)
        fake_torch = _make_torch_mock()
        fake_diffusers = _make_controlnet_mock(pipe)

        with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}):
            load_sdxl_controlnet_pipeline(cfg)

        _, kwargs = fake_diffusers.StableDiffusionXLControlNetPipeline.from_pretrained.call_args
        assert kwargs.get("use_safetensors") is True

    def test_dual_controlnet_loaded(self) -> None:
        """Two ControlNetModel.from_pretrained calls must be made."""
        pipe = _make_pipe_mock()
        cfg = GenerationConfig(use_ip_adapter=False)
        fake_torch = _make_torch_mock()
        fake_diffusers = _make_controlnet_mock(pipe)

        with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}):
            load_sdxl_controlnet_pipeline(cfg)

        assert fake_diffusers.ControlNetModel.from_pretrained.call_count == 2


# ─────────────────────────────────────────────────────────────────────────────
# Caching
# ─────────────────────────────────────────────────────────────────────────────

class TestCaching:
    def test_cached_on_second_call(self) -> None:
        pipe = _make_pipe_mock()
        cfg = GenerationConfig(use_ip_adapter=False)
        fake_torch = _make_torch_mock()
        fake_diffusers = _make_controlnet_mock(pipe)

        with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}):
            first = load_sdxl_controlnet_pipeline(cfg)
            second = load_sdxl_controlnet_pipeline(cfg)

        assert first is second
        assert fake_diffusers.StableDiffusionXLControlNetPipeline.from_pretrained.call_count == 1

    def test_is_cached_true_after_load(self) -> None:
        pipe = _make_pipe_mock()
        cfg = GenerationConfig(use_ip_adapter=False)
        fake_torch = _make_torch_mock()
        fake_diffusers = _make_controlnet_mock(pipe)

        assert not is_cached(cfg)
        with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}):
            load_sdxl_controlnet_pipeline(cfg)

        assert is_cached(cfg)

    def test_clear_cache_resets_is_cached(self) -> None:
        pipe = _make_pipe_mock()
        cfg = GenerationConfig(use_ip_adapter=False)
        fake_torch = _make_torch_mock()
        fake_diffusers = _make_controlnet_mock(pipe)

        with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}):
            load_sdxl_controlnet_pipeline(cfg)

        assert is_cached(cfg)
        clear_pipeline_cache()
        assert not is_cached(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Error handling
# ─────────────────────────────────────────────────────────────────────────────

class TestErrorHandling:
    def test_missing_torch_raises_import_error(self) -> None:
        cfg = GenerationConfig(use_ip_adapter=False)
        with patch.dict(sys.modules, {"torch": None, "diffusers": None}):
            with pytest.raises(ImportError, match="torch"):
                load_sdxl_controlnet_pipeline(cfg)

    def test_fp16_variant_not_used_on_cpu(self) -> None:
        """variant='fp16' must NOT be passed when device is CPU."""
        pipe = _make_pipe_mock()
        cfg = GenerationConfig(device="cpu", torch_dtype="float16", use_ip_adapter=False)
        fake_torch = _make_torch_mock()
        fake_diffusers = _make_controlnet_mock(pipe)

        with patch.dict(sys.modules, {"torch": fake_torch, "diffusers": fake_diffusers}):
            load_sdxl_controlnet_pipeline(cfg)

        _, kwargs = fake_diffusers.StableDiffusionXLControlNetPipeline.from_pretrained.call_args
        assert "variant" not in kwargs
