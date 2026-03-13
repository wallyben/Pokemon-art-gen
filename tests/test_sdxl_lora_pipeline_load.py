"""
tests/test_sdxl_lora_pipeline_load.py — SDXL pipeline + LoRA loading tests.

Verifies that _apply_lora calls validate_lora_sdxl_compatible first,
raises ValueError for SD1.5 LoRAs, and calls load_lora_weights only for
valid SDXL LoRAs.  The validator is mocked to avoid needing safetensors.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

from pokemon_stencil.config import GenerationConfig
from pokemon_stencil.models.model_loader import (
    clear_pipeline_cache,
    load_sdxl_controlnet_pipeline,
)

_REJECT_PREFIX = "Loaded LoRA is not SDXL-compatible. Please provide an SDXL LoRA."


def _make_torch_mock():
    m = ModuleType("torch")
    m.float32 = "float32_sentinel"
    m.float16 = "float16_sentinel"
    m.bfloat16 = "bfloat16_sentinel"
    return m


def _make_pipe_mock():
    pipe = MagicMock()
    pipe.to.return_value = pipe
    return pipe


def _make_diffusers_mock(pipe):
    fake = ModuleType("diffusers")
    cn = MagicMock(); cn.from_pretrained.return_value = MagicMock()
    fake.ControlNetModel = cn
    sdxl = MagicMock(); sdxl.from_pretrained.return_value = pipe
    fake.StableDiffusionXLControlNetPipeline = sdxl
    sdp = MagicMock(); sdp.from_pretrained.return_value = MagicMock()
    fake.StableDiffusionPipeline = sdp
    return fake


@pytest.fixture(autouse=True)
def clear_cache():
    clear_pipeline_cache()
    yield
    clear_pipeline_cache()


class TestValidSdxlLoraAccepted:
    def test_load_lora_weights_called_for_valid_sdxl_lora(self, tmp_path):
        lora = tmp_path / "pokemon_xl.safetensors"; lora.write_bytes(b"\x00" * 64)
        pipe = _make_pipe_mock()
        cfg = GenerationConfig(lora_path=lora, lora_scale=0.8, use_ip_adapter=False)

        with patch(
            "pokemon_stencil.models.model_loader.validate_lora_sdxl_compatible",
            return_value=(True, "SDXL-compatible (mocked)"),
        ), patch.dict(sys.modules, {"torch": _make_torch_mock(), "diffusers": _make_diffusers_mock(pipe)}):
            load_sdxl_controlnet_pipeline(cfg)

        pipe.load_lora_weights.assert_called_once()

    def test_pipeline_returned_after_valid_lora(self, tmp_path):
        lora = tmp_path / "valid.safetensors"; lora.write_bytes(b"\x00" * 64)
        pipe = _make_pipe_mock()
        cfg = GenerationConfig(lora_path=lora, use_ip_adapter=False)

        with patch(
            "pokemon_stencil.models.model_loader.validate_lora_sdxl_compatible",
            return_value=(True, "OK"),
        ), patch.dict(sys.modules, {"torch": _make_torch_mock(), "diffusers": _make_diffusers_mock(pipe)}):
            result = load_sdxl_controlnet_pipeline(cfg)

        assert result is pipe

    def test_lora_path_included_in_cache_key(self, tmp_path):
        lora_a = tmp_path / "lora_a.safetensors"; lora_a.write_bytes(b"\x00" * 64)
        lora_b = tmp_path / "lora_b.safetensors"; lora_b.write_bytes(b"\x00" * 64)
        pipe_a, pipe_b = _make_pipe_mock(), _make_pipe_mock()

        fake_diffusers = ModuleType("diffusers")
        cn = MagicMock(); cn.from_pretrained.return_value = MagicMock()
        fake_diffusers.ControlNetModel = cn
        sdxl = MagicMock(); sdxl.from_pretrained.side_effect = [pipe_a, pipe_b]
        fake_diffusers.StableDiffusionXLControlNetPipeline = sdxl

        cfg_a = GenerationConfig(lora_path=lora_a, use_ip_adapter=False)
        cfg_b = GenerationConfig(lora_path=lora_b, use_ip_adapter=False)

        with patch(
            "pokemon_stencil.models.model_loader.validate_lora_sdxl_compatible",
            return_value=(True, "OK"),
        ), patch.dict(sys.modules, {"torch": _make_torch_mock(), "diffusers": fake_diffusers}):
            r_a = load_sdxl_controlnet_pipeline(cfg_a)
            r_b = load_sdxl_controlnet_pipeline(cfg_b)

        assert r_a is pipe_a
        assert r_b is pipe_b


class TestSd15LoraRejected:
    def test_sd15_lora_raises_value_error(self, tmp_path):
        lora = tmp_path / "sd15.safetensors"; lora.write_bytes(b"\x00" * 64)
        pipe = _make_pipe_mock()
        cfg = GenerationConfig(lora_path=lora, use_ip_adapter=False)

        with patch(
            "pokemon_stencil.models.model_loader.validate_lora_sdxl_compatible",
            return_value=(False, _REJECT_PREFIX + " (cross-attn dim=768)"),
        ), patch.dict(sys.modules, {"torch": _make_torch_mock(), "diffusers": _make_diffusers_mock(pipe)}):
            with pytest.raises(ValueError) as exc_info:
                load_sdxl_controlnet_pipeline(cfg)

        assert _REJECT_PREFIX in str(exc_info.value)

    def test_sd15_lora_does_not_call_load_lora_weights(self, tmp_path):
        lora = tmp_path / "sd15.safetensors"; lora.write_bytes(b"\x00" * 64)
        pipe = _make_pipe_mock()
        cfg = GenerationConfig(lora_path=lora, use_ip_adapter=False)

        with patch(
            "pokemon_stencil.models.model_loader.validate_lora_sdxl_compatible",
            return_value=(False, _REJECT_PREFIX),
        ), patch.dict(sys.modules, {"torch": _make_torch_mock(), "diffusers": _make_diffusers_mock(pipe)}):
            with pytest.raises(ValueError):
                load_sdxl_controlnet_pipeline(cfg)

        pipe.load_lora_weights.assert_not_called()

    def test_error_message_contains_canonical_text(self, tmp_path):
        lora = tmp_path / "sd15.safetensors"; lora.write_bytes(b"\x00" * 64)
        cfg = GenerationConfig(lora_path=lora, use_ip_adapter=False)

        with patch(
            "pokemon_stencil.models.model_loader.validate_lora_sdxl_compatible",
            return_value=(False, _REJECT_PREFIX + " extra context"),
        ), patch.dict(sys.modules, {"torch": _make_torch_mock(), "diffusers": _make_diffusers_mock(_make_pipe_mock())}):
            with pytest.raises(ValueError) as exc_info:
                load_sdxl_controlnet_pipeline(cfg)

        assert _REJECT_PREFIX in str(exc_info.value)


class TestMissingLoraFile:
    def test_missing_lora_raises_value_error(self, tmp_path):
        cfg = GenerationConfig(
            lora_path=tmp_path / "nonexistent.safetensors",
            use_ip_adapter=False,
        )
        # Validator returns False for missing file (no mock needed — real validator handles it)
        with patch.dict(sys.modules, {"torch": _make_torch_mock(), "diffusers": _make_diffusers_mock(_make_pipe_mock())}):
            with pytest.raises(ValueError, match="not found"):
                load_sdxl_controlnet_pipeline(cfg)
