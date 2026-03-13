"""
tests/test_lora_validation.py — LoRA compatibility validation tests.

All safetensors I/O is mocked — no real safetensors installation required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

_SDXL_REJECT_MSG = "Loaded LoRA is not SDXL-compatible. Please provide an SDXL LoRA."


class _FakeTensor:
    def __init__(self, shape):
        self.shape = shape


class _FakeSafeOpen:
    def __init__(self, metadata, tensors):
        self._metadata = metadata or {}
        self._tensors = tensors

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass

    def metadata(self):
        return self._metadata

    def keys(self):
        return list(self._tensors.keys())

    def get_tensor(self, key):
        return _FakeTensor(self._tensors[key])


def _patch_st(metadata, tensors):
    fake_st = ModuleType("safetensors")
    fake_st.safe_open = MagicMock(return_value=_FakeSafeOpen(metadata, tensors))
    return patch.dict(sys.modules, {"safetensors": fake_st})


from pokemon_stencil.models.lora_validator import validate_lora_sdxl_compatible


class TestMissingAndInvalid:
    def test_missing_file(self, tmp_path):
        ok, msg = validate_lora_sdxl_compatible(tmp_path / "x.safetensors")
        assert ok is False
        assert "not found" in msg.lower()

    def test_wrong_extension(self, tmp_path):
        f = tmp_path / "lora.bin"; f.write_bytes(b"\x00")
        ok, msg = validate_lora_sdxl_compatible(f)
        assert ok is False and ".safetensors" in msg

    def test_safetensors_not_installed(self, tmp_path):
        f = tmp_path / "lora.safetensors"; f.write_bytes(b"\x00" * 64)
        with patch.dict(sys.modules, {"safetensors": None}):
            ok, msg = validate_lora_sdxl_compatible(f)
        assert ok is False


class TestMetadataDetection:
    def _f(self, tmp_path, name="lora.safetensors"):
        f = tmp_path / name; f.write_bytes(b"\x00" * 64); return f

    def test_sdxl_metadata_accepted(self, tmp_path):
        f = self._f(tmp_path)
        with _patch_st({"ss_base_model_version": "sd_xl_base_1.0"}, {"k": [4, 128]}):
            ok, _ = validate_lora_sdxl_compatible(f)
        assert ok is True

    def test_sd15_metadata_rejected(self, tmp_path):
        f = self._f(tmp_path)
        with _patch_st({"ss_base_model_version": "sd-1-5"}, {"k": [4, 128]}):
            ok, msg = validate_lora_sdxl_compatible(f)
        assert ok is False and _SDXL_REJECT_MSG in msg

    def test_sdxl_keywords(self, tmp_path):
        for ver in ("sdxl", "xl_base_1.0", "stable_diffusion_xl"):
            f = self._f(tmp_path, f"{ver}.safetensors")
            with _patch_st({"ss_base_model_version": ver}, {"k": [4, 128]}):
                ok, _ = validate_lora_sdxl_compatible(f)
            assert ok is True, f"Failed for base_ver='{ver}'"

    def test_sd15_keywords(self, tmp_path):
        for ver in ("v1-5", "sd1", "sd15", "stable_diffusion_v1"):
            f = self._f(tmp_path, f"{ver}.safetensors")
            with _patch_st({"ss_base_model_version": ver}, {"k": [4, 128]}):
                ok, msg = validate_lora_sdxl_compatible(f)
            assert ok is False and _SDXL_REJECT_MSG in msg, f"Failed for '{ver}'"

    def test_no_metadata_falls_to_shape(self, tmp_path):
        f = self._f(tmp_path)
        key = "lora_unet_down_blocks_0_attn2_to_k.lora_down.weight"
        with _patch_st({}, {key: [4, 2048]}):
            ok, _ = validate_lora_sdxl_compatible(f)
        assert ok is True


class TestCrossAttnDimDetection:
    def _f(self, tmp_path, name="lora.safetensors"):
        f = tmp_path / name; f.write_bytes(b"\x00" * 64); return f

    def test_sdxl_2048_accepted(self, tmp_path):
        f = self._f(tmp_path)
        key = "lora_unet_down_blocks_0_attn2_to_k.lora_down.weight"
        with _patch_st(None, {key: [4, 2048]}):
            ok, _ = validate_lora_sdxl_compatible(f)
        assert ok is True

    def test_sd15_768_rejected(self, tmp_path):
        f = self._f(tmp_path)
        key = "lora_unet_down_blocks_0_attn2_to_k.lora_down.weight"
        with _patch_st(None, {key: [4, 768]}):
            ok, msg = validate_lora_sdxl_compatible(f)
        assert ok is False and _SDXL_REJECT_MSG in msg

    def test_to_v_also_detected(self, tmp_path):
        f = self._f(tmp_path)
        key = "lora_unet_down_blocks_0_attn2_to_v.lora_down.weight"
        with _patch_st(None, {key: [4, 2048]}):
            ok, _ = validate_lora_sdxl_compatible(f)
        assert ok is True


class TestMaxDimHeuristic:
    def _f(self, tmp_path, name="lora.safetensors"):
        f = tmp_path / name; f.write_bytes(b"\x00" * 64); return f

    def test_large_dim_accepted(self, tmp_path):
        f = self._f(tmp_path)
        with _patch_st(None, {"lora_generic.lora_down.weight": [4, 2048]}):
            ok, _ = validate_lora_sdxl_compatible(f)
        assert ok is True

    def test_768_dim_rejected(self, tmp_path):
        f = self._f(tmp_path)
        with _patch_st(None, {"lora_generic.lora_down.weight": [4, 768]}):
            ok, msg = validate_lora_sdxl_compatible(f)
        assert ok is False and _SDXL_REJECT_MSG in msg

    def test_1280_dim_rejected(self, tmp_path):
        """1280 is the SD1.5 UNet max channel dim — must be rejected."""
        f = self._f(tmp_path)
        with _patch_st(None, {"lora_generic.lora_down.weight": [4, 1280]}):
            ok, msg = validate_lora_sdxl_compatible(f)
        assert ok is False and _SDXL_REJECT_MSG in msg

    def test_ambiguous_range_rejected(self, tmp_path):
        """Dims in 1280-2048 range must be rejected conservatively."""
        f = self._f(tmp_path)
        for dim in (1281, 1500, 1900, 2047):
            with _patch_st(None, {"lora_generic.lora_down.weight": [4, dim]}):
                ok, msg = validate_lora_sdxl_compatible(f)
            assert ok is False and _SDXL_REJECT_MSG in msg, f"Expected rejection for dim={dim}"


class TestMetadataPriority:
    def _f(self, tmp_path):
        f = tmp_path / "lora.safetensors"; f.write_bytes(b"\x00" * 64); return f

    def test_sdxl_meta_overrides_sd15_shape(self, tmp_path):
        f = self._f(tmp_path)
        with _patch_st({"ss_base_model_version": "sd_xl_base_1.0"}, {"k": [4, 768]}):
            ok, _ = validate_lora_sdxl_compatible(f)
        assert ok is True

    def test_sd15_meta_overrides_sdxl_shape(self, tmp_path):
        f = self._f(tmp_path)
        with _patch_st({"ss_base_model_version": "v1-5"}, {"k": [4, 2048]}):
            ok, msg = validate_lora_sdxl_compatible(f)
        assert ok is False and _SDXL_REJECT_MSG in msg
