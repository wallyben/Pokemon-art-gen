"""
tests/test_provider_config.py — ProviderConfig and provider abstraction tests.

Tests the configuration layer and provider availability logic.
No real API calls, no GPU.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pokemon_stencil.config import PipelineConfig, ProviderConfig
from pokemon_stencil.generation.base import (
    AbstractGenerationProvider,
    GenerationRequest,
    GenerationResult,
)


class TestProviderConfig:
    def test_default_provider_is_auto(self):
        cfg = ProviderConfig()
        assert cfg.provider == "auto"

    def test_default_n_candidates(self):
        cfg = ProviderConfig()
        assert cfg.n_candidates == 8
        assert cfg.n_top == 3

    def test_default_steps(self):
        cfg = ProviderConfig()
        assert cfg.num_steps == 28

    def test_default_guidance_scale(self):
        cfg = ProviderConfig()
        assert cfg.guidance_scale == 3.5

    def test_default_reference_strength(self):
        cfg = ProviderConfig()
        assert cfg.reference_strength == 0.80

    def test_pipeline_config_includes_provider(self):
        cfg = PipelineConfig()
        assert hasattr(cfg, "provider")
        assert isinstance(cfg.provider, ProviderConfig)

    def test_fal_model_default(self):
        cfg = ProviderConfig()
        assert "flux" in cfg.fal_model.lower()

    def test_local_lora_path_default_none(self):
        cfg = ProviderConfig()
        assert cfg.local_lora_path is None


class TestGenerationRequest:
    def test_default_values(self):
        req = GenerationRequest(prompt="test")
        assert req.width == 1024
        assert req.height == 1024
        assert req.num_images == 1
        assert req.seed is None
        assert req.reference_images == []

    def test_with_references(self, tmp_path):
        ref = tmp_path / "ref.png"
        ref.write_bytes(b"\x00")
        req = GenerationRequest(
            prompt="test",
            reference_images=[ref],
            seed=42,
        )
        assert len(req.reference_images) == 1
        assert req.seed == 42

    def test_extra_dict(self):
        req = GenerationRequest(
            prompt="test",
            extra={"pokemon_name": "Pikachu", "custom_key": "value"},
        )
        assert req.extra["pokemon_name"] == "Pikachu"


class TestGenerationResult:
    def test_ok_true_when_images_present(self, tmp_path):
        img = tmp_path / "test.png"
        img.write_bytes(b"\x00")
        result = GenerationResult(images=[img], provider="test")
        assert result.ok is True

    def test_ok_false_when_empty(self):
        result = GenerationResult(images=[], provider="test")
        assert result.ok is False

    def test_metadata_default_empty(self):
        result = GenerationResult(images=[], provider="test")
        assert result.metadata == {}


class TestFalProviderAvailability:
    def test_unavailable_without_fal_key(self):
        from pokemon_stencil.generation.fal_provider import FalProvider
        provider = FalProvider()
        env_key = os.environ.pop("FAL_KEY", None)
        try:
            assert not provider.is_available()
        finally:
            if env_key:
                os.environ["FAL_KEY"] = env_key

    def test_unavailable_without_fal_client(self):
        from pokemon_stencil.generation.fal_provider import FalProvider
        provider = FalProvider()
        with patch.dict(os.environ, {"FAL_KEY": "test-key"}):
            # Hide fal_client
            saved = sys.modules.pop("fal_client", None)
            sys.modules["fal_client"] = None  # type: ignore[assignment]
            try:
                assert not provider.is_available()
            finally:
                if saved is None:
                    sys.modules.pop("fal_client", None)
                else:
                    sys.modules["fal_client"] = saved

    def test_name_is_fal(self):
        from pokemon_stencil.generation.fal_provider import FalProvider
        assert FalProvider().name == "fal"

    def test_returns_empty_result_without_key(self, tmp_path):
        """FalProvider.generate() must NEVER raise — returns empty result instead."""
        from pokemon_stencil.generation.fal_provider import FalProvider
        provider = FalProvider()
        env_key = os.environ.pop("FAL_KEY", None)
        try:
            req = GenerationRequest(prompt="test")
            result = provider.generate(req, tmp_path)
            assert isinstance(result, GenerationResult)
            assert result.images == []
        finally:
            if env_key:
                os.environ["FAL_KEY"] = env_key


class TestLocalSDXLProviderAvailability:
    def test_name_is_local_sdxl(self):
        from pokemon_stencil.generation.local_provider import LocalSDXLProvider
        assert LocalSDXLProvider().name == "local_sdxl"

    def test_availability_depends_on_torch(self):
        from pokemon_stencil.generation.local_provider import LocalSDXLProvider
        provider = LocalSDXLProvider()
        # is_available() should return a bool without crashing
        result = provider.is_available()
        assert isinstance(result, bool)

    def test_generate_returns_result_on_import_error(self, tmp_path):
        """LocalSDXLProvider.generate() must NEVER raise."""
        from pokemon_stencil.generation.local_provider import LocalSDXLProvider
        provider = LocalSDXLProvider()
        with patch.dict(sys.modules, {"pokemon_stencil.image_gen.generator": None}):
            req = GenerationRequest(prompt="test")
            result = provider.generate(req, tmp_path)
            # Should return a result (possibly empty) without raising
            assert isinstance(result, GenerationResult)


class TestProviderFactory:
    def test_get_provider_fal_raises_without_key(self):
        from pokemon_stencil.generation.provider_factory import get_provider
        env_key = os.environ.pop("FAL_KEY", None)
        try:
            with pytest.raises(RuntimeError, match="FAL_KEY"):
                get_provider("fal")
        finally:
            if env_key:
                os.environ["FAL_KEY"] = env_key

    def test_get_provider_local_raises_without_torch(self):
        """If torch is not available, get_provider('local') should raise."""
        from pokemon_stencil.generation.local_provider import LocalSDXLProvider
        with patch.object(LocalSDXLProvider, "is_available", return_value=False):
            from pokemon_stencil.generation.provider_factory import get_provider
            with pytest.raises(RuntimeError):
                get_provider("local")

    def test_get_provider_auto_returns_fal_when_available(self):
        from pokemon_stencil.generation.fal_provider import FalProvider
        from pokemon_stencil.generation.provider_factory import get_provider
        with patch.object(FalProvider, "is_available", return_value=True):
            provider = get_provider("auto")
            assert provider.name == "fal"

    def test_get_provider_auto_falls_back_to_local(self):
        from pokemon_stencil.generation.fal_provider import FalProvider
        from pokemon_stencil.generation.local_provider import LocalSDXLProvider
        from pokemon_stencil.generation.provider_factory import get_provider
        with patch.object(FalProvider, "is_available", return_value=False), \
             patch.object(LocalSDXLProvider, "is_available", return_value=True):
            provider = get_provider("auto")
            assert provider.name == "local_sdxl"

    def test_get_provider_auto_raises_when_nothing_available(self):
        from pokemon_stencil.generation.fal_provider import FalProvider
        from pokemon_stencil.generation.local_provider import LocalSDXLProvider
        from pokemon_stencil.generation.provider_factory import get_provider
        with patch.object(FalProvider, "is_available", return_value=False), \
             patch.object(LocalSDXLProvider, "is_available", return_value=False):
            with pytest.raises(RuntimeError):
                get_provider("auto")
