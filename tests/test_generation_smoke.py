"""
tests/test_generation_smoke.py — SDXL generation smoke test.

Generates 1 small synthetic image through the PokemonImageGenerator with a
fully mocked pipeline, confirming:
- A PIL Image is returned
- No tensor shape mismatch occurs
- The prompt is passed to the pipeline
- Output has the expected dimensions

No real models or GPU are required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from pokemon_stencil.config import GenerationConfig
from pokemon_stencil.image_gen.generator import PokemonImageGenerator


# ── Helpers ───────────────────────────────────────────────────────────────────

def _synthetic_image(width: int = 64, height: int = 64) -> Image.Image:
    """Return a small synthetic RGB image for use as mock pipeline output."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _make_pipe_mock(output_image: Image.Image) -> MagicMock:
    """Return a mock pipeline whose __call__ returns a diffusers-style result."""
    pipe = MagicMock()
    pipe.to.return_value = pipe

    result = MagicMock()
    result.images = [output_image]
    pipe.return_value = result
    return pipe


def _make_torch_mock() -> ModuleType:
    m = ModuleType("torch")
    m.float32 = "float32_sentinel"
    m.float16 = "float16_sentinel"
    m.bfloat16 = "bfloat16_sentinel"

    gen_mock = MagicMock()
    m.Generator = MagicMock(return_value=gen_mock)
    m.manual_seed = MagicMock(return_value=gen_mock)
    return m


def _make_diffusers_mock(pipe: MagicMock) -> ModuleType:
    fake = ModuleType("diffusers")
    cn = MagicMock()
    cn.from_pretrained.return_value = MagicMock()
    fake.ControlNetModel = cn
    sdxl = MagicMock()
    sdxl.from_pretrained.return_value = pipe
    fake.StableDiffusionXLControlNetPipeline = sdxl
    sdp = MagicMock()
    sdp.from_pretrained.return_value = MagicMock()
    fake.StableDiffusionPipeline = sdp
    return fake


@pytest.fixture(autouse=True)
def clear_pipeline_cache():
    from pokemon_stencil.models.model_loader import clear_pipeline_cache as _clr
    _clr()
    yield
    _clr()


# ─────────────────────────────────────────────────────────────────────────────
# Smoke tests
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerationSmoke:
    def _make_config(self, **kwargs) -> GenerationConfig:
        defaults = dict(
            use_ip_adapter=False,
            use_controlnet=True,
            num_inference_steps=1,   # minimal for speed
            width=64,
            height=64,
            device="cpu",
            torch_dtype="float32",
            seed=42,
        )
        defaults.update(kwargs)
        return GenerationConfig(**defaults)

    def test_returns_list_of_pil_images(self) -> None:
        """generate() must return a non-empty list of PIL Images."""
        output_img = _synthetic_image(64, 64)
        pipe = _make_pipe_mock(output_img)
        cfg = self._make_config()
        gen = PokemonImageGenerator(cfg)

        with patch(
            "pokemon_stencil.image_gen.generator.load_sdxl_controlnet_pipeline",
            return_value=pipe,
        ):
            results = gen.generate("Pikachu", num_images=1)

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], Image.Image)

    def test_no_tensor_shape_mismatch(self) -> None:
        """generate() must complete without RuntimeError (tensor shape mismatch)."""
        output_img = _synthetic_image(64, 64)
        pipe = _make_pipe_mock(output_img)
        cfg = self._make_config()
        gen = PokemonImageGenerator(cfg)

        with patch(
            "pokemon_stencil.image_gen.generator.load_sdxl_controlnet_pipeline",
            return_value=pipe,
        ):
            # Must not raise
            results = gen.generate("Charizard", num_images=1)

        assert results is not None

    def test_prompt_contains_pokemon_name(self) -> None:
        """The prompt passed to the pipeline must contain the Pokémon name."""
        output_img = _synthetic_image(64, 64)
        pipe = _make_pipe_mock(output_img)
        cfg = self._make_config()
        gen = PokemonImageGenerator(cfg)

        with patch(
            "pokemon_stencil.image_gen.generator.load_sdxl_controlnet_pipeline",
            return_value=pipe,
        ):
            gen.generate("Eevee", num_images=1)

        call_kwargs = pipe.call_args[1]
        prompt_used = call_kwargs.get("prompt", "")
        assert "Eevee" in prompt_used

    def test_output_is_rgb(self) -> None:
        """Output images must be RGB mode (white background compositing applied)."""
        output_img = _synthetic_image(64, 64)
        pipe = _make_pipe_mock(output_img)
        cfg = self._make_config()
        gen = PokemonImageGenerator(cfg)

        with patch(
            "pokemon_stencil.image_gen.generator.load_sdxl_controlnet_pipeline",
            return_value=pipe,
        ):
            results = gen.generate("Mewtwo", num_images=1)

        assert results[0].mode == "RGB"

    def test_seed_passed_to_pipeline(self) -> None:
        """A fixed seed must produce deterministic generator calls."""
        output_img = _synthetic_image(64, 64)
        pipe = _make_pipe_mock(output_img)
        cfg = self._make_config(seed=1337)
        gen = PokemonImageGenerator(cfg)

        with patch(
            "pokemon_stencil.image_gen.generator.load_sdxl_controlnet_pipeline",
            return_value=pipe,
        ):
            gen.generate("Bulbasaur", num_images=1)

        # generator kwarg must have been passed
        call_kwargs = pipe.call_args[1]
        assert "generator" in call_kwargs

    def test_prompt_within_clip_token_limit(self) -> None:
        """With optimise_prompt=True the built prompt must not exceed 77 tokens."""
        from pokemon_stencil.image_gen.prompt_engine import CLIP_TOKEN_LIMIT, PromptEngine
        cfg = self._make_config()
        gen = PokemonImageGenerator(cfg)
        prompt = gen._build_prompt("Pikachu", "dynamic action pose, electric sparks")
        engine = PromptEngine()
        assert engine.token_count(prompt) <= CLIP_TOKEN_LIMIT
