"""
Tests for pokemon_stencil.image_gen.generator.PokemonImageGenerator.

The Stable Diffusion pipeline is always mocked; no GPU or network access
is required.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from PIL import Image

from pokemon_stencil.config import GenerationConfig
from pokemon_stencil.image_gen.generator import (
    PokemonImageGenerator,
    _NEGATIVE_PROMPT,
    _PROMPT_TEMPLATE,
)
from pokemon_stencil.models.model_loader import clear_pipeline_cache


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_pipe_mock(output_image: Image.Image) -> MagicMock:
    """Mock pipeline that returns *output_image* on every call."""
    result_mock = MagicMock()
    result_mock.images = [output_image]
    pipe = MagicMock(return_value=result_mock)
    pipe.to.return_value = pipe
    pipe.enable_attention_slicing = MagicMock()
    return pipe


def _synthetic_image(size=(64, 64), colour=(200, 150, 100)) -> Image.Image:
    return Image.new("RGB", size, colour)


@pytest.fixture(autouse=True)
def flush_cache():
    """Always start each test with an empty pipeline cache."""
    clear_pipeline_cache()
    yield
    clear_pipeline_cache()


@pytest.fixture()
def default_config() -> GenerationConfig:
    return GenerationConfig(
        num_inference_steps=1,   # fast for tests
        width=64,
        height=64,
        seed=42,
    )


def _patch_loader(pipe_mock: MagicMock):
    """Patch load_sdxl_controlnet_pipeline to return *pipe_mock* directly."""
    return patch(
        "pokemon_stencil.image_gen.generator.load_sdxl_controlnet_pipeline",
        return_value=pipe_mock,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Prompt construction
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildPrompt:
    def test_includes_pokemon_name(self, default_config):
        gen = PokemonImageGenerator(default_config)
        prompt = gen._build_prompt("Pikachu", "")
        assert "Pikachu" in prompt

    def test_uses_prompt_template(self, default_config):
        gen = PokemonImageGenerator(default_config)
        prompt = gen._build_prompt("Gengar", "")
        # Template says "A dynamic illustration of {pokemon_name}"
        assert "Gengar" in prompt
        assert "bold outlines" in prompt.lower() or "stencil" in prompt.lower()

    def test_appends_extras(self, default_config):
        gen = PokemonImageGenerator(default_config)
        prompt = gen._build_prompt("Eevee", "brown fluffy tail")
        assert "brown fluffy tail" in prompt

    def test_appends_style_suffix(self, default_config):
        gen = PokemonImageGenerator(default_config)
        prompt = gen._build_prompt("Bulbasaur", "")
        assert default_config.style_suffix in prompt

    def test_no_extras_no_trailing_comma(self, default_config):
        gen = PokemonImageGenerator(default_config)
        prompt = gen._build_prompt("Mewtwo", "")
        assert not prompt.endswith(",")

    def test_different_names_produce_different_prompts(self, default_config):
        gen = PokemonImageGenerator(default_config)
        a = gen._build_prompt("Pikachu", "")
        b = gen._build_prompt("Charizard", "")
        assert a != b


# ─────────────────────────────────────────────────────────────────────────────
# White background compositing
# ─────────────────────────────────────────────────────────────────────────────

class TestEnsureWhiteBackground:
    def test_rgb_passthrough_converts_to_rgb(self):
        img = Image.new("RGB", (32, 32), (100, 150, 200))
        result = PokemonImageGenerator._ensure_white_background(img)
        assert result.mode == "RGB"

    def test_rgba_composited_onto_white(self):
        # Fully transparent RGBA → result should be white
        img = Image.new("RGBA", (32, 32), (0, 0, 255, 0))
        result = PokemonImageGenerator._ensure_white_background(img)
        arr = np.array(result)
        assert result.mode == "RGB"
        assert arr.mean() > 200  # mostly white

    def test_fully_opaque_rgba_preserves_colour(self):
        red = (255, 0, 0, 255)
        img = Image.new("RGBA", (16, 16), red)
        result = PokemonImageGenerator._ensure_white_background(img)
        arr = np.array(result)
        assert int(arr[:, :, 0].mean()) == 255  # red channel stays
        assert int(arr[:, :, 1].mean()) == 0    # green gone
        assert int(arr[:, :, 2].mean()) == 0    # blue gone


# ─────────────────────────────────────────────────────────────────────────────
# generate() – mocked pipeline
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerate:
    def test_returns_list_of_pil_images(self, default_config):
        pipe = _make_pipe_mock(_synthetic_image())
        gen = PokemonImageGenerator(default_config)
        with _patch_loader(pipe):
            images = gen.generate("Pikachu", num_images=2)
        assert len(images) == 2
        assert all(isinstance(img, Image.Image) for img in images)

    def test_num_images_controls_count(self, default_config):
        pipe = _make_pipe_mock(_synthetic_image())
        gen = PokemonImageGenerator(default_config)
        with _patch_loader(pipe):
            images = gen.generate("Pikachu", num_images=3)
        assert len(images) == 3
        assert pipe.call_count == 3

    def test_passes_prompt_to_pipeline(self, default_config):
        pipe = _make_pipe_mock(_synthetic_image())
        gen = PokemonImageGenerator(default_config)
        with _patch_loader(pipe):
            gen.generate("Snorlax", num_images=1)
        _, kwargs = pipe.call_args
        assert "Snorlax" in kwargs.get("prompt", "")

    def test_passes_negative_prompt(self, default_config):
        pipe = _make_pipe_mock(_synthetic_image())
        gen = PokemonImageGenerator(default_config)
        with _patch_loader(pipe):
            gen.generate("Pikachu", num_images=1)
        _, kwargs = pipe.call_args
        assert kwargs.get("negative_prompt") is not None

    def test_passes_num_inference_steps(self, default_config):
        pipe = _make_pipe_mock(_synthetic_image())
        gen = PokemonImageGenerator(default_config)
        with _patch_loader(pipe):
            gen.generate("Pikachu", num_images=1)
        _, kwargs = pipe.call_args
        assert kwargs.get("num_inference_steps") == default_config.num_inference_steps

    def test_passes_guidance_scale(self, default_config):
        pipe = _make_pipe_mock(_synthetic_image())
        gen = PokemonImageGenerator(default_config)
        with _patch_loader(pipe):
            gen.generate("Pikachu", num_images=1)
        _, kwargs = pipe.call_args
        assert kwargs.get("guidance_scale") == default_config.guidance_scale

    def test_passes_image_dimensions(self, default_config):
        pipe = _make_pipe_mock(_synthetic_image())
        gen = PokemonImageGenerator(default_config)
        with _patch_loader(pipe):
            gen.generate("Pikachu", num_images=1)
        _, kwargs = pipe.call_args
        assert kwargs.get("width") == default_config.width
        assert kwargs.get("height") == default_config.height

    def test_config_seed_used_when_no_override(self, default_config):
        """Verify config.seed flows into the per-image seed arithmetic."""
        pipe = _make_pipe_mock(_synthetic_image())
        gen = PokemonImageGenerator(default_config)
        with patch(
            "pokemon_stencil.image_gen.generator.load_sdxl_controlnet_pipeline",
            return_value=pipe,
        ), patch.object(gen, "_make_generator", wraps=gen._make_generator) as mock_mg:
            gen.generate("Pikachu", num_images=1)
        # seed arg passed is config.seed + 0 = 42
        mock_mg.assert_called_once_with(42)

    def test_explicit_seed_overrides_config(self, default_config):
        pipe = _make_pipe_mock(_synthetic_image())
        gen = PokemonImageGenerator(default_config)
        with patch(
            "pokemon_stencil.image_gen.generator.load_sdxl_controlnet_pipeline",
            return_value=pipe,
        ), patch.object(gen, "_make_generator", wraps=gen._make_generator) as mock_mg:
            gen.generate("Pikachu", num_images=1, seed=999)
        mock_mg.assert_called_once_with(999)

    def test_prompt_extras_included(self, default_config):
        pipe = _make_pipe_mock(_synthetic_image())
        gen = PokemonImageGenerator(default_config)
        with _patch_loader(pipe):
            gen.generate("Pikachu", prompt_extras="electric sparks", num_images=1)
        _, kwargs = pipe.call_args
        assert "electric sparks" in kwargs.get("prompt", "")

    def test_output_images_are_rgb(self, default_config):
        pipe = _make_pipe_mock(_synthetic_image())
        gen = PokemonImageGenerator(default_config)
        with _patch_loader(pipe):
            images = gen.generate("Pikachu", num_images=1)
        assert images[0].mode == "RGB"

    def test_rgba_output_converted_to_rgb(self, default_config):
        rgba_img = Image.new("RGBA", (64, 64), (255, 0, 0, 128))
        pipe = _make_pipe_mock(rgba_img)
        gen = PokemonImageGenerator(default_config)
        with _patch_loader(pipe):
            images = gen.generate("Pikachu", num_images=1)
        assert images[0].mode == "RGB"


# ─────────────────────────────────────────────────────────────────────────────
# save_images()
# ─────────────────────────────────────────────────────────────────────────────

class TestSaveImages:
    def test_saves_all_images(self, tmp_path, default_config):
        imgs = [_synthetic_image() for _ in range(3)]
        gen = PokemonImageGenerator(default_config)
        paths = gen.save_images(imgs, tmp_path / "out", prefix="test")
        assert len(paths) == 3
        for p in paths:
            assert p.exists()

    def test_creates_output_dir(self, tmp_path, default_config):
        out_dir = tmp_path / "nested" / "output"
        gen = PokemonImageGenerator(default_config)
        gen.save_images([_synthetic_image()], out_dir)
        assert out_dir.is_dir()

    def test_filenames_use_prefix(self, tmp_path, default_config):
        gen = PokemonImageGenerator(default_config)
        paths = gen.save_images([_synthetic_image()], tmp_path, prefix="gen")
        assert paths[0].name.startswith("gen_")

    def test_filenames_zero_padded(self, tmp_path, default_config):
        gen = PokemonImageGenerator(default_config)
        imgs = [_synthetic_image() for _ in range(2)]
        paths = gen.save_images(imgs, tmp_path)
        # Expect gen_000.png, gen_001.png
        assert paths[0].name == "gen_000.png"
        assert paths[1].name == "gen_001.png"

    def test_returns_path_objects(self, tmp_path, default_config):
        gen = PokemonImageGenerator(default_config)
        paths = gen.save_images([_synthetic_image()], tmp_path)
        assert all(isinstance(p, Path) for p in paths)


# ─────────────────────────────────────────────────────────────────────────────
# _make_generator (torch path)
# ─────────────────────────────────────────────────────────────────────────────

class TestMakeGenerator:
    def test_returns_none_when_seed_is_none(self, default_config):
        gen = PokemonImageGenerator(default_config)
        result = gen._make_generator(None)
        assert result is None

    def test_returns_none_when_torch_missing(self, default_config):
        gen = PokemonImageGenerator(default_config)
        with patch.dict(sys.modules, {"torch": None}):
            result = gen._make_generator(42)
        assert result is None
