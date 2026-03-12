"""
Tests for pokemon_stencil.image_gen.composition_guidance.

All tests use synthetic PIL Images; no Stable Diffusion execution is required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from pokemon_stencil.image_gen.composition_guidance import (
    CompositionMode,
    generate_composition_map,
)


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _uniform_image(size: tuple = (64, 64), colour: tuple = (128, 128, 128)) -> Image.Image:
    """Return a solid-colour RGB image with no edges."""
    return Image.new("RGB", size, colour)


def _gradient_image(size: tuple = (64, 64)) -> Image.Image:
    """Return an image with a strong left-to-right brightness gradient."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for x in range(size[0]):
        val = int(255 * x / max(size[0] - 1, 1))
        arr[:, x, :] = val
    return Image.fromarray(arr)


def _checkerboard_image(size: tuple = (64, 64), block: int = 8) -> Image.Image:
    """Return a checkerboard image with strong edges."""
    arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    for y in range(size[1]):
        for x in range(size[0]):
            if (x // block + y // block) % 2 == 0:
                arr[y, x, :] = 255
    return Image.fromarray(arr)


# ─────────────────────────────────────────────────────────────────────────────
# EDGE mode
# ─────────────────────────────────────────────────────────────────────────────

class TestEdgeMapGeneration:
    """Tests for Canny-based EDGE mode."""

    def test_returns_numpy_array(self):
        img = _uniform_image()
        result = generate_composition_map(img, mode=CompositionMode.EDGE)
        assert isinstance(result, np.ndarray)

    def test_output_dtype_is_uint8(self):
        img = _gradient_image()
        result = generate_composition_map(img, mode=CompositionMode.EDGE)
        assert result.dtype == np.uint8

    def test_correct_output_dimensions_matches_input(self):
        img = _gradient_image(size=(80, 60))
        result = generate_composition_map(img, mode=CompositionMode.EDGE)
        # Result shape is (H, W) — note PIL size is (W, H).
        assert result.shape == (60, 80)

    def test_square_image_dimensions(self):
        img = _gradient_image(size=(64, 64))
        result = generate_composition_map(img, mode=CompositionMode.EDGE)
        assert result.shape == (64, 64)

    def test_uniform_image_produces_few_edges(self):
        """A solid-colour image should produce almost no edges."""
        img = _uniform_image(colour=(128, 128, 128))
        result = generate_composition_map(img, mode=CompositionMode.EDGE)
        # Very few non-zero pixels expected for a uniform image.
        nonzero_frac = np.count_nonzero(result) / result.size
        assert nonzero_frac < 0.05

    def test_gradient_image_produces_edges(self):
        """A strong brightness gradient should produce detectable edges."""
        img = _checkerboard_image()
        result = generate_composition_map(img, mode=CompositionMode.EDGE)
        assert result.max() == 255

    def test_deterministic_output(self):
        """Same input must always produce the same output."""
        img = _gradient_image()
        result1 = generate_composition_map(img, mode=CompositionMode.EDGE)
        result2 = generate_composition_map(img, mode=CompositionMode.EDGE)
        np.testing.assert_array_equal(result1, result2)

    def test_dilation_increases_edge_density(self):
        """Dilation should produce more non-zero pixels than no dilation."""
        img = _checkerboard_image()
        no_dil = generate_composition_map(
            img, mode=CompositionMode.EDGE, dilate=False
        )
        with_dil = generate_composition_map(
            img, mode=CompositionMode.EDGE, dilate=True, dilate_kernel_size=3
        )
        assert np.count_nonzero(with_dil) >= np.count_nonzero(no_dil)

    def test_invalid_blur_kernel_even_raises(self):
        img = _uniform_image()
        with pytest.raises(ValueError, match="blur_kernel_size"):
            generate_composition_map(img, mode=CompositionMode.EDGE, blur_kernel_size=4)

    def test_invalid_blur_kernel_zero_raises(self):
        img = _uniform_image()
        with pytest.raises(ValueError, match="blur_kernel_size"):
            generate_composition_map(img, mode=CompositionMode.EDGE, blur_kernel_size=0)

    def test_invalid_blur_kernel_negative_raises(self):
        img = _uniform_image()
        with pytest.raises(ValueError, match="blur_kernel_size"):
            generate_composition_map(img, mode=CompositionMode.EDGE, blur_kernel_size=-1)


# ─────────────────────────────────────────────────────────────────────────────
# SKETCH mode
# ─────────────────────────────────────────────────────────────────────────────

class TestSketchMapGeneration:
    """Tests for Sobel + Laplacian SKETCH mode."""

    def test_returns_numpy_array(self):
        img = _uniform_image()
        result = generate_composition_map(img, mode=CompositionMode.SKETCH)
        assert isinstance(result, np.ndarray)

    def test_output_dtype_is_uint8(self):
        img = _gradient_image()
        result = generate_composition_map(img, mode=CompositionMode.SKETCH)
        assert result.dtype == np.uint8

    def test_correct_output_dimensions(self):
        img = _gradient_image(size=(80, 60))
        result = generate_composition_map(img, mode=CompositionMode.SKETCH)
        assert result.shape == (60, 80)

    def test_gradient_image_produces_edges(self):
        img = _checkerboard_image()
        result = generate_composition_map(img, mode=CompositionMode.SKETCH)
        assert result.max() > 0

    def test_deterministic_output(self):
        img = _gradient_image()
        result1 = generate_composition_map(img, mode=CompositionMode.SKETCH)
        result2 = generate_composition_map(img, mode=CompositionMode.SKETCH)
        np.testing.assert_array_equal(result1, result2)

    def test_uniform_image_low_output(self):
        """A uniform image should produce very low sketch values."""
        img = _uniform_image(colour=(128, 128, 128))
        result = generate_composition_map(img, mode=CompositionMode.SKETCH)
        assert result.max() < 30


# ─────────────────────────────────────────────────────────────────────────────
# General / mode-agnostic behaviour
# ─────────────────────────────────────────────────────────────────────────────

class TestCompositionMapGeneral:
    """Mode-agnostic tests and input-type acceptance checks."""

    def test_default_mode_is_edge(self):
        img = _gradient_image()
        default = generate_composition_map(img)
        explicit_edge = generate_composition_map(img, mode=CompositionMode.EDGE)
        np.testing.assert_array_equal(default, explicit_edge)

    def test_edge_and_sketch_outputs_differ(self):
        img = _checkerboard_image()
        edge = generate_composition_map(img, mode=CompositionMode.EDGE)
        sketch = generate_composition_map(img, mode=CompositionMode.SKETCH)
        assert not np.array_equal(edge, sketch)

    def test_rgba_input_accepted(self):
        img = Image.new("RGBA", (64, 64), (200, 100, 50, 255))
        result = generate_composition_map(img)
        assert isinstance(result, np.ndarray)
        assert result.shape == (64, 64)

    def test_grayscale_l_mode_input_accepted(self):
        img = Image.new("L", (64, 64), 128)
        result = generate_composition_map(img)
        assert isinstance(result, np.ndarray)
        assert result.shape == (64, 64)

    def test_output_is_2d(self):
        img = _gradient_image()
        result = generate_composition_map(img)
        assert result.ndim == 2

    def test_pixel_values_in_valid_range(self):
        img = _checkerboard_image()
        result = generate_composition_map(img, mode=CompositionMode.EDGE)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_sketch_pixel_values_in_valid_range(self):
        img = _checkerboard_image()
        result = generate_composition_map(img, mode=CompositionMode.SKETCH)
        assert result.min() >= 0
        assert result.max() <= 255

    def test_non_square_image_preserves_dimensions(self):
        img = _gradient_image(size=(100, 50))
        for mode in CompositionMode:
            result = generate_composition_map(img, mode=mode)
            assert result.shape == (50, 100), f"Failed for mode={mode}"


# ─────────────────────────────────────────────────────────────────────────────
# Generator integration: composition_map parameter
# ─────────────────────────────────────────────────────────────────────────────

class TestGeneratorAcceptsCompositionMap:
    """Verify PokemonImageGenerator.generate() accepts a composition_map arg."""

    @pytest.fixture(autouse=True)
    def flush_cache(self):
        from pokemon_stencil.models.model_loader import clear_pipeline_cache
        clear_pipeline_cache()
        yield
        clear_pipeline_cache()

    def _make_pipe_mock(self, output_image: Image.Image) -> MagicMock:
        result_mock = MagicMock()
        result_mock.images = [output_image]
        pipe = MagicMock(return_value=result_mock)
        return pipe

    def test_generate_accepts_none_composition_map(self):
        """Passing composition_map=None keeps existing behaviour."""
        from pokemon_stencil.config import GenerationConfig
        from pokemon_stencil.image_gen.generator import PokemonImageGenerator

        config = GenerationConfig(num_inference_steps=1, width=64, height=64)
        img = Image.new("RGB", (64, 64), (100, 150, 200))
        pipe = self._make_pipe_mock(img)

        with patch(
            "pokemon_stencil.image_gen.generator.load_sd_pipeline",
            return_value=pipe,
        ):
            gen = PokemonImageGenerator(config)
            images = gen.generate("Pikachu", composition_map=None)

        assert len(images) == 1
        # No image/strength kwargs when composition_map is None.
        _, kwargs = pipe.call_args
        assert "image" not in kwargs
        assert "strength" not in kwargs

    def test_generate_with_composition_map_passes_image_and_strength(self):
        """When composition_map is provided, image + strength reach the pipe."""
        from pokemon_stencil.config import GenerationConfig
        from pokemon_stencil.image_gen.generator import PokemonImageGenerator

        config = GenerationConfig(
            num_inference_steps=1, width=64, height=64,
            use_composition_guidance=True,
            composition_strength=0.5,
        )
        img = Image.new("RGB", (64, 64), (100, 150, 200))
        pipe = self._make_pipe_mock(img)
        comp_map = np.zeros((64, 64), dtype=np.uint8)

        with patch(
            "pokemon_stencil.image_gen.generator.load_sd_pipeline",
            return_value=pipe,
        ):
            gen = PokemonImageGenerator(config)
            images = gen.generate("Pikachu", composition_map=comp_map)

        assert len(images) == 1
        _, kwargs = pipe.call_args
        assert "image" in kwargs
        assert kwargs["strength"] == pytest.approx(0.5)

    def test_generate_composition_map_disabled_ignores_map(self):
        """use_composition_guidance=False skips conditioning even if map given."""
        from pokemon_stencil.config import GenerationConfig
        from pokemon_stencil.image_gen.generator import PokemonImageGenerator

        config = GenerationConfig(
            num_inference_steps=1, width=64, height=64,
            use_composition_guidance=False,
        )
        img = Image.new("RGB", (64, 64), (100, 150, 200))
        pipe = self._make_pipe_mock(img)
        comp_map = np.ones((64, 64), dtype=np.uint8) * 255

        with patch(
            "pokemon_stencil.image_gen.generator.load_sd_pipeline",
            return_value=pipe,
        ):
            gen = PokemonImageGenerator(config)
            images = gen.generate("Pikachu", composition_map=comp_map)

        assert len(images) == 1
        _, kwargs = pipe.call_args
        assert "image" not in kwargs
        assert "strength" not in kwargs

    def test_composition_map_to_image_2d_input(self):
        """_composition_map_to_image handles 2-D (H, W) grayscale arrays."""
        from pokemon_stencil.image_gen.generator import PokemonImageGenerator

        comp_map = np.zeros((48, 64), dtype=np.uint8)
        result = PokemonImageGenerator._composition_map_to_image(comp_map, (64, 48))
        assert isinstance(result, Image.Image)
        assert result.size == (64, 48)

    def test_composition_map_to_image_3d_input(self):
        """_composition_map_to_image handles 3-D (H, W, 3) arrays."""
        from pokemon_stencil.image_gen.generator import PokemonImageGenerator

        comp_map = np.zeros((48, 64, 3), dtype=np.uint8)
        result = PokemonImageGenerator._composition_map_to_image(comp_map, (64, 48))
        assert isinstance(result, Image.Image)
        assert result.size == (64, 48)

    def test_generate_returns_rgb_images_with_composition_map(self):
        """Output images are RGB regardless of composition map presence."""
        from pokemon_stencil.config import GenerationConfig
        from pokemon_stencil.image_gen.generator import PokemonImageGenerator

        config = GenerationConfig(
            num_inference_steps=1, width=64, height=64,
            use_composition_guidance=True,
        )
        img = Image.new("RGB", (64, 64), (100, 150, 200))
        pipe = self._make_pipe_mock(img)
        comp_map = np.zeros((64, 64), dtype=np.uint8)

        with patch(
            "pokemon_stencil.image_gen.generator.load_sd_pipeline",
            return_value=pipe,
        ):
            gen = PokemonImageGenerator(config)
            images = gen.generate("Gengar", composition_map=comp_map)

        assert images[0].mode == "RGB"
