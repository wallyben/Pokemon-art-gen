"""
Tests for pokemon_stencil.pipeline.Pipeline.

All tests use synthetic PIL Images rather than loading real files or
running Stable Diffusion, so no GPU or internet access is required.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from pokemon_stencil.config import PipelineConfig, ProcessingConfig
from pokemon_stencil.pipeline import Pipeline, PipelineResult


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def tiny_config(tmp_path) -> PipelineConfig:
    """PipelineConfig with tiny output size and minimal colours for speed."""
    from pokemon_stencil.config import OutputConfig

    cfg = PipelineConfig(
        processing=ProcessingConfig(
            n_colors=3,
            output_size=(64, 64),
            min_region_area=10,
        ),
        output=OutputConfig(base_dir=tmp_path),
        skip_generation=True,
    )
    return cfg


@pytest.fixture()
def synthetic_image() -> Image.Image:
    """64×64 RGB image with three solid colour blocks."""
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:21, :] = [255, 0, 0]    # red top third
    arr[21:42, :] = [0, 255, 0]  # green middle third
    arr[42:, :] = [0, 0, 255]    # blue bottom third
    return Image.fromarray(arr)


@pytest.fixture()
def ref_dir(tmp_path, synthetic_image) -> Path:
    """Temporary directory with one synthetic reference image."""
    d = tmp_path / "refs"
    d.mkdir()
    synthetic_image.save(d / "ref_000.png")
    return d


# ─────────────────────────────────────────────────────────────────────────────
# PipelineResult
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineResult:
    def test_svg_dir_none_when_no_svgs(self):
        r = PipelineResult(run_name="test")
        assert r.svg_dir is None

    def test_svg_dir_returns_parent(self, tmp_path):
        r = PipelineResult(run_name="test")
        r.svg_paths.append(tmp_path / "svg" / "layer_00.svg")
        assert r.svg_dir == tmp_path / "svg"


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline.run_from_image
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineRunFromImage:
    def test_creates_output_dirs(self, tiny_config, tmp_path, synthetic_image):
        img_file = tmp_path / "input.png"
        synthetic_image.save(img_file)

        pipeline = Pipeline(tiny_config)
        result = pipeline.run_from_image(
            image_path=img_file,
            design_name="test",
            run_name="test_00",
        )

        assert tiny_config.output.simplified_path("test_00").is_dir()
        assert tiny_config.output.masks_path("test_00").is_dir()

    def test_saves_simplified_image(self, tiny_config, tmp_path, synthetic_image):
        img_file = tmp_path / "input.png"
        synthetic_image.save(img_file)

        pipeline = Pipeline(tiny_config)
        result = pipeline.run_from_image(
            image_path=img_file,
            design_name="test",
            run_name="test_00",
        )

        assert len(result.simplified_paths) == 1
        assert result.simplified_paths[0].exists()

    def test_saves_mask_files(self, tiny_config, tmp_path, synthetic_image):
        img_file = tmp_path / "input.png"
        synthetic_image.save(img_file)

        pipeline = Pipeline(tiny_config)
        result = pipeline.run_from_image(
            image_path=img_file,
            design_name="test",
            run_name="test_00",
        )

        # At least one mask should be written (may be fewer than n_colors
        # if some regions are below min_region_area in a tiny synthetic image)
        assert len(result.mask_paths) >= 1
        for mp in result.mask_paths:
            assert mp.exists()

    def test_returns_pipeline_result(self, tiny_config, tmp_path, synthetic_image):
        img_file = tmp_path / "input.png"
        synthetic_image.save(img_file)

        pipeline = Pipeline(tiny_config)
        result = pipeline.run_from_image(
            image_path=img_file,
            design_name="test",
            run_name="test_00",
        )
        assert isinstance(result, PipelineResult)
        assert result.run_name == "test_00"

    def test_nonexistent_image_raises(self, tiny_config):
        pipeline = Pipeline(tiny_config)
        with pytest.raises(FileNotFoundError):
            pipeline.run_from_image(
                image_path=Path("/nonexistent/image.png"),
                design_name="test",
                run_name="test_00",
            )


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline.run (skip_generation path)
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineRunSkipGeneration:
    def test_run_with_refs(self, tiny_config, ref_dir):
        pipeline = Pipeline(tiny_config)
        result = pipeline.run(
            pokemon_name="Pikachu",
            run_name="pikachu_00",
            reference_dir=ref_dir,
        )
        assert isinstance(result, PipelineResult)
        assert len(result.image_paths) >= 1

    def test_run_no_refs_raises(self, tiny_config):
        pipeline = Pipeline(tiny_config)
        with pytest.raises(ValueError, match="reference_dir"):
            pipeline.run(
                pokemon_name="Pikachu",
                run_name="pikachu_00",
                reference_dir=None,
            )

    def test_source_images_saved(self, tiny_config, ref_dir):
        pipeline = Pipeline(tiny_config)
        result = pipeline.run(
            pokemon_name="Pikachu",
            run_name="pikachu_00",
            reference_dir=ref_dir,
        )
        for p in result.image_paths:
            assert p.exists()
