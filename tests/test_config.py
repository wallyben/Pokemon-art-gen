"""Tests for pokemon_stencil.config dataclasses."""

from pathlib import Path

import pytest

from pokemon_stencil.config import (
    BatchConfig,
    GenerationConfig,
    OutputConfig,
    PipelineConfig,
    ProcessingConfig,
    StencilConfig,
    VectorConfig,
)


class TestGenerationConfig:
    def test_defaults(self):
        cfg = GenerationConfig()
        assert cfg.device == "cpu"
        assert cfg.torch_dtype == "float32"
        assert cfg.num_inference_steps == 28
        assert cfg.width == 1024
        assert cfg.height == 1024

    def test_model_id_returns_hub_id_when_local_absent(self, tmp_path):
        cfg = GenerationConfig(model_local_path=tmp_path / "nonexistent")
        assert cfg.model_id == cfg.model_hub_id

    def test_model_id_returns_local_path_when_present(self, tmp_path):
        local = tmp_path / "sd15"
        local.mkdir()
        cfg = GenerationConfig(model_local_path=local)
        assert cfg.model_id == local

    def test_custom_seed(self):
        cfg = GenerationConfig(seed=42)
        assert cfg.seed == 42

    def test_negative_prompt_non_empty(self):
        cfg = GenerationConfig()
        assert len(cfg.negative_prompt) > 0


class TestProcessingConfig:
    def test_defaults(self):
        cfg = ProcessingConfig()
        assert cfg.n_colors == 6
        assert cfg.output_size == (1024, 1024)
        assert cfg.min_region_area > 0

    def test_custom_n_colors(self):
        cfg = ProcessingConfig(n_colors=4)
        assert cfg.n_colors == 4


class TestVectorConfig:
    def test_canvas_dimensions_mm(self):
        cfg = VectorConfig()
        # 12 inches = 304.8 mm
        assert cfg.canvas_width_mm == pytest.approx(304.8)
        assert cfg.canvas_height_mm == pytest.approx(304.8)

    def test_dpi(self):
        cfg = VectorConfig()
        assert cfg.dpi == 96.0


class TestStencilConfig:
    def test_defaults(self):
        cfg = StencilConfig()
        assert cfg.bridge_width == 8
        assert cfg.min_cut_width_mm > 0
        assert cfg.add_registration_marks is True


class TestBatchConfig:
    def test_defaults(self):
        cfg = BatchConfig()
        assert cfg.count == 3
        assert cfg.workers == 1

    def test_custom_workers(self):
        cfg = BatchConfig(workers=4)
        assert cfg.workers == 4


class TestOutputConfig:
    def test_base_dir_is_outputs(self):
        cfg = OutputConfig()
        assert cfg.base_dir == Path("outputs")

    def test_path_helpers(self, tmp_path):
        cfg = OutputConfig(base_dir=tmp_path)
        run = "pikachu_00"
        assert cfg.images_path(run) == tmp_path / run / "images"
        assert cfg.simplified_path(run) == tmp_path / run / "simplified"
        assert cfg.masks_path(run) == tmp_path / run / "masks"
        assert cfg.svg_path(run) == tmp_path / run / "svg"

    def test_ensure_run_dirs_creates_all(self, tmp_path):
        cfg = OutputConfig(base_dir=tmp_path)
        cfg.ensure_run_dirs("test_run")
        assert cfg.images_path("test_run").is_dir()
        assert cfg.simplified_path("test_run").is_dir()
        assert cfg.masks_path("test_run").is_dir()
        assert cfg.svg_path("test_run").is_dir()

    def test_run_root(self, tmp_path):
        cfg = OutputConfig(base_dir=tmp_path)
        assert cfg.run_root("pikachu_00") == tmp_path / "pikachu_00"


class TestPipelineConfig:
    def test_aggregates_sub_configs(self):
        cfg = PipelineConfig()
        assert isinstance(cfg.generation, GenerationConfig)
        assert isinstance(cfg.processing, ProcessingConfig)
        assert isinstance(cfg.vector, VectorConfig)
        assert isinstance(cfg.stencil, StencilConfig)
        assert isinstance(cfg.output, OutputConfig)
        assert isinstance(cfg.batch, BatchConfig)

    def test_skip_generation_default_false(self):
        cfg = PipelineConfig()
        assert cfg.skip_generation is False

    def test_independent_instances(self):
        """Two PipelineConfig instances must not share mutable sub-configs."""
        a = PipelineConfig()
        b = PipelineConfig()
        a.processing.n_colors = 99
        assert b.processing.n_colors != 99
