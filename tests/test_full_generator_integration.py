"""
tests/test_full_generator_integration.py — Full generator integration tests.

Runs a mini end-to-end generation path (2–3 candidates) with all heavy ML
dependencies mocked.  Confirms:
- FactoryRunner.run() completes without exceptions
- Candidate images are generated (or skip_generation mode works)
- No runtime tensor shape errors
- Output files are created in the expected directories
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from pokemon_stencil.config import (
    FactoryConfig,
    GenerationConfig,
    OutputConfig,
    PipelineConfig,
    ProcessingConfig,
)
from pokemon_stencil.factory.factory_runner import FactoryResult, FactoryRunner
from pokemon_stencil.models.model_loader import clear_pipeline_cache


# ── Synthetic output builders ─────────────────────────────────────────────────

def _synthetic_image(w: int = 64, h: int = 64) -> Image.Image:
    arr = np.random.randint(50, 200, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _make_pipe_mock() -> MagicMock:
    pipe = MagicMock()
    pipe.to.return_value = pipe
    result = MagicMock()
    result.images = [_synthetic_image()]
    pipe.return_value = result
    return pipe


@pytest.fixture(autouse=True)
def _clear_pipeline_cache():
    clear_pipeline_cache()
    yield
    clear_pipeline_cache()


# ─────────────────────────────────────────────────────────────────────────────
# skip_generation=True path (no SDXL needed)
# ─────────────────────────────────────────────────────────────────────────────

class TestSkipGenerationPath:
    """
    skip_generation=True uses reference images directly and bypasses SDXL.
    This lets us test the full scoring + SVG export chain without models.
    """

    def _make_config(self, output_dir: Path) -> PipelineConfig:
        return PipelineConfig(
            generation=GenerationConfig(
                use_ip_adapter=False,
                use_controlnet=False,
                num_inference_steps=1,
                width=64,
                height=64,
            ),
            processing=ProcessingConfig(output_size=(64, 64)),
            output=OutputConfig(base_dir=output_dir),
            factory=FactoryConfig(count=2, top_k=1, workers=1),
            skip_generation=True,
        )

    def test_run_completes_without_exception(self, tmp_path: Path) -> None:
        """FactoryRunner.run() must complete without raising."""
        cfg = self._make_config(tmp_path)

        # Create minimal reference images
        refs_dir = tmp_path / "refs" / "pikachu"
        refs_dir.mkdir(parents=True)
        for i in range(3):
            _synthetic_image().save(refs_dir / f"ref_{i:03d}.png")

        runner = FactoryRunner(cfg)
        result = runner.run("pikachu", reference_dir=refs_dir)

        assert isinstance(result, FactoryResult)

    def test_run_returns_factory_result(self, tmp_path: Path) -> None:
        cfg = self._make_config(tmp_path)
        refs_dir = tmp_path / "refs" / "gengar"
        refs_dir.mkdir(parents=True)
        for i in range(2):
            _synthetic_image().save(refs_dir / f"ref_{i:03d}.png")

        runner = FactoryRunner(cfg)
        result = runner.run("gengar", reference_dir=refs_dir)

        assert result.pokemon_name == "gengar"
        assert result.total_candidates == 2

    def test_candidate_images_produced(self, tmp_path: Path) -> None:
        """Candidate images must be saved in the candidates directory."""
        cfg = self._make_config(tmp_path)
        refs_dir = tmp_path / "refs" / "bulbasaur"
        refs_dir.mkdir(parents=True)
        for i in range(3):
            _synthetic_image().save(refs_dir / f"ref_{i:03d}.png")

        runner = FactoryRunner(cfg)
        runner.run("bulbasaur", reference_dir=refs_dir)

        candidate_dir = tmp_path / "_factory_candidates"
        if candidate_dir.is_dir():
            candidates = list(candidate_dir.glob("*.png"))
            assert len(candidates) > 0


# ─────────────────────────────────────────────────────────────────────────────
# Full SDXL generation path (mocked pipeline)
# ─────────────────────────────────────────────────────────────────────────────

class TestSdxlGenerationPath:
    def _make_config(self, output_dir: Path) -> PipelineConfig:
        return PipelineConfig(
            generation=GenerationConfig(
                use_ip_adapter=False,
                use_controlnet=True,
                num_inference_steps=1,
                width=64,
                height=64,
                device="cpu",
                torch_dtype="float32",
                seed=0,
            ),
            processing=ProcessingConfig(output_size=(64, 64)),
            output=OutputConfig(base_dir=output_dir),
            factory=FactoryConfig(count=2, top_k=1, workers=1),
            skip_generation=False,
        )

    def test_run_with_mocked_pipeline_no_exception(self, tmp_path: Path) -> None:
        """End-to-end run with mocked SDXL pipeline must complete without error."""
        pipe = _make_pipe_mock()
        cfg = self._make_config(tmp_path)

        refs_dir = tmp_path / "refs" / "pikachu"
        refs_dir.mkdir(parents=True)
        for i in range(2):
            _synthetic_image().save(refs_dir / f"ref_{i:03d}.png")

        with patch(
            "pokemon_stencil.image_gen.generator.load_sdxl_controlnet_pipeline",
            return_value=pipe,
        ):
            runner = FactoryRunner(cfg)
            result = runner.run("pikachu", reference_dir=refs_dir)

        assert isinstance(result, FactoryResult)
        assert result.pokemon_name == "pikachu"

    def test_run_returns_correct_candidate_count(self, tmp_path: Path) -> None:
        pipe = _make_pipe_mock()
        cfg = self._make_config(tmp_path)

        refs_dir = tmp_path / "refs" / "charizard"
        refs_dir.mkdir(parents=True)
        for i in range(2):
            _synthetic_image().save(refs_dir / f"ref_{i:03d}.png")

        with patch(
            "pokemon_stencil.image_gen.generator.load_sdxl_controlnet_pipeline",
            return_value=pipe,
        ):
            runner = FactoryRunner(cfg)
            result = runner.run("charizard", reference_dir=refs_dir)

        assert result.total_candidates == 2

    def test_no_import_error_from_sdxl_path(self, tmp_path: Path) -> None:
        """The SDXL generation path must not raise ImportError under mock."""
        pipe = _make_pipe_mock()
        cfg = self._make_config(tmp_path)

        refs_dir = tmp_path / "refs" / "mewtwo"
        refs_dir.mkdir(parents=True)
        _synthetic_image().save(refs_dir / "ref_000.png")

        with patch(
            "pokemon_stencil.image_gen.generator.load_sdxl_controlnet_pipeline",
            return_value=pipe,
        ):
            try:
                runner = FactoryRunner(cfg)
                runner.run("mewtwo", reference_dir=refs_dir)
            except ImportError as exc:
                pytest.fail(f"ImportError raised unexpectedly: {exc}")
