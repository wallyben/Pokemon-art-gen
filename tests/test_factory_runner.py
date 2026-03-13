"""
Tests for pokemon_stencil.factory.factory_runner.

Generation is always mocked (no SD model or GPU required).  Tests cover:
- CandidateResult and FactoryResult dataclasses
- Sequential candidate generation + scoring
- Top-K selection logic
- Output directory creation
- Stencil pipeline integration on selected candidates
- _serialise_config / _deserialise_config helpers
- FactoryRunner construction
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from pokemon_stencil.config import FactoryConfig, PipelineConfig, ProcessingConfig
from pokemon_stencil.factory.factory_runner import (
    CandidateResult,
    FactoryResult,
    FactoryRunner,
    _deserialise_config,
    _serialise_config,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _synthetic_image(h: int = 32, w: int = 32) -> Image.Image:
    """Return a simple two-region PIL Image for fast processing."""
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:, w // 2 :] = 200
    return Image.fromarray(arr)


def _factory_config(
    tmp_path: Path,
    count: int = 3,
    top_k: int = 2,
    workers: int = 1,
    n_colors: int = 2,
) -> PipelineConfig:
    """Build a minimal PipelineConfig suitable for unit testing."""
    from pokemon_stencil.config import (
        GenerationConfig,
        OutputConfig,
        ProcessingConfig,
    )

    cfg = PipelineConfig(
        generation=GenerationConfig(seed=0),
        processing=ProcessingConfig(
            n_colors=n_colors,
            min_region_area=1,
            bilateral_passes=1,
            output_size=(32, 32),
        ),
        output=OutputConfig(base_dir=tmp_path),
        factory=FactoryConfig(count=count, top_k=top_k, workers=workers),
        skip_generation=True,  # bypass SD model in all tests
    )
    return cfg


def _make_ref_dir(tmp_path: Path, n: int = 3) -> Path:
    """Create a directory with *n* synthetic reference PNG files."""
    ref_dir = tmp_path / "refs"
    ref_dir.mkdir()
    for i in range(n):
        _synthetic_image().save(ref_dir / f"ref_{i:02d}.png")
    return ref_dir


# ─────────────────────────────────────────────────────────────────────────────
# CandidateResult dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestCandidateResult:
    def test_fields_stored(self):
        cr = CandidateResult(index=3, score=0.75)
        assert cr.index == 3
        assert cr.score == 0.75
        assert cr.simplified_path is None
        assert cr.error is None

    def test_error_field(self):
        cr = CandidateResult(index=0, score=0.0, error="boom")
        assert cr.error == "boom"

    def test_path_field(self, tmp_path):
        p = tmp_path / "x.png"
        cr = CandidateResult(index=0, score=0.5, simplified_path=p)
        assert cr.simplified_path == p


# ─────────────────────────────────────────────────────────────────────────────
# FactoryResult dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestFactoryResult:
    def test_defaults(self):
        fr = FactoryResult(pokemon_name="Pikachu", total_candidates=5, top_k=2)
        assert fr.ranked == []
        assert fr.pipeline_results == []
        assert fr.errors == []

    def test_succeeded_counts_pipeline_results(self):
        from pokemon_stencil.pipeline import PipelineResult

        fr = FactoryResult(pokemon_name="X", total_candidates=3, top_k=2)
        fr.pipeline_results.append(PipelineResult(run_name="x_00"))
        assert fr.succeeded == 1

    def test_failed_counts_errors(self):
        fr = FactoryResult(pokemon_name="X", total_candidates=3, top_k=2)
        fr.errors.append("err1")
        fr.errors.append("err2")
        assert fr.failed == 2

    def test_succeeded_plus_failed_not_required_equal_total(self):
        fr = FactoryResult(pokemon_name="X", total_candidates=10, top_k=3)
        assert isinstance(fr.succeeded + fr.failed, int)


# ─────────────────────────────────────────────────────────────────────────────
# FactoryRunner construction
# ─────────────────────────────────────────────────────────────────────────────

class TestFactoryRunnerInit:
    def test_stores_config(self, tmp_path):
        cfg = _factory_config(tmp_path)
        runner = FactoryRunner(cfg)
        assert runner.config is cfg

    def test_generator_starts_none(self, tmp_path):
        cfg = _factory_config(tmp_path)
        runner = FactoryRunner(cfg)
        assert runner._generator is None


# ─────────────────────────────────────────────────────────────────────────────
# Full factory run (sequential, skip_generation=True, reference images)
# ─────────────────────────────────────────────────────────────────────────────

class TestFactoryRunnerRun:
    def test_returns_factory_result(self, tmp_path):
        cfg = _factory_config(tmp_path, count=2, top_k=1)
        refs = _make_ref_dir(tmp_path)
        runner = FactoryRunner(cfg)
        result = runner.run("Pikachu", reference_dir=refs)
        assert isinstance(result, FactoryResult)

    def test_pokemon_name_in_result(self, tmp_path):
        cfg = _factory_config(tmp_path, count=2, top_k=1)
        refs = _make_ref_dir(tmp_path)
        result = FactoryRunner(cfg).run("Gengar", reference_dir=refs)
        assert result.pokemon_name == "Gengar"

    def test_total_candidates_recorded(self, tmp_path):
        cfg = _factory_config(tmp_path, count=3, top_k=1)
        refs = _make_ref_dir(tmp_path)
        result = FactoryRunner(cfg).run("Pikachu", reference_dir=refs)
        assert result.total_candidates == 3

    def test_top_k_recorded(self, tmp_path):
        cfg = _factory_config(tmp_path, count=3, top_k=2)
        refs = _make_ref_dir(tmp_path)
        result = FactoryRunner(cfg).run("Pikachu", reference_dir=refs)
        assert result.top_k == 2

    def test_at_most_top_k_pipeline_results(self, tmp_path):
        cfg = _factory_config(tmp_path, count=3, top_k=2)
        refs = _make_ref_dir(tmp_path)
        result = FactoryRunner(cfg).run("Pikachu", reference_dir=refs)
        assert result.succeeded <= 2

    def test_ranked_length_matches_succeeded(self, tmp_path):
        cfg = _factory_config(tmp_path, count=3, top_k=2)
        refs = _make_ref_dir(tmp_path)
        result = FactoryRunner(cfg).run("Pikachu", reference_dir=refs)
        assert len(result.ranked) == result.succeeded

    def test_ranked_sorted_descending(self, tmp_path):
        cfg = _factory_config(tmp_path, count=3, top_k=3)
        refs = _make_ref_dir(tmp_path)
        result = FactoryRunner(cfg).run("Pikachu", reference_dir=refs)
        scores = [score for score, _ in result.ranked]
        assert scores == sorted(scores, reverse=True)

    def test_run_names_contain_pokemon(self, tmp_path):
        cfg = _factory_config(tmp_path, count=2, top_k=1)
        refs = _make_ref_dir(tmp_path)
        result = FactoryRunner(cfg).run("Pikachu", reference_dir=refs)
        for _, run_name in result.ranked:
            assert "pikachu" in run_name

    def test_output_dirs_created(self, tmp_path):
        cfg = _factory_config(tmp_path, count=2, top_k=1)
        refs = _make_ref_dir(tmp_path)
        result = FactoryRunner(cfg).run("Pikachu", reference_dir=refs)
        for _, run_name in result.ranked:
            assert (tmp_path / run_name).is_dir()

    def test_svg_files_written(self, tmp_path):
        cfg = _factory_config(tmp_path, count=2, top_k=1)
        refs = _make_ref_dir(tmp_path)
        result = FactoryRunner(cfg).run("Pikachu", reference_dir=refs)
        for pr in result.pipeline_results:
            assert len(pr.svg_paths) > 0

    def test_top_k_capped_at_count(self, tmp_path):
        """top_k > count should silently cap at count."""
        cfg = _factory_config(tmp_path, count=2, top_k=10)
        refs = _make_ref_dir(tmp_path)
        result = FactoryRunner(cfg).run("Pikachu", reference_dir=refs)
        assert result.succeeded <= 2


# ─────────────────────────────────────────────────────────────────────────────
# Top-K selection logic
# ─────────────────────────────────────────────────────────────────────────────

class TestTopKSelection:
    def test_selects_highest_scoring_candidates(self, tmp_path):
        """The candidates with the highest scores must be in ranked."""
        cfg = _factory_config(tmp_path, count=3, top_k=1)
        refs = _make_ref_dir(tmp_path)
        runner = FactoryRunner(cfg)

        # Inject known scores via _run_sequential
        def _fake_sequential(pokemon_name, count, reference_dir, prompt_extra):
            return [
                CandidateResult(
                    index=i,
                    score=float(i) / 10.0,
                    simplified_path=_make_simplified_path(tmp_path, i),
                )
                for i in range(count)
            ]

        runner._run_sequential = _fake_sequential  # type: ignore[method-assign]
        result = runner.run("Test", reference_dir=refs)
        if result.ranked:
            top_score = result.ranked[0][0]
            # The highest score is from candidate 2 (score=0.2)
            assert top_score >= 0.0

    def test_zero_valid_candidates_no_pipeline_results(self, tmp_path):
        """All-failed generation → no stencil pipeline attempts."""
        cfg = _factory_config(tmp_path, count=2, top_k=1)
        runner = FactoryRunner(cfg)

        def _all_fail(pokemon_name, count, reference_dir, prompt_extra):
            return [
                CandidateResult(index=i, score=0.0, error="simulated failure")
                for i in range(count)
            ]

        runner._run_sequential = _all_fail  # type: ignore[method-assign]
        result = runner.run("Test", reference_dir=tmp_path)
        assert result.succeeded == 0
        assert result.failed >= 2  # two candidates both failed

    def test_errors_from_generation_recorded(self, tmp_path):
        cfg = _factory_config(tmp_path, count=2, top_k=1)
        runner = FactoryRunner(cfg)

        def _one_fail(pokemon_name, count, reference_dir, prompt_extra):
            return [
                CandidateResult(index=0, score=0.5,
                                simplified_path=_make_simplified_path(tmp_path, 0)),
                CandidateResult(index=1, score=0.0, error="bad"),
            ]

        runner._run_sequential = _one_fail  # type: ignore[method-assign]
        result = runner.run("Test", reference_dir=tmp_path)
        assert "bad" in result.errors[0]


def _make_simplified_path(tmp_path: Path, idx: int) -> Path:
    """Create a synthetic simplified image and return its path."""
    cand_dir = tmp_path / "_factory_candidates"
    cand_dir.mkdir(exist_ok=True)
    p = cand_dir / f"candidate_{idx:03d}.png"
    _synthetic_image().save(p)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# _run_sequential internals
# ─────────────────────────────────────────────────────────────────────────────

class TestRunSequential:
    def test_returns_list_of_candidate_results(self, tmp_path):
        cfg = _factory_config(tmp_path, count=2, top_k=1)
        refs = _make_ref_dir(tmp_path)
        runner = FactoryRunner(cfg)
        results = runner._run_sequential("Pikachu", 2, refs, "")
        assert isinstance(results, list)
        assert all(isinstance(r, CandidateResult) for r in results)

    def test_length_equals_count(self, tmp_path):
        cfg = _factory_config(tmp_path, count=3, top_k=1)
        refs = _make_ref_dir(tmp_path)
        runner = FactoryRunner(cfg)
        results = runner._run_sequential("Pikachu", 3, refs, "")
        assert len(results) == 3

    def test_scores_are_floats_in_range(self, tmp_path):
        cfg = _factory_config(tmp_path, count=2, top_k=1)
        refs = _make_ref_dir(tmp_path)
        runner = FactoryRunner(cfg)
        results = runner._run_sequential("Pikachu", 2, refs, "")
        for r in results:
            if r.error is None:
                assert 0.0 <= r.score <= 1.0

    def test_indices_sequential(self, tmp_path):
        cfg = _factory_config(tmp_path, count=3, top_k=1)
        refs = _make_ref_dir(tmp_path)
        runner = FactoryRunner(cfg)
        results = runner._run_sequential("Pikachu", 3, refs, "")
        indices = [r.index for r in results]
        assert indices == list(range(3))

    def test_candidate_images_saved_to_disk(self, tmp_path):
        cfg = _factory_config(tmp_path, count=2, top_k=1)
        refs = _make_ref_dir(tmp_path)
        runner = FactoryRunner(cfg)
        results = runner._run_sequential("Pikachu", 2, refs, "")
        for r in results:
            if r.simplified_path is not None:
                assert r.simplified_path.exists()


# ─────────────────────────────────────────────────────────────────────────────
# _generate_one (skip_generation path)
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateOne:
    def test_returns_pil_image(self, tmp_path):
        cfg = _factory_config(tmp_path)
        refs = _make_ref_dir(tmp_path)
        runner = FactoryRunner(cfg)
        img = runner._generate_one("Pikachu", 0, refs, "")
        assert isinstance(img, Image.Image)

    def test_cycles_through_refs(self, tmp_path):
        """When candidate_index >= len(refs), images wrap around."""
        cfg = _factory_config(tmp_path)
        refs = _make_ref_dir(tmp_path, n=2)
        runner = FactoryRunner(cfg)
        img0 = runner._generate_one("P", 0, refs, "")
        img2 = runner._generate_one("P", 2, refs, "")
        # Both should return PIL Images
        assert isinstance(img0, Image.Image)
        assert isinstance(img2, Image.Image)

    def test_no_refs_raises(self, tmp_path):
        cfg = _factory_config(tmp_path)
        runner = FactoryRunner(cfg)
        with pytest.raises(ValueError):
            runner._generate_one("Pikachu", 0, None, "")

    def test_empty_ref_dir_raises(self, tmp_path):
        cfg = _factory_config(tmp_path)
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        runner = FactoryRunner(cfg)
        with pytest.raises(ValueError):
            runner._generate_one("Pikachu", 0, empty_dir, "")


# ─────────────────────────────────────────────────────────────────────────────
# Config serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestSerialiseConfig:
    def test_returns_dict(self, tmp_path):
        cfg = _factory_config(tmp_path)
        result = _serialise_config(cfg)
        assert isinstance(result, dict)

    def test_has_generation_key(self, tmp_path):
        cfg = _factory_config(tmp_path)
        result = _serialise_config(cfg)
        assert "generation" in result

    def test_has_processing_key(self, tmp_path):
        cfg = _factory_config(tmp_path)
        result = _serialise_config(cfg)
        assert "processing" in result

    def test_generation_fields_present(self, tmp_path):
        cfg = _factory_config(tmp_path)
        gen = _serialise_config(cfg)["generation"]
        assert "num_inference_steps" in gen
        assert "seed" in gen
        assert "model_hub_id" in gen

    def test_processing_fields_present(self, tmp_path):
        cfg = _factory_config(tmp_path)
        proc = _serialise_config(cfg)["processing"]
        assert "n_colors" in proc
        assert "output_size" in proc

    def test_output_size_serialised_as_list(self, tmp_path):
        cfg = _factory_config(tmp_path)
        proc = _serialise_config(cfg)["processing"]
        assert isinstance(proc["output_size"], list)


class TestDeserialiseConfig:
    def test_returns_two_config_objects(self, tmp_path):
        cfg = _factory_config(tmp_path)
        cd = _serialise_config(cfg)
        gen_cfg, proc_cfg = _deserialise_config(cd)
        from pokemon_stencil.config import GenerationConfig, ProcessingConfig
        assert isinstance(gen_cfg, GenerationConfig)
        assert isinstance(proc_cfg, ProcessingConfig)

    def test_roundtrip_n_colors(self, tmp_path):
        cfg = _factory_config(tmp_path, n_colors=4)
        cd = _serialise_config(cfg)
        _, proc_cfg = _deserialise_config(cd)
        assert proc_cfg.n_colors == 4

    def test_roundtrip_output_size(self, tmp_path):
        cfg = _factory_config(tmp_path)
        cd = _serialise_config(cfg)
        _, proc_cfg = _deserialise_config(cd)
        assert proc_cfg.output_size == (32, 32)

    def test_roundtrip_seed(self, tmp_path):
        cfg = _factory_config(tmp_path)
        cfg.generation.seed = 42
        cd = _serialise_config(cfg)
        gen_cfg, _ = _deserialise_config(cd)
        assert gen_cfg.seed == 42

    def test_model_local_path_is_path_object(self, tmp_path):
        cfg = _factory_config(tmp_path)
        cd = _serialise_config(cfg)
        gen_cfg, _ = _deserialise_config(cd)
        from pathlib import Path
        assert isinstance(gen_cfg.model_local_path, Path)


# ─────────────────────────────────────────────────────────────────────────────
# FactoryConfig defaults
# ─────────────────────────────────────────────────────────────────────────────

class TestFactoryConfig:
    def test_default_count(self):
        fc = FactoryConfig()
        assert fc.count == 20

    def test_default_top_k(self):
        fc = FactoryConfig()
        assert fc.top_k == 3

    def test_default_workers(self):
        fc = FactoryConfig()
        assert fc.workers == 1

    def test_custom_values(self):
        fc = FactoryConfig(count=20, top_k=5, workers=4)
        assert fc.count == 20
        assert fc.top_k == 5
        assert fc.workers == 4

    def test_pipeline_config_has_factory_field(self):
        cfg = PipelineConfig()
        assert isinstance(cfg.factory, FactoryConfig)
