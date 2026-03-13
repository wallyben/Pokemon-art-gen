"""
tests/test_candidate_runner.py — CandidateRunner and scoring tests.

Uses a stub provider to avoid real API/GPU calls.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pokemon_stencil.generation.base import (
    AbstractGenerationProvider,
    GenerationRequest,
    GenerationResult,
)
from pokemon_stencil.generation.candidate_runner import (
    CandidateRunner,
    CandidateScore,
    _score_image,
)
from pokemon_stencil.generation.prompt_builder import PokemonPromptBuilder


# ── Stub provider ──────────────────────────────────────────────────────────────

class _StubProvider(AbstractGenerationProvider):
    """Returns synthetic PNG files for each generate() call."""

    def __init__(self, fail_at: set = None):
        self._call_count = 0
        self._fail_at = fail_at or set()

    @property
    def name(self) -> str:
        return "stub"

    def is_available(self) -> bool:
        return True

    def generate(self, request: GenerationRequest, output_dir: Path) -> GenerationResult:
        idx = self._call_count
        self._call_count += 1
        if idx in self._fail_at:
            return GenerationResult(images=[], provider=self.name)

        # Write a real minimal PNG using Pillow
        try:
            from PIL import Image
            img = Image.new("RGB", (64, 64), color=(idx * 10 % 255, 200, 100))
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / "gen_0000.png"
            img.save(str(path))
            return GenerationResult(images=[path], provider=self.name)
        except ImportError:
            # PIL not installed — write raw bytes minimal stub
            output_dir.mkdir(parents=True, exist_ok=True)
            path = output_dir / "gen_0000.png"
            # Minimal 1x1 white PNG bytes
            path.write_bytes(
                b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
                b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00"
                b"\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00\x05\x18\xd8N"
                b"\x00\x00\x00\x00IEND\xaeB`\x82"
            )
            return GenerationResult(images=[path], provider=self.name)


# ── CandidateRunner tests ──────────────────────────────────────────────────────

class TestCandidateRunner:
    def test_generates_correct_count(self, tmp_path):
        provider = _StubProvider()
        runner = CandidateRunner(provider)
        result = runner.run("Pikachu", tmp_path, n_candidates=4, top_k=2)
        assert result.total_generated == 4

    def test_returns_top_k(self, tmp_path):
        provider = _StubProvider()
        runner = CandidateRunner(provider)
        result = runner.run("Pikachu", tmp_path, n_candidates=6, top_k=3)
        assert len(result.top_k) <= 3

    def test_top_k_limited_by_generated(self, tmp_path):
        """If only 2 images generate successfully, top_k is at most 2."""
        provider = _StubProvider(fail_at={1, 2, 3})  # only idx 0 succeeds
        runner = CandidateRunner(provider)
        result = runner.run("Pikachu", tmp_path, n_candidates=4, top_k=3)
        assert len(result.top_k) <= 1

    def test_best_property(self, tmp_path):
        provider = _StubProvider()
        runner = CandidateRunner(provider)
        result = runner.run("Pikachu", tmp_path, n_candidates=3, top_k=1)
        if result.top_k:
            assert result.best == result.top_k[0].path

    def test_best_images_list(self, tmp_path):
        provider = _StubProvider()
        runner = CandidateRunner(provider)
        result = runner.run("Pikachu", tmp_path, n_candidates=4, top_k=2)
        assert result.best_images == [c.path for c in result.top_k]

    def test_all_candidates_stored(self, tmp_path):
        provider = _StubProvider()
        runner = CandidateRunner(provider)
        result = runner.run("Pikachu", tmp_path, n_candidates=5, top_k=2)
        assert len(result.all_candidates) <= 5

    def test_pokemon_name_stored(self, tmp_path):
        provider = _StubProvider()
        runner = CandidateRunner(provider)
        result = runner.run("Charizard", tmp_path, n_candidates=2, top_k=1)
        assert result.pokemon_name == "Charizard"

    def test_output_dir_created(self, tmp_path):
        provider = _StubProvider()
        runner = CandidateRunner(provider)
        out = tmp_path / "my_output"
        result = runner.run("Pikachu", out, n_candidates=2, top_k=1)
        assert out.exists()

    def test_candidates_subdir_created(self, tmp_path):
        provider = _StubProvider()
        runner = CandidateRunner(provider)
        runner.run("Pikachu", tmp_path, n_candidates=2, top_k=1)
        assert (tmp_path / "candidates").exists()

    def test_all_failures_returns_empty_top_k(self, tmp_path):
        provider = _StubProvider(fail_at={0, 1, 2, 3})
        runner = CandidateRunner(provider)
        result = runner.run("Pikachu", tmp_path, n_candidates=4, top_k=3)
        assert result.top_k == []
        assert result.best is None

    def test_extra_prompt_forwarded(self, tmp_path):
        """extra_prompt should appear in the request prompts."""
        seen_prompts = []

        class _CapturingProvider(_StubProvider):
            def generate(self, req, out):
                seen_prompts.append(req.prompt)
                return super().generate(req, out)

        runner = CandidateRunner(_CapturingProvider())
        runner.run("Pikachu", tmp_path, n_candidates=2, top_k=1, extra_prompt="UNIQUE_MARKER")
        assert all("UNIQUE_MARKER" in p for p in seen_prompts)

    def test_reference_images_passed_to_provider(self, tmp_path):
        """Valid reference images should be forwarded in the request."""
        seen_refs = []
        ref = tmp_path / "ref.png"
        ref.write_bytes(b"\x89PNG\r\n")  # minimal file

        class _RefCapture(_StubProvider):
            def generate(self, req, out):
                seen_refs.extend(req.reference_images)
                return super().generate(req, out)

        runner = CandidateRunner(_RefCapture())
        runner.run("Pikachu", tmp_path, reference_images=[ref], n_candidates=2, top_k=1)
        assert len(seen_refs) > 0
        assert all(r == ref for r in seen_refs)

    def test_nonexistent_reference_filtered_out(self, tmp_path):
        """References that don't exist should be excluded."""
        seen_refs = []
        nonexistent = tmp_path / "does_not_exist.png"

        class _RefCapture(_StubProvider):
            def generate(self, req, out):
                seen_refs.extend(req.reference_images)
                return super().generate(req, out)

        runner = CandidateRunner(_RefCapture())
        runner.run("Pikachu", tmp_path, reference_images=[nonexistent], n_candidates=2, top_k=1)
        assert nonexistent not in seen_refs

    def test_seed_incremented_per_candidate(self, tmp_path):
        """Each candidate should get a different seed."""
        seen_seeds = []

        class _SeedCapture(_StubProvider):
            def generate(self, req, out):
                seen_seeds.append(req.seed)
                return super().generate(req, out)

        runner = CandidateRunner(_SeedCapture())
        runner.run("Pikachu", tmp_path, n_candidates=3, top_k=1, seed=100)
        assert seen_seeds == [100, 101, 102]


class TestCandidateScore:
    def test_sort_order_is_descending(self):
        """Higher score should sort first."""
        a = CandidateScore(path=Path("a"), score=0.9, index=0, prompt="", provider="x")
        b = CandidateScore(path=Path("b"), score=0.1, index=1, prompt="", provider="x")
        scores = sorted([b, a])
        assert scores[0].score == 0.9  # highest first (via __lt__ reversal)

    def test_equality_by_score(self):
        a = CandidateScore(path=Path("a"), score=0.5, index=0, prompt="", provider="x")
        b = CandidateScore(path=Path("b"), score=0.5, index=1, prompt="", provider="x")
        # Neither is less than the other when scores are equal
        assert not (a < b)
        assert not (b < a)


class TestScoreImage:
    def test_returns_float(self, tmp_path):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        img = Image.new("RGB", (64, 64), color=(200, 200, 50))
        path = tmp_path / "test.png"
        img.save(str(path))
        score = _score_image(path)
        assert isinstance(score, float)

    def test_score_in_range(self, tmp_path):
        try:
            from PIL import Image
        except ImportError:
            pytest.skip("Pillow not installed")

        img = Image.new("RGB", (64, 64), color=(100, 150, 200))
        path = tmp_path / "test.png"
        img.save(str(path))
        score = _score_image(path)
        assert 0.0 <= score <= 1.0

    def test_score_missing_file_returns_neutral(self, tmp_path):
        score = _score_image(tmp_path / "nonexistent.png")
        assert score == 0.5

    def test_high_contrast_image_has_higher_score(self, tmp_path):
        """A high-contrast image should score higher on contrast metric."""
        try:
            from PIL import Image
            import numpy as np
        except ImportError:
            pytest.skip("Pillow/numpy not installed")

        # High contrast: half black, half white
        arr_high = np.zeros((128, 128, 3), dtype=np.uint8)
        arr_high[:, 64:] = 255
        img_high = Image.fromarray(arr_high)
        path_high = tmp_path / "high.png"
        img_high.save(str(path_high))

        # Low contrast: uniform grey
        arr_low = np.full((128, 128, 3), 128, dtype=np.uint8)
        img_low = Image.fromarray(arr_low)
        path_low = tmp_path / "low.png"
        img_low.save(str(path_low))

        score_high = _score_image(path_high)
        score_low = _score_image(path_low)
        assert score_high > score_low
