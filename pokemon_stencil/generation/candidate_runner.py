"""
pokemon_stencil.generation.candidate_runner — generate N candidates, keep top K.

Workflow:
    1. Build N prompt variants (camera angle × lighting × action combinations).
    2. Submit all N to the configured provider (batched or sequential).
    3. Score each output for stencil suitability.
    4. Return the top-K candidates, sorted best-first.

Scoring criteria (weighted):
    - Silhouette clarity  (0.40) — clear edges, recognisable outline
    - Connected components (0.30) — few large regions vs many tiny islands
    - Contrast             (0.30) — wide tonal range for clean K-means segmentation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from pokemon_stencil.generation.base import (
    AbstractGenerationProvider,
    GenerationRequest,
    GenerationResult,
)
from pokemon_stencil.generation.prompt_builder import PokemonPromptBuilder

logger = logging.getLogger(__name__)


@dataclass
class CandidateScore:
    """Scored generation candidate."""

    path: Path
    score: float
    index: int
    prompt: str
    provider: str

    def __lt__(self, other: "CandidateScore") -> bool:
        return self.score > other.score  # higher score = better rank


@dataclass
class CandidateRunResult:
    """Result of a full candidate generation run."""

    pokemon_name: str
    total_generated: int
    top_k: List[CandidateScore]
    all_candidates: List[CandidateScore]
    output_dir: Path

    @property
    def best(self) -> Optional[Path]:
        return self.top_k[0].path if self.top_k else None

    @property
    def best_images(self) -> List[Path]:
        return [c.path for c in self.top_k]


class CandidateRunner:
    """Generate N candidate images and return the top-K scoring ones.

    Usage::

        from pokemon_stencil.generation import get_provider, CandidateRunner

        provider = get_provider()
        runner = CandidateRunner(provider)
        result = runner.run(
            pokemon_name="Pikachu",
            output_dir=Path("outputs/pikachu_run"),
            reference_images=[Path("refs/pikachu/front.png")],
            n_candidates=8,
            top_k=3,
        )
        best_image = result.best
    """

    def __init__(
        self,
        provider: AbstractGenerationProvider,
        prompt_builder: Optional[PokemonPromptBuilder] = None,
    ) -> None:
        self.provider = provider
        self.builder = prompt_builder or PokemonPromptBuilder()

    def run(
        self,
        pokemon_name: str,
        output_dir: Path,
        reference_images: Optional[List[Path]] = None,
        n_candidates: int = 8,
        top_k: int = 3,
        seed: Optional[int] = None,
        guidance_scale: float = 3.5,
        num_steps: int = 28,
        reference_strength: float = 0.80,
        extra_prompt: str = "",
        extra: Optional[dict] = None,
    ) -> CandidateRunResult:
        """Generate *n_candidates* images, score them, return top *top_k*.

        Args:
            pokemon_name: Species name for prompt building and metadata.
            output_dir: Root output directory; candidates go in candidates/ subdir.
            reference_images: Optional reference images for conditioning.
            n_candidates: How many images to generate.
            top_k: How many best images to return.
            seed: Base seed (incremented per candidate for reproducibility).
            guidance_scale: Prompt guidance strength.
            num_steps: Inference steps per image.
            reference_strength: How closely to follow the reference (0–1).
            extra_prompt: Additional prompt text appended to each variant.
            extra: Extra kwargs forwarded to the provider (e.g. lora_path).

        Returns:
            CandidateRunResult with .top_k sorted best-first.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        candidates_dir = output_dir / "candidates"
        candidates_dir.mkdir(exist_ok=True)

        # ── Determine provider type for prompt style ──────────────────────────
        provider_name = self.provider.name
        prompt_style = "flux" if provider_name == "fal" else "sdxl"

        # ── Build diversity prompts ───────────────────────────────────────────
        prompts = self.builder.diversity_variants(
            pokemon_name, n_candidates, provider=prompt_style
        )
        if extra_prompt:
            prompts = [f"{p}, {extra_prompt}" for p in prompts]

        negative = self.builder.negative_prompt()
        valid_refs = [r for r in (reference_images or []) if r.exists()]

        # ── Generate candidates ───────────────────────────────────────────────
        all_results: List[GenerationResult] = []
        for i, prompt in enumerate(prompts):
            candidate_seed = (seed + i) if seed is not None else None
            req = GenerationRequest(
                prompt=prompt,
                negative_prompt=negative,
                reference_images=valid_refs,
                num_images=1,
                seed=candidate_seed,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                reference_strength=reference_strength,
                extra={
                    "pokemon_name": pokemon_name,
                    **(extra or {}),
                },
            )
            sub_dir = candidates_dir / f"candidate_{i:03d}"
            result = self.provider.generate(req, sub_dir)
            all_results.append(result)
            if result.ok:
                logger.info(
                    "Candidate %d/%d generated: %s", i + 1, n_candidates, result.images[0].name
                )
            else:
                logger.warning("Candidate %d/%d failed to generate.", i + 1, n_candidates)

        # ── Collect and score ─────────────────────────────────────────────────
        all_candidates: List[CandidateScore] = []
        for i, (result, prompt) in enumerate(zip(all_results, prompts)):
            for img_path in result.images:
                score = _score_image(img_path)
                all_candidates.append(
                    CandidateScore(
                        path=img_path,
                        score=score,
                        index=i,
                        prompt=prompt,
                        provider=result.provider,
                    )
                )

        # ── Sort by score (best first) ────────────────────────────────────────
        all_candidates.sort()
        top_candidates = all_candidates[:top_k]

        # ── Copy top-K to output root for easy access ─────────────────────────
        for rank, cand in enumerate(top_candidates):
            dest = output_dir / f"top_{rank + 1:02d}_{cand.path.name}"
            try:
                import shutil
                shutil.copy2(cand.path, dest)
                logger.info("Top-%d image saved to %s (score=%.3f)", rank + 1, dest, cand.score)
            except Exception as exc:
                logger.warning("Failed to copy top candidate: %s", exc)

        logger.info(
            "CandidateRunner complete: %d generated, %d scored, top-%d selected.",
            len(all_results),
            len(all_candidates),
            len(top_candidates),
        )

        return CandidateRunResult(
            pokemon_name=pokemon_name,
            total_generated=len(all_candidates),
            top_k=top_candidates,
            all_candidates=all_candidates,
            output_dir=output_dir,
        )


# ── Scoring ────────────────────────────────────────────────────────────────────

def _score_image(image_path: Path) -> float:
    """Score a generated image for stencil suitability.

    Returns a value in [0, 1].  Higher = better.

    Scoring components:
        - Silhouette clarity (40%)  — Canny edge pixel density
        - Connected components (30%) — penalises fragmented small islands
        - Contrast (30%)             — grayscale standard deviation

    Returns 0.5 as a neutral score if the image cannot be read.
    """
    try:
        import numpy as np
        import cv2
        from PIL import Image

        img = Image.open(image_path).convert("RGB")
        arr = np.array(img)

        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

        # ── 1. Silhouette clarity — Canny edge density ────────────────────────
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(edges.sum() / 255) / (gray.shape[0] * gray.shape[1])
        # Target: ~10% of pixels are edges; penalise too few or too many
        clarity_score = 1.0 - abs(edge_density - 0.10) / 0.10
        clarity_score = max(0.0, min(1.0, clarity_score))

        # ── 2. Connected components — penalise fragmentation ─────────────────
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresh)
        small_count = sum(
            1 for i in range(1, num_labels) if stats[i, cv2.CC_STAT_AREA] < 500
        )
        component_score = max(0.0, 1.0 - small_count / max(num_labels, 1) * 2)

        # ── 3. Contrast — grayscale standard deviation ────────────────────────
        contrast_score = min(1.0, float(gray.std()) / 80.0)

        score = 0.40 * clarity_score + 0.30 * component_score + 0.30 * contrast_score
        return round(score, 4)

    except Exception as exc:
        logger.debug("Scoring failed for %s: %s — using neutral 0.5", image_path.name, exc)
        return 0.5
