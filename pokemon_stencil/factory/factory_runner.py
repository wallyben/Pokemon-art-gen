"""
Art factory runner (Phase 4).

Automates multi-candidate generation, quality scoring, top-K selection, and
stencil SVG export for a single Pokémon.

Workflow
--------
1. Generate *count* candidate artworks (style transfer + simplification
   applied to each).
2. Score every candidate for stencil suitability using ``StencilScorer``.
3. Select the top *top_k* candidates by score.
4. Run the full stencil pipeline (segmentation → layer building → vector
   tracing → SVG export) on the selected candidates.
5. Return a ``FactoryResult`` summarising all outputs.

Parallelism
-----------
When ``config.factory.workers > 1`` the generation + preprocessing step is
distributed across a multiprocessing Pool using the module-level
``_candidate_worker`` function (required for pickling on macOS/Windows spawn
contexts).  Steps 4-5 are always sequential because the SD model cache
cannot be safely shared across processes.
"""

from __future__ import annotations

import logging
import multiprocessing
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from pokemon_stencil.config import PipelineConfig
from pokemon_stencil.factory.diversity import DesignDiversityFilter
from pokemon_stencil.factory.scoring import StencilScorer
from pokemon_stencil.image_gen.composition_guidance import (
    CompositionMode,
    generate_composition_map,
)
from pokemon_stencil.image_gen.generator import PokemonImageGenerator
from pokemon_stencil.image_gen.prompt_engine import (
    CAMERA_ANGLE_TOKENS,
    LIGHTING_TOKENS,
)
from pokemon_stencil.image_gen.style_transfer import StencilStyleTransfer
from pokemon_stencil.image_proc.loader import ReferenceImageLoader
from pokemon_stencil.image_proc.segmenter import ColourLayer, ColourSegmenter
from pokemon_stencil.image_proc.simplifier import ImageSimplifier
from pokemon_stencil.pipeline import PipelineResult
from pokemon_stencil.stencil.layer_builder import StencilLayerBuilder
from pokemon_stencil.stencil.svg_exporter import SVGExporter
from pokemon_stencil.vector.path_builder import StencilPathBuilder
from pokemon_stencil.vector.tracer import VectorTracer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data containers
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CandidateResult:
    """Outcome of a single candidate generation attempt."""

    index: int
    """Zero-based candidate index within the factory run."""

    score: float
    """Stencil-suitability score in [0, 1]."""

    simplified_path: Optional[Path] = None
    """Path to the saved simplified image, or ``None`` on failure."""

    error: Optional[str] = None
    """Traceback string if generation failed, otherwise ``None``."""


@dataclass
class FactoryResult:
    """Complete summary of a factory run."""

    pokemon_name: str
    """Pokemon that was processed."""

    total_candidates: int
    """Number of candidates that were generated (or attempted)."""

    top_k: int
    """Maximum number of candidates selected for stencil conversion."""

    ranked: List[Tuple[float, str]] = field(default_factory=list)
    """``(score, run_name)`` pairs for the selected candidates, best first."""

    pipeline_results: List[PipelineResult] = field(default_factory=list)
    """One ``PipelineResult`` per successfully converted candidate."""

    errors: List[str] = field(default_factory=list)
    """Error messages from failed generation or stencil conversion steps."""

    @property
    def succeeded(self) -> int:
        """Number of candidates successfully converted to SVG packs."""
        return len(self.pipeline_results)

    @property
    def failed(self) -> int:
        """Total number of errors (generation + stencil failures)."""
        return len(self.errors)


# ─────────────────────────────────────────────────────────────────────────────
# Module-level worker (must be at module scope for pickle / spawn)
# ─────────────────────────────────────────────────────────────────────────────

def _candidate_worker(args: Dict) -> Dict:
    """
    Generate, preprocess, and score one candidate image in a worker process.

    This function must remain at module level so that Python's ``spawn``
    multiprocessing context can pickle it.

    Args:
        args: Dict with keys:
            ``config_dict``      – serialised config (see ``_serialise_config``).
            ``pokemon_name``     – str.
            ``prompt_extra``     – str.
            ``candidate_index``  – int.
            ``base_seed``        – Optional[int].
            ``candidate_dir``    – str path for saving the simplified image.

    Returns:
        Dict with keys ``index``, ``score``, ``simplified_path``, ``error``.
    """
    idx: int = args.get("candidate_index", -1)
    try:
        from pokemon_stencil.config import GenerationConfig, ProcessingConfig
        from pokemon_stencil.factory.scoring import StencilScorer
        from pokemon_stencil.image_gen.generator import PokemonImageGenerator
        from pokemon_stencil.image_gen.style_transfer import StencilStyleTransfer
        from pokemon_stencil.image_proc.simplifier import ImageSimplifier

        gen_cfg, proc_cfg = _deserialise_config(args["config_dict"])
        if args.get("base_seed") is not None:
            gen_cfg.seed = int(args["base_seed"]) + idx

        generator = PokemonImageGenerator(gen_cfg)
        images = generator.generate(
            pokemon_name=args["pokemon_name"],
            prompt_extras=args.get("prompt_extra", ""),
            num_images=1,
            seed=gen_cfg.seed,
        )
        if not images:
            return {
                "index": idx, "score": 0.0,
                "simplified_path": None,
                "error": "Generation returned no images.",
            }

        styled = StencilStyleTransfer(proc_cfg).apply(images[0])
        simplified = ImageSimplifier(proc_cfg).simplify(styled)
        score = StencilScorer().score(simplified)

        out_path = Path(args["candidate_dir"]) / f"candidate_{idx:03d}.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        simplified.save(str(out_path))

        return {
            "index": idx,
            "score": score,
            "simplified_path": str(out_path),
            "error": None,
        }

    except Exception:
        return {
            "index": idx,
            "score": 0.0,
            "simplified_path": None,
            "error": traceback.format_exc(),
        }


# ─────────────────────────────────────────────────────────────────────────────
# FactoryRunner
# ─────────────────────────────────────────────────────────────────────────────

class FactoryRunner:
    """
    Orchestrates the art factory pipeline for a single Pokémon.

    Args:
        config: ``PipelineConfig`` controlling all stage parameters.
            Factory-specific parameters live in ``config.factory``.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._scorer = StencilScorer()
        self._style_transfer = StencilStyleTransfer(config.processing)
        self._simplifier = ImageSimplifier(config.processing)
        self._diversity_filter = DesignDiversityFilter(
            threshold=config.factory.diversity_threshold
        )
        # Generator is lazy-loaded to avoid 4 GB model load when mocked.
        self._generator: Optional[PokemonImageGenerator] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        pokemon_name: str,
        reference_dir: Optional[Path] = None,
        prompt_extra: str = "",
    ) -> FactoryResult:
        """
        Execute the full factory workflow for *pokemon_name*.

        Args:
            pokemon_name: Canonical Pokémon name, e.g. ``"Pikachu"``.
            reference_dir: Optional directory of reference images used to
                           guide generation (or as source when
                           ``config.skip_generation`` is ``True``).
            prompt_extra: Additional text appended to the SD prompt.

        Returns:
            ``FactoryResult`` summarising every output artefact.
        """
        count = self.config.factory.count
        top_k = min(self.config.factory.top_k, count)
        workers = self.config.factory.workers

        logger.info(
            "FactoryRunner.run | pokemon=%s | count=%d | top_k=%d | workers=%d",
            pokemon_name, count, top_k, workers,
        )

        # ── Step 0: Auto-fetch references if enabled and directory is empty ───
        if (
            self.config.generation.auto_fetch_references
            and reference_dir is not None
            and not _has_images(reference_dir)
        ):
            logger.info(
                "auto_fetch_references=True and %s is empty – fetching now.",
                reference_dir,
            )
            from pokemon_stencil.data.reference_fetcher import ReferenceImageFetcher
            fetcher = ReferenceImageFetcher(
                max_images=self.config.generation.max_reference_images
            )
            fetcher.fetch_references(pokemon_name, reference_dir)

        # ── Steps 1-2: Generate and score all candidates ──────────────────────
        if workers > 1:
            candidates = self._run_parallel(pokemon_name, count, prompt_extra)
        else:
            candidates = self._run_sequential(
                pokemon_name, count, reference_dir, prompt_extra
            )

        valid = [c for c in candidates if c.error is None]
        gen_errors = [c.error for c in candidates if c.error is not None]

        # ── Step 3: Select top K with diversity filtering ─────────────────────
        selected = self._diversity_filter.select_diverse(valid, top_k)
        logger.info(
            "Selected %d/%d valid candidates for stencil conversion "
            "(diversity_threshold=%.2f).",
            len(selected),
            len(valid),
            self.config.factory.diversity_threshold,
        )

        # ── Steps 4-5: Stencil pipeline on selected candidates ────────────────
        result = FactoryResult(
            pokemon_name=pokemon_name,
            total_candidates=count,
            top_k=top_k,
        )
        result.errors.extend([e for e in gen_errors if e])

        safe_name = pokemon_name.lower().replace(" ", "_")
        for rank, candidate in enumerate(selected):
            run_name = f"{safe_name}_{rank:02d}"
            logger.info(
                "Stencil pipeline | rank=%d | candidate=%d | score=%.3f | run=%s",
                rank, candidate.index, candidate.score, run_name,
            )
            try:
                pipe_result = self._run_stencil_on_candidate(candidate, run_name)
                result.pipeline_results.append(pipe_result)
                result.ranked.append((candidate.score, run_name))
            except Exception as exc:
                msg = (
                    f"Stencil pipeline failed for candidate "
                    f"{candidate.index}: {exc}"
                )
                logger.error(msg, exc_info=True)
                result.errors.append(msg)

        return result

    # ------------------------------------------------------------------
    # Sequential generation path
    # ------------------------------------------------------------------

    def _run_sequential(
        self,
        pokemon_name: str,
        count: int,
        reference_dir: Optional[Path],
        prompt_extra: str,
    ) -> List[CandidateResult]:
        """Generate *count* candidates one-by-one in the current process.

        When ``config.generation.use_composition_guidance`` is ``True`` and a
        *reference_dir* is provided, a composition map is generated once from
        the first reference image and shared across all candidates.
        """
        # ── Stage 1b: Composition guidance map ────────────────────────────────
        composition_map: Optional[np.ndarray] = None
        if (
            self.config.generation.use_composition_guidance
            and not self.config.skip_generation
            and reference_dir is not None
        ):
            loader = ReferenceImageLoader(
                target_size=self.config.processing.output_size
            )
            ref_images = loader.load_from_directory(reference_dir)
            if ref_images:
                composition_map = generate_composition_map(ref_images[0])
                logger.info(
                    "Generated composition map from reference image in %s.",
                    reference_dir,
                )

        results: List[CandidateResult] = []
        for i in range(count):
            try:
                simplified = self._generate_one(
                    pokemon_name, i, reference_dir, prompt_extra, composition_map
                )
                score = self._scorer.score(simplified)
                cand_path = self._candidate_dir() / f"candidate_{i:03d}.png"
                cand_path.parent.mkdir(parents=True, exist_ok=True)
                simplified.save(str(cand_path))
                results.append(
                    CandidateResult(index=i, score=score, simplified_path=cand_path)
                )
                logger.debug("Candidate %d scored %.3f.", i, score)
            except Exception as exc:
                err = traceback.format_exc()
                logger.warning("Candidate %d failed: %s", i, exc)
                results.append(CandidateResult(index=i, score=0.0, error=err))
        return results

    # ------------------------------------------------------------------
    # Parallel generation path
    # ------------------------------------------------------------------

    def _run_parallel(
        self,
        pokemon_name: str,
        count: int,
        prompt_extra: str,
    ) -> List[CandidateResult]:
        """Distribute candidate generation across a multiprocessing Pool."""
        workers = self.config.factory.workers
        config_dict = _serialise_config(self.config)
        candidate_dir = str(self._candidate_dir())

        args_list = [
            {
                "config_dict": config_dict,
                "pokemon_name": pokemon_name,
                "prompt_extra": prompt_extra,
                "candidate_index": i,
                "base_seed": self.config.generation.seed,
                "candidate_dir": candidate_dir,
            }
            for i in range(count)
        ]

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=workers) as pool:
            raw = pool.map(_candidate_worker, args_list)

        return [
            CandidateResult(
                index=r["index"],
                score=r["score"],
                simplified_path=Path(r["simplified_path"]) if r["simplified_path"] else None,
                error=r["error"],
            )
            for r in raw
        ]

    # ------------------------------------------------------------------
    # Single-candidate generation helper
    # ------------------------------------------------------------------

    def _generate_one(
        self,
        pokemon_name: str,
        candidate_index: int,
        reference_dir: Optional[Path],
        prompt_extra: str,
        composition_map: Optional[np.ndarray] = None,
    ) -> Image.Image:
        """
        Generate, style-transfer, and simplify one candidate image.

        When ``config.skip_generation`` is ``True`` the reference images in
        *reference_dir* are used directly as source material and
        *composition_map* is ignored.

        Diversity tokens (camera angle + lighting) are cycled per candidate
        index so the factory produces a visually varied set of designs.

        Args:
            pokemon_name: Canonical Pokémon name.
            candidate_index: Zero-based index for seed offset.
            reference_dir: Directory of reference images (required when
                ``skip_generation=True``).
            prompt_extra: Additional prompt text.
            composition_map: Optional structural guide from
                :func:`~pokemon_stencil.image_gen.composition_guidance.generate_composition_map`.
                Passed to the generator when SD generation is active and
                ``config.generation.use_composition_guidance`` is ``True``.
        """
        if self.config.skip_generation:
            if reference_dir is None:
                raise ValueError(
                    "skip_generation=True requires reference_dir to be provided."
                )
            loader = ReferenceImageLoader(
                target_size=self.config.processing.output_size
            )
            images = loader.load_from_directory(reference_dir)
            if not images:
                raise ValueError(f"No reference images found in {reference_dir}.")
            img = images[candidate_index % len(images)]
        else:
            if self._generator is None:
                self._generator = PokemonImageGenerator(self.config.generation)

            # ── Generation diversity ───────────────────────────────────────────
            seed: Optional[int] = None
            if self.config.generation.seed is not None:
                seed = self.config.generation.seed + candidate_index

            camera_angle = CAMERA_ANGLE_TOKENS[
                candidate_index % len(CAMERA_ANGLE_TOKENS)
            ]
            lighting = LIGHTING_TOKENS[
                candidate_index % len(LIGHTING_TOKENS)
            ]

            imgs = self._generator.generate(
                pokemon_name=pokemon_name,
                prompt_extras=prompt_extra,
                num_images=1,
                seed=seed,
                composition_map=composition_map,
                camera_angle=camera_angle,
                lighting=lighting,
            )
            if not imgs:
                raise RuntimeError(
                    f"Generation returned no images for candidate {candidate_index}."
                )
            img = imgs[0]
            logger.debug(
                "Candidate %d: seed=%s | angle='%s' | lighting='%s'",
                candidate_index,
                seed,
                camera_angle,
                lighting,
            )

        styled = self._style_transfer.apply(img)
        return self._simplifier.simplify(styled)

    # ------------------------------------------------------------------
    # Stencil conversion for a selected candidate
    # ------------------------------------------------------------------

    def _run_stencil_on_candidate(
        self,
        candidate: CandidateResult,
        run_name: str,
    ) -> PipelineResult:
        """
        Run stages 5-9 (segmentation → SVG export) on one already-simplified
        candidate image.

        Reuses the existing Phase 2-3 stage objects directly to avoid
        repeating style transfer and simplification.

        Args:
            candidate: Selected candidate with a valid ``simplified_path``.
            run_name: Output directory identifier, e.g. ``"pikachu_00"``.

        Returns:
            ``PipelineResult`` with all written artefact paths.
        """
        self.config.output.ensure_run_dirs(run_name)
        result = PipelineResult(run_name=run_name)

        simplified = Image.open(candidate.simplified_path).convert("RGB")
        result.simplified_paths.append(candidate.simplified_path)

        # ── Stage 5: Colour segmentation ──────────────────────────────────────
        segmenter = ColourSegmenter(self.config.processing)
        layers: List[ColourLayer] = segmenter.segment(simplified)
        logger.info("%d colour layer(s) for %s.", len(layers), run_name)

        masks_dir = self.config.output.masks_path(run_name)
        for layer in layers:
            mask_img = Image.fromarray(layer.mask)
            mask_path = masks_dir / f"img000_layer{layer.index:02d}.png"
            mask_img.save(mask_path)
            result.mask_paths.append(mask_path)

        # ── Stage 6: Stencil-safety layering ──────────────────────────────────
        image_size = self.config.processing.output_size
        layer_builder = StencilLayerBuilder(self.config.stencil)
        safe_layers = layer_builder.build(layers, image_size)

        # ── Stages 7-8: Vector tracing ────────────────────────────────────────
        tracer = VectorTracer(self.config.vector)
        path_builder = StencilPathBuilder(tracer)
        layer_paths = path_builder.build_paths(safe_layers)

        # ── Stage 9: SVG export ───────────────────────────────────────────────
        svg_exporter = SVGExporter(self.config.vector, self.config.stencil)
        svg_dir = self.config.output.svg_path(run_name)
        svg_paths = svg_exporter.export(layer_paths, svg_dir, "img000", image_size)
        result.svg_paths.extend(svg_paths)

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _candidate_dir(self) -> Path:
        """Temporary directory for candidate simplified images."""
        return self.config.output.base_dir / "_factory_candidates"


# ─────────────────────────────────────────────────────────────────────────────
# Config serialisation for cross-process passing
# ─────────────────────────────────────────────────────────────────────────────

def _serialise_config(config: PipelineConfig) -> Dict:
    """
    Serialise the generation and processing sub-configs to a plain dict.

    Includes all fields needed by the SDXL worker function.
    """
    gen = config.generation
    proc = config.processing
    return {
        "generation": {
            # SDXL base model
            "model_local_path": str(gen.model_local_path),
            "model_hub_id": gen.model_hub_id,
            # Legacy paths
            "legacy_model_local_path": str(gen.legacy_model_local_path),
            "legacy_model_hub_id": gen.legacy_model_hub_id,
            # ControlNet (SDXL-compatible)
            "controlnet_openpose_local_path": str(gen.controlnet_openpose_local_path),
            "controlnet_openpose_hub_id": gen.controlnet_openpose_hub_id,
            "controlnet_canny_local_path": str(gen.controlnet_canny_local_path),
            "controlnet_canny_hub_id": gen.controlnet_canny_hub_id,
            "use_controlnet": gen.use_controlnet,
            "controlnet_openpose_scale": gen.controlnet_openpose_scale,
            "controlnet_canny_scale": gen.controlnet_canny_scale,
            # IP-Adapter
            "ip_adapter_local_path": str(gen.ip_adapter_local_path),
            "ip_adapter_hub_id": gen.ip_adapter_hub_id,
            "use_ip_adapter": gen.use_ip_adapter,
            "ip_adapter_scale": gen.ip_adapter_scale,
            "ip_adapter_min_images": gen.ip_adapter_min_images,
            "ip_adapter_max_images": gen.ip_adapter_max_images,
            # LoRA
            "lora_path": str(gen.lora_path) if gen.lora_path else None,
            "lora_dir": str(gen.lora_dir),
            "lora_scale": gen.lora_scale,
            # Reference encoding (legacy img2img)
            "use_reference_encoding": gen.use_reference_encoding,
            "reference_strength": gen.reference_strength,
            # Generation parameters
            "num_inference_steps": gen.num_inference_steps,
            "guidance_scale": gen.guidance_scale,
            "width": gen.width,
            "height": gen.height,
            "seed": gen.seed,
            "torch_dtype": gen.torch_dtype,
            "device": gen.device,
            # Prompt
            "style_suffix": gen.style_suffix,
            "negative_prompt": gen.negative_prompt,
            "optimise_prompt": gen.optimise_prompt,
            # Composition
            "use_composition_guidance": gen.use_composition_guidance,
            "composition_strength": gen.composition_strength,
        },
        "processing": {
            "bilateral_d": proc.bilateral_d,
            "bilateral_sigma_color": proc.bilateral_sigma_color,
            "bilateral_sigma_space": proc.bilateral_sigma_space,
            "bilateral_passes": proc.bilateral_passes,
            "n_colors": proc.n_colors,
            "min_region_area": proc.min_region_area,
            "morph_kernel_size": proc.morph_kernel_size,
            "canny_low": proc.canny_low,
            "canny_high": proc.canny_high,
            "output_size": list(proc.output_size),
            "adaptive_edge_thinning": proc.adaptive_edge_thinning,
            "merge_small_components": proc.merge_small_components,
            "min_island_area": proc.min_island_area,
            "merge_adjacent_regions": proc.merge_adjacent_regions,
            "region_merge_threshold": proc.region_merge_threshold,
        },
    }


def _deserialise_config(cd: Dict):
    """
    Reconstruct ``GenerationConfig`` and ``ProcessingConfig`` from a dict
    produced by ``_serialise_config``.
    """
    from pathlib import Path

    from pokemon_stencil.config import GenerationConfig, ProcessingConfig

    gen_d = dict(cd["generation"])

    # Convert path strings to Path objects.
    for path_key in (
        "model_local_path",
        "legacy_model_local_path",
        "controlnet_openpose_local_path",
        "controlnet_canny_local_path",
        "ip_adapter_local_path",
        "lora_dir",
    ):
        if path_key in gen_d:
            gen_d[path_key] = Path(gen_d[path_key])

    gen_d["lora_path"] = Path(gen_d["lora_path"]) if gen_d.get("lora_path") else None

    gen_cfg = GenerationConfig(**gen_d)

    proc_d = dict(cd["processing"])
    proc_d["output_size"] = tuple(proc_d["output_size"])
    proc_cfg = ProcessingConfig(**proc_d)

    return gen_cfg, proc_cfg


# ─────────────────────────────────────────────────────────────────────────────
# Misc helpers
# ─────────────────────────────────────────────────────────────────────────────

def _has_images(directory: Path) -> bool:
    """Return ``True`` if *directory* contains at least one image file."""
    if not directory.is_dir():
        return False
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return any(
        p.suffix.lower() in image_suffixes
        for p in directory.iterdir()
        if p.is_file()
    )
