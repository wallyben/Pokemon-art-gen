"""
Pipeline orchestrator for the Pokemon Stencil Art Factory.

``Pipeline`` is the single integration point that wires every processing
stage together in the correct order.  Each public method corresponds to
one entry path from the CLI:

- ``run()``             – full path: optional SD generation → process → export
- ``run_from_image()``  – skip generation; start from an existing image file

Both methods return a ``PipelineResult`` dataclass describing where every
output artefact was written.

Stage execution order
---------------------
1.  Reference loading       (image_proc.loader)
2.  Image generation        (image_gen.generator)   ← skipped when skip_generation
3.  Style transfer          (image_gen.style_transfer)
4.  Image simplification    (image_proc.simplifier)
5.  Colour segmentation     (image_proc.segmenter)
6.  Stencil-safety layering (stencil.layer_builder)   ← Phase 3
7.  Vector tracing          (vector.tracer)            ← Phase 3
8.  Path building           (vector.path_builder)      ← Phase 3
9.  SVG export              (stencil.svg_exporter)     ← Phase 3

Stages 6-9 are marked as Phase 3 stubs in this release; they are called
through a clearly documented interface so that Phase 3 can slot in without
touching this file.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from PIL import Image

from pokemon_stencil.config import PipelineConfig
from pokemon_stencil.image_gen.generator import PokemonImageGenerator
from pokemon_stencil.image_gen.style_transfer import StencilStyleTransfer
from pokemon_stencil.image_proc.loader import ReferenceImageLoader
from pokemon_stencil.image_proc.segmenter import ColourLayer, ColourSegmenter
from pokemon_stencil.image_proc.simplifier import ImageSimplifier
from pokemon_stencil.stencil.layer_builder import StencilLayerBuilder
from pokemon_stencil.stencil.svg_exporter import SVGExporter
from pokemon_stencil.vector.path_builder import StencilPathBuilder
from pokemon_stencil.vector.tracer import VectorTracer

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineResult:
    """Paths to every artefact produced by a single pipeline run.

    Fields that are not yet produced (Phase 3 stages) are empty lists.
    """

    run_name: str
    """Unique identifier for this run, e.g. ``pikachu_00``."""

    image_paths: List[Path] = field(default_factory=list)
    """Raw generated or source image files."""

    simplified_paths: List[Path] = field(default_factory=list)
    """Simplified (post-processed) image files."""

    mask_paths: List[Path] = field(default_factory=list)
    """Per-layer binary mask PNG files."""

    svg_paths: List[Path] = field(default_factory=list)
    """Exported SVG stencil files (per-layer + combined)."""

    @property
    def svg_dir(self) -> Optional[Path]:
        """Parent directory of exported SVGs, or ``None`` if none were written."""
        if self.svg_paths:
            return self.svg_paths[0].parent
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline
# ─────────────────────────────────────────────────────────────────────────────

class Pipeline:
    """
    Executes the full stencil-generation pipeline for one design.

    Args:
        config: ``PipelineConfig`` controlling all stage parameters.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

        # Instantiate stage objects once; they are stateless between runs.
        self._loader = ReferenceImageLoader(
            target_size=config.processing.output_size,
        )
        self._style_transfer = StencilStyleTransfer(config.processing)
        self._simplifier = ImageSimplifier(config.processing)
        self._segmenter = ColourSegmenter(config.processing)

        # Phase 3 stage objects.
        self._layer_builder = StencilLayerBuilder(config.stencil)
        self._tracer = VectorTracer(config.vector)
        self._path_builder = StencilPathBuilder(self._tracer)
        self._svg_exporter = SVGExporter(config.vector, config.stencil)

        # Generator is lazy-loaded to avoid pulling 4 GB of model weights
        # into RAM when generation is skipped.
        self._generator: Optional[PokemonImageGenerator] = None

    # ── Public entry points ───────────────────────────────────────────────────

    def run(
        self,
        pokemon_name: str,
        run_name: str,
        reference_dir: Optional[Path] = None,
        prompt_extra: str = "",
    ) -> PipelineResult:
        """
        Execute the full pipeline for *pokemon_name*.

        Args:
            pokemon_name: Canonical Pokemon name (e.g. ``"Pikachu"``).
            run_name: Unique run identifier used to name output directories.
            reference_dir: Optional directory of reference images.  Required
                           when ``config.skip_generation`` is ``True``.
            prompt_extra: Additional text appended to the SD prompt.

        Returns:
            ``PipelineResult`` with paths to every written artefact.

        Raises:
            ValueError: If ``skip_generation`` is ``True`` but no
                        ``reference_dir`` is provided.
        """
        logger.info("Pipeline.run | run_name=%s | pokemon=%s", run_name, pokemon_name)
        self.config.output.ensure_run_dirs(run_name)
        result = PipelineResult(run_name=run_name)

        # ── Stage 1: Load reference images ────────────────────────────────────
        ref_images: List[Image.Image] = []
        if reference_dir is not None:
            ref_images = self._loader.load_from_directory(reference_dir)
            logger.info("Loaded %d reference image(s).", len(ref_images))

        # ── Stage 2: Generate (or adopt) source image ─────────────────────────
        if self.config.skip_generation:
            if not ref_images:
                raise ValueError(
                    "skip_generation=True but no reference_dir was provided."
                )
            source_images = ref_images
            logger.info("Skipping SD generation; using reference images directly.")
        else:
            source_images = self._generate(
                pokemon_name=pokemon_name,
                ref_images=ref_images,
                run_name=run_name,
                prompt_extra=prompt_extra,
            )

        # Save source images and record paths.
        result.image_paths = self._save_images(
            source_images,
            self.config.output.images_path(run_name),
            prefix="source",
        )

        # ── Stages 3-5 (per source image) ─────────────────────────────────────
        for img_idx, source_img in enumerate(source_images):
            self._process_single_image(
                image=source_img,
                image_index=img_idx,
                run_name=run_name,
                result=result,
            )

        logger.info(
            "Pipeline complete | run_name=%s | svg_dir=%s",
            run_name,
            result.svg_dir,
        )
        return result

    def run_from_image(
        self,
        image_path: Path,
        design_name: str,
        run_name: str,
    ) -> PipelineResult:
        """
        Execute the pipeline starting from an already-existing image file.

        Generation (Stage 2) is always skipped.  The provided image goes
        directly into style transfer → simplification → segmentation.

        Args:
            image_path: Path to the input PNG/JPG.
            design_name: Human-readable name used in log messages.
            run_name: Unique run identifier for output directories.

        Returns:
            ``PipelineResult`` with paths to every written artefact.
        """
        logger.info(
            "Pipeline.run_from_image | run_name=%s | image=%s",
            run_name,
            image_path,
        )
        self.config.output.ensure_run_dirs(run_name)
        result = PipelineResult(run_name=run_name)

        source_img = self._loader.load_single(image_path)
        result.image_paths = self._save_images(
            [source_img],
            self.config.output.images_path(run_name),
            prefix="source",
        )

        self._process_single_image(
            image=source_img,
            image_index=0,
            run_name=run_name,
            result=result,
        )

        logger.info(
            "Pipeline complete | run_name=%s | svg_dir=%s",
            run_name,
            result.svg_dir,
        )
        return result

    # ── Internal stage methods ────────────────────────────────────────────────

    def _generate(
        self,
        pokemon_name: str,
        ref_images: List[Image.Image],
        run_name: str,
        prompt_extra: str,
    ) -> List[Image.Image]:
        """Stage 2: invoke Stable Diffusion to produce source images."""
        if self._generator is None:
            self._generator = PokemonImageGenerator(self.config.generation)

        logger.info("Stage 2: SD generation | pokemon=%s", pokemon_name)
        images = self._generator.generate(
            pokemon_name=pokemon_name,
            prompt_extras=prompt_extra,
            num_images=1,
            seed=self.config.generation.seed,
        )
        return images

    def _process_single_image(
        self,
        image: Image.Image,
        image_index: int,
        run_name: str,
        result: PipelineResult,
    ) -> None:
        """
        Run stages 3-9 on a single source image and append artefact paths
        to *result* in-place.

        Stages 6-9 are Phase 3 stubs; they log a placeholder message and
        return without writing files.

        Args:
            image: Source PIL Image (RGB).
            image_index: Zero-based index within this run's source images.
            run_name: Run identifier for output path construction.
            result: Mutable result object to append output paths to.
        """
        prefix = f"img{image_index:03d}"

        # ── Stage 3: Style transfer ───────────────────────────────────────────
        logger.info("Stage 3: Style transfer | index=%d", image_index)
        styled = self._style_transfer.apply(image)

        # ── Stage 4: Simplification ───────────────────────────────────────────
        logger.info("Stage 4: Simplification | index=%d", image_index)
        simplified = self._simplifier.simplify(styled)
        simplified_path = (
            self.config.output.simplified_path(run_name)
            / f"simplified_{prefix}.png"
        )
        simplified.save(simplified_path)
        result.simplified_paths.append(simplified_path)
        logger.debug("Saved simplified image: %s", simplified_path)

        # ── Stage 5: Colour segmentation ─────────────────────────────────────
        logger.info("Stage 5: Colour segmentation | index=%d", image_index)
        layers: List[ColourLayer] = self._segmenter.segment(simplified)
        logger.info("Produced %d colour layer(s).", len(layers))

        # Save per-layer masks.
        for layer in layers:
            mask_img = Image.fromarray(layer.mask)
            mask_path = (
                self.config.output.masks_path(run_name)
                / f"{prefix}_layer{layer.index:02d}.png"
            )
            mask_img.save(mask_path)
            result.mask_paths.append(mask_path)

        # ── Stage 6: Stencil-safety layering ─────────────────────────────────
        logger.info("Stage 6: Stencil safety | index=%d", image_index)
        image_size = self.config.processing.output_size
        safe_layers = self._layer_builder.build(layers, image_size)

        # ── Stages 7-8: Vector tracing ────────────────────────────────────────
        logger.info("Stage 7-8: Vector tracing | index=%d", image_index)
        layer_paths = self._path_builder.build_paths(safe_layers)

        # ── Stage 9: SVG export ───────────────────────────────────────────────
        logger.info("Stage 9: SVG export | index=%d", image_index)
        svg_dir = self.config.output.svg_path(run_name)
        svg_paths = self._svg_exporter.export(
            layer_paths, svg_dir, prefix, image_size
        )
        result.svg_paths.extend(svg_paths)

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _save_images(
        images: List[Image.Image],
        directory: Path,
        prefix: str,
    ) -> List[Path]:
        """Save PIL Images to *directory* and return their paths."""
        directory.mkdir(parents=True, exist_ok=True)
        paths: List[Path] = []
        for idx, img in enumerate(images):
            path = directory / f"{prefix}_{idx:03d}.png"
            img.save(path)
            logger.debug("Saved image: %s", path)
            paths.append(path)
        return paths
