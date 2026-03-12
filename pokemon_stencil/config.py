"""
Configuration management for the Pokemon Stencil Art Factory.

All tunable parameters across every pipeline stage are defined here as
Python dataclasses.  Modules accept config objects rather than raw
literals, making the system easy to reconfigure from the CLI or tests
without touching internal logic.

Directory conventions (applied corrections)
-------------------------------------------
- All runtime outputs live under ``outputs/`` (not ``output/``).
- Pre-downloaded Stable Diffusion 1.5 weights are expected at
  ``models/sd15/`` so the pipeline can run fully offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple


# ── Model paths ───────────────────────────────────────────────────────────────

#: Default local path for the SD 1.5 model weights.
#: The pipeline falls back to the HuggingFace Hub ID when this path is absent.
DEFAULT_MODEL_LOCAL_PATH: Path = Path("models/sd15")
DEFAULT_MODEL_HUB_ID: str = "runwayml/stable-diffusion-v1-5"


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationConfig:
    """Parameters for Stable Diffusion 1.5 image generation.

    The model is loaded from ``model_local_path`` when that directory exists,
    otherwise it is downloaded from ``model_hub_id`` (requires internet).
    """

    model_local_path: Path = field(default_factory=lambda: DEFAULT_MODEL_LOCAL_PATH)
    model_hub_id: str = DEFAULT_MODEL_HUB_ID

    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512

    #: Optional fixed seed for reproducible outputs.
    seed: Optional[int] = None

    #: CPU-safe dtype.  float16 is not reliable on most CPU torch builds.
    torch_dtype: str = "float32"
    device: str = "cpu"

    #: Prompt suffix appended to every generation to push output toward the
    #: flat, bold aesthetics required for clean stencil cutting.
    style_suffix: str = (
        "flat color illustration, bold outlines, minimal detail, "
        "stencil art style, clean shapes, white background"
    )

    negative_prompt: str = (
        "photorealistic, gradient, shadow, complex texture, noise, "
        "blurry, watermark, signature, multiple characters"
    )

    #: When True, reference images are used to condition generation via the
    #: ReferenceEncoder, improving character identity preservation.
    use_reference_conditioning: bool = True

    #: Strength of reference conditioning in image-to-image generation.
    #: 0.0 = fully reproduce reference; 1.0 = fully ignore reference.
    reference_strength: float = 0.6

    @property
    def model_id(self) -> str | Path:
        """Return local path if it exists, otherwise the Hub model ID."""
        if self.model_local_path.exists():
            return self.model_local_path
        return self.model_hub_id


@dataclass
class ProcessingConfig:
    """Parameters for image simplification and colour segmentation."""

    # ── Bilateral filter ──────────────────────────────────────────────────────
    #: Filter diameter.  Larger values = more smoothing, slower.
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    #: Number of sequential bilateral passes.
    bilateral_passes: int = 3

    # ── Colour quantisation ───────────────────────────────────────────────────
    #: Number of K-Means clusters == number of stencil colour layers.
    n_colors: int = 6

    #: Layers with fewer pixels than this are discarded as noise.
    min_region_area: int = 500

    # ── Morphological cleanup ─────────────────────────────────────────────────
    morph_kernel_size: int = 5

    #: Kernel size for morphological opening (removes thin slivers/noise).
    morph_open_kernel: int = 3

    #: Kernel size for morphological closing (fills small holes in masks).
    morph_close_kernel: int = 3

    # ── Edge detection (informational; not used in stencil paths) ─────────────
    canny_low: int = 50
    canny_high: int = 150

    # ── Image dimensions ──────────────────────────────────────────────────────
    #: All images are resized to this before processing.
    output_size: Tuple[int, int] = (512, 512)


@dataclass
class VectorConfig:
    """Parameters for bitmap-to-vector tracing via potrace."""

    #: Ignore speckles smaller than this many pixels (potrace turdsize).
    turdsize: int = 10

    #: Corner rounding threshold: 0 = sharp corners, 1.333 = fully rounded.
    alphamax: float = 0.8

    #: Bezier curve optimisation tolerance.
    opttolerance: float = 0.2

    #: Paths shorter than this (in SVG user-units / pixels) are discarded.
    min_path_length: float = 20.0

    # ── SVG canvas ────────────────────────────────────────────────────────────
    #: Cricut standard mat: 12 × 12 inches = 304.8 × 304.8 mm.
    canvas_width_mm: float = 304.8
    canvas_height_mm: float = 304.8

    #: DPI used when converting pixel coordinates to millimetres.
    dpi: float = 96.0


@dataclass
class StencilConfig:
    """Parameters for stencil-safety processing and layer construction."""

    #: Minimum bridge width (px) connecting floating islands to the frame.
    bridge_width: int = 8

    #: Minimum feature width in pixels.  Regions thinner than this are dilated.
    min_cut_width_px: int = 3

    #: Bridge width in pixels for improved island-to-region bridging.
    bridge_width_px: int = 3

    #: Stroke width for the outline layer in pixels at output resolution.
    outline_stroke_width: float = 2.0

    #: Minimum cuttable feature width in mm (Cricut can cut ~0.5 mm).
    min_cut_width_mm: float = 0.8

    #: Stamp registration circles onto each layer for multi-layer alignment.
    add_registration_marks: bool = True

    #: Radius of registration mark circles in mm.
    reg_mark_radius_mm: float = 3.0

    #: Clear margin around the design in mm.
    padding_mm: float = 10.0


@dataclass
class OutputConfig:
    """Output path configuration.

    All runtime artefacts (images, masks, SVGs) are written under
    ``base_dir / run_name / <sub-dir>``.
    """

    #: Root output directory.  Standardised to ``outputs/``.
    base_dir: Path = field(default_factory=lambda: Path("outputs"))

    images_dir: str = "images"
    simplified_dir: str = "simplified"
    masks_dir: str = "masks"
    svg_dir: str = "svg"

    def run_root(self, run_name: str) -> Path:
        """Return the root directory for a named run."""
        return self.base_dir / run_name

    def images_path(self, run_name: str) -> Path:
        """Directory for raw generated / reference images."""
        return self.base_dir / run_name / self.images_dir

    def simplified_path(self, run_name: str) -> Path:
        """Directory for simplified (post-processed) images."""
        return self.base_dir / run_name / self.simplified_dir

    def masks_path(self, run_name: str) -> Path:
        """Directory for per-layer binary mask PNGs."""
        return self.base_dir / run_name / self.masks_dir

    def svg_path(self, run_name: str) -> Path:
        """Directory for exported SVG stencil files."""
        return self.base_dir / run_name / self.svg_dir

    def ensure_run_dirs(self, run_name: str) -> None:
        """Create all output sub-directories for *run_name* if absent."""
        for path in (
            self.images_path(run_name),
            self.simplified_path(run_name),
            self.masks_path(run_name),
            self.svg_path(run_name),
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class BatchConfig:
    """Parameters specific to batch-factory generation."""

    #: Number of designs to produce per Pokemon entry.
    count: int = 3

    #: Number of worker processes for parallel generation.
    #: 1 = sequential (safe on low-RAM machines); >1 uses multiprocessing.
    workers: int = 1


@dataclass
class FactoryConfig:
    """Parameters for art factory mode (Phase 4).

    The factory generates *count* candidate artworks, scores them for
    stencil suitability, selects the top *top_k*, and converts only those
    to full stencil SVG packs.
    """

    #: Total number of candidate designs to generate.
    count: int = 10

    #: Number of top-scoring candidates to convert to stencil SVGs.
    top_k: int = 3

    #: Parallel worker processes for candidate generation.
    #: 1 = sequential (safe on low-RAM machines); >1 uses multiprocessing.
    workers: int = 1

    #: Minimum score a candidate must achieve to be considered for stencil
    #: conversion.  Candidates below this threshold are automatically discarded.
    quality_threshold: float = 0.35


# ─────────────────────────────────────────────────────────────────────────────
# Top-level aggregate
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    """Top-level configuration aggregating all pipeline sub-configs.

    Pass an instance of this class to ``Pipeline`` or ``BatchRunner``.
    """

    generation: GenerationConfig = field(default_factory=GenerationConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)
    stencil: StencilConfig = field(default_factory=StencilConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    factory: FactoryConfig = field(default_factory=FactoryConfig)

    #: When True the SD generation stage is skipped and reference images are
    #: used directly as the source for simplification and segmentation.
    skip_generation: bool = False
