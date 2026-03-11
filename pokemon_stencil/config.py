"""
Configuration management for the Pokemon Stencil Art Factory.

Centralizes all tunable parameters across image generation, processing,
vector tracing, stencil construction, and export stages.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


@dataclass
class GenerationConfig:
    """Parameters for Stable Diffusion image generation."""

    model_id: str = "runwayml/stable-diffusion-v1-5"
    num_inference_steps: int = 25
    guidance_scale: float = 7.5
    width: int = 512
    height: int = 512
    seed: Optional[int] = None
    # CPU-optimised: use float32 (float16 not supported on most CPU builds)
    torch_dtype: str = "float32"
    device: str = "cpu"
    # Prompt fragments appended to every generation for stencil-friendliness
    style_suffix: str = (
        "flat color illustration, bold outlines, minimal detail, "
        "stencil art style, clean shapes, white background"
    )
    negative_prompt: str = (
        "photorealistic, gradient, shadow, complex texture, noise, "
        "blurry, watermark, signature, multiple characters"
    )


@dataclass
class ProcessingConfig:
    """Parameters for image simplification and colour segmentation."""

    # Bilateral filter – keeps edges while smoothing within regions
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0

    # Number of colour passes through the bilateral filter
    bilateral_passes: int = 3

    # K-Means colour quantisation
    n_colors: int = 6  # number of stencil colour layers

    # Minimum pixel area for a colour region to be kept
    min_region_area: int = 500

    # Morphological operations to clean up small holes / noise
    morph_kernel_size: int = 5

    # Canny edge thresholds (used for reference only; not in stencil path)
    canny_low: int = 50
    canny_high: int = 150

    # Output image size (resize before processing)
    output_size: Tuple[int, int] = (512, 512)


@dataclass
class VectorConfig:
    """Parameters for bitmap-to-vector tracing via potrace."""

    # potrace turdsize – ignore speckles smaller than this many pixels
    turdsize: int = 10

    # potrace alphamax – corner threshold (0 = sharp, 1.333 = round)
    alphamax: float = 0.8

    # potrace opttolerance – curve optimisation tolerance
    opttolerance: float = 0.2

    # Minimum path length to keep (in SVG user units)
    min_path_length: float = 20.0

    # SVG canvas size in mm (Cricut standard: 12 × 12 inch = 304.8 mm)
    canvas_width_mm: float = 304.8
    canvas_height_mm: float = 304.8

    # DPI used when converting pixel coords to mm
    dpi: float = 96.0


@dataclass
class StencilConfig:
    """Parameters for stencil layer construction."""

    # Minimum bridge width in pixels to keep isolated islands connected
    bridge_width: int = 8

    # Stroke width for outline layer (px at output resolution)
    outline_stroke_width: float = 2.0

    # Minimum cut width in mm (Cricut can cut down to ~0.5 mm)
    min_cut_width_mm: float = 0.8

    # Whether to add registration marks on each layer
    add_registration_marks: bool = True

    # Registration mark radius in mm
    reg_mark_radius_mm: float = 3.0

    # Padding around the design in mm
    padding_mm: float = 10.0


@dataclass
class OutputConfig:
    """Output path configuration."""

    base_dir: Path = Path("output")
    images_dir: str = "images"
    simplified_dir: str = "simplified"
    svg_dir: str = "svg"

    def images_path(self, run_name: str) -> Path:
        """Return the path for generated images of a run."""
        return self.base_dir / run_name / self.images_dir

    def simplified_path(self, run_name: str) -> Path:
        """Return the path for simplified images of a run."""
        return self.base_dir / run_name / self.simplified_dir

    def svg_path(self, run_name: str) -> Path:
        """Return the path for SVG outputs of a run."""
        return self.base_dir / run_name / self.svg_dir


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration aggregating all sub-configs."""

    generation: GenerationConfig = field(default_factory=GenerationConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)
    stencil: StencilConfig = field(default_factory=StencilConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    # Number of designs to generate per Pokemon in batch mode
    batch_count: int = 3

    # Whether to skip SD generation and use provided reference images directly
    skip_generation: bool = False
