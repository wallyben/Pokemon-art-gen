"""
Configuration management for the Pokemon Stencil Art Factory.

All tunable parameters across every pipeline stage are defined here as
Python dataclasses.  Modules accept config objects rather than raw
literals, making the system easy to reconfigure from the CLI or tests
without touching internal logic.

Directory conventions
---------------------
- All runtime outputs live under ``outputs/``.
- Pre-downloaded model weights are expected under ``models/`` so the
  pipeline can run fully offline.

Generation backends (ProviderConfig)
--------------------------------------
- "auto"  : fal.ai if FAL_KEY set, else local SDXL  ← recommended
- "fal"   : fal.ai FLUX.1 Dev (cloud, no GPU, set FAL_KEY)
- "local" : local SDXL + ControlNet + IP-Adapter stack

Model stack (local SDXL — fallback)
---------------------------------
- Base model:              stabilityai/stable-diffusion-xl-base-1.0
- Pipeline:                StableDiffusionXLControlNetPipeline
- ControlNet OpenPose:     thibaud/controlnet-openpose-sdxl-1.0
- ControlNet Canny:        diffusers/controlnet-canny-sdxl-1.0
- IP-Adapter:              h94/IP-Adapter (SDXL variant)
- LoRA:                    optional character-enhancement LoRA (models/lora/)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple


# ── SDXL model paths ──────────────────────────────────────────────────────────

#: SDXL base model – primary generation backbone.
DEFAULT_SDXL_LOCAL_PATH: Path = Path("models/sdxl")
DEFAULT_SDXL_HUB_ID: str = "stabilityai/stable-diffusion-xl-base-1.0"

#: ControlNet OpenPose – SDXL-compatible pose guidance.
DEFAULT_CONTROLNET_OPENPOSE_LOCAL_PATH: Path = Path("models/controlnet_pose")
DEFAULT_CONTROLNET_OPENPOSE_HUB_ID: str = "thibaud/controlnet-openpose-sdxl-1.0"

#: ControlNet Canny – SDXL-compatible edge/structure guidance.
DEFAULT_CONTROLNET_CANNY_LOCAL_PATH: Path = Path("models/controlnet_canny")
DEFAULT_CONTROLNET_CANNY_HUB_ID: str = "diffusers/controlnet-canny-sdxl-1.0"

#: IP-Adapter – reference image conditioning for character preservation.
DEFAULT_IP_ADAPTER_LOCAL_PATH: Path = Path("models/ip_adapter")
DEFAULT_IP_ADAPTER_HUB_ID: str = "h94/IP-Adapter"

#: LoRA directory – character-specific fine-tuning weights.
DEFAULT_LORA_DIR: Path = Path("models/lora")

# ── Legacy SD 1.5 constants (kept for backward compatibility) ─────────────────
DEFAULT_MODEL_LOCAL_PATH: Path = Path("models/sd15")
DEFAULT_MODEL_HUB_ID: str = "runwayml/stable-diffusion-v1-5"

#: Legacy DreamShaper model (SD 1.5 fine-tune).
DEFAULT_DREAMSHAPER_LOCAL_PATH: Path = Path("models/dreamshaper")
DEFAULT_DREAMSHAPER_HUB_ID: str = "Lykon/DreamShaper"

#: SD 1.5 ControlNet IDs (kept for reference; use SDXL variants above).
SD15_CONTROLNET_OPENPOSE_HUB_ID: str = "lllyasviel/control_v11p_sd15_openpose"
SD15_CONTROLNET_CANNY_HUB_ID: str = "lllyasviel/control_v11p_sd15_canny"


# ─────────────────────────────────────────────────────────────────────────────
# Sub-configs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class GenerationConfig:
    """Parameters for image generation using the SDXL + ControlNet + IP-Adapter stack.

    Model resolution order:
    1. Local path (``model_local_path``) if directory is non-empty.
    2. HuggingFace Hub ID (``model_hub_id``).

    Pipeline:
    - ``StableDiffusionXLControlNetPipeline`` with OpenPose + Canny ControlNets
    - IP-Adapter conditioning from reference images in ``refs/<pokemon_name>/``
    - Optional LoRA weights merged from ``models/lora/``

    Generation settings per spec:
    - steps = 28
    - guidance_scale = 7.5
    - width = height = 1024
    - reference_strength (IP-Adapter scale) = 0.45
    """

    # ── SDXL base model ───────────────────────────────────────────────────────
    model_local_path: Path = field(
        default_factory=lambda: DEFAULT_SDXL_LOCAL_PATH
    )
    model_hub_id: str = DEFAULT_SDXL_HUB_ID

    # ── Legacy SD 1.5 / DreamShaper fallback ─────────────────────────────────
    legacy_model_hub_id: str = DEFAULT_MODEL_HUB_ID
    legacy_model_local_path: Path = field(
        default_factory=lambda: DEFAULT_DREAMSHAPER_LOCAL_PATH
    )

    # ── ControlNet models (SDXL-compatible) ───────────────────────────────────
    controlnet_openpose_local_path: Path = field(
        default_factory=lambda: DEFAULT_CONTROLNET_OPENPOSE_LOCAL_PATH
    )
    controlnet_openpose_hub_id: str = DEFAULT_CONTROLNET_OPENPOSE_HUB_ID

    controlnet_canny_local_path: Path = field(
        default_factory=lambda: DEFAULT_CONTROLNET_CANNY_LOCAL_PATH
    )
    controlnet_canny_hub_id: str = DEFAULT_CONTROLNET_CANNY_HUB_ID

    #: When True, use ControlNet pipeline (OpenPose + Canny).
    use_controlnet: bool = True

    #: Conditioning scale for the ControlNet OpenPose branch [0..2].
    controlnet_openpose_scale: float = 0.8

    #: Conditioning scale for the ControlNet Canny branch [0..2].
    controlnet_canny_scale: float = 0.6

    # ── IP-Adapter (reference image conditioning) ─────────────────────────────
    ip_adapter_local_path: Path = field(
        default_factory=lambda: DEFAULT_IP_ADAPTER_LOCAL_PATH
    )
    ip_adapter_hub_id: str = DEFAULT_IP_ADAPTER_HUB_ID

    #: When True, encode reference images via IP-Adapter CLIP encoder.
    use_ip_adapter: bool = True

    #: IP-Adapter conditioning strength; 0.45 per spec.
    ip_adapter_scale: float = 0.45

    #: Minimum reference images to load for IP-Adapter conditioning.
    ip_adapter_min_images: int = 10

    #: Maximum reference images to load for IP-Adapter conditioning.
    ip_adapter_max_images: int = 30

    # ── LoRA ──────────────────────────────────────────────────────────────────
    #: Path to an optional LoRA ``.safetensors`` file for character enhancement.
    lora_path: Optional[Path] = None

    #: Directory containing available LoRA weight files.
    lora_dir: Path = field(default_factory=lambda: DEFAULT_LORA_DIR)

    #: LoRA blending scale applied via ``load_lora_weights``.
    lora_scale: float = 0.8

    # ── Reference encoding (legacy img2img path) ───────────────────────────────
    #: When True and IP-Adapter is disabled, use img2img reference encoding.
    use_reference_encoding: bool = True

    #: img2img denoising strength; lower = more faithful to reference.
    reference_strength: float = 0.45

    # ── Generation parameters (per spec) ─────────────────────────────────────
    num_inference_steps: int = 28
    guidance_scale: float = 7.5
    width: int = 1024
    height: int = 1024

    #: Optional fixed seed for reproducible outputs.
    seed: Optional[int] = None

    #: dtype – use float16 for GPU, float32 for CPU.
    torch_dtype: str = "float32"
    device: str = "cpu"

    # ── Prompt ────────────────────────────────────────────────────────────────
    #: Prompt suffix appended to every generation.
    style_suffix: str = (
        "clean cartoon illustration, bold black outlines, vector illustration style, "
        "high contrast lighting, flat colour shapes, poster illustration style, "
        "stencil-friendly composition"
    )

    negative_prompt: str = (
        "photorealistic, gradient, complex texture, noise, "
        "blurry, watermark, signature, multiple characters, "
        "extra limbs, deformed, low quality"
    )

    #: When True, pass the prompt through PromptEngine to enforce 77-token limit.
    optimise_prompt: bool = True

    # ── Composition guidance ──────────────────────────────────────────────────
    use_composition_guidance: bool = True
    composition_strength: float = 0.6

    # ── Pose reference ────────────────────────────────────────────────────────
    #: Optional path to a pose reference image for OpenPose conditioning.
    pose_reference_path: Optional[Path] = None

    # ── Reference fetching ────────────────────────────────────────────────────
    auto_fetch_references: bool = False
    max_reference_images: int = 25

    @property
    def model_id(self) -> str | Path:
        """Return local path if it exists, otherwise the Hub model ID."""
        if self.model_local_path.exists():
            return self.model_local_path
        return self.model_hub_id

    def list_available_loras(self) -> List[Path]:
        """Return all .safetensors files in ``lora_dir``."""
        if not self.lora_dir.is_dir():
            return []
        return sorted(self.lora_dir.glob("*.safetensors"))


@dataclass
class ProcessingConfig:
    """Parameters for image simplification and colour segmentation."""

    # ── Bilateral filter ──────────────────────────────────────────────────────
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    bilateral_passes: int = 3

    # ── Colour quantisation ───────────────────────────────────────────────────
    #: Number of K-Means clusters == number of stencil colour layers.
    #: Range 4–8 per spec.
    n_colors: int = 6

    #: Layers with fewer pixels than this are discarded as noise.
    min_region_area: int = 500

    # ── Morphological cleanup ─────────────────────────────────────────────────
    morph_kernel_size: int = 5

    # ── Edge detection ────────────────────────────────────────────────────────
    canny_low: int = 50
    canny_high: int = 150

    # ── Image dimensions ──────────────────────────────────────────────────────
    #: Processing resolution.  1024 for SDXL generation.
    output_size: Tuple[int, int] = (1024, 1024)

    # ── Stencil safety improvements ───────────────────────────────────────────
    adaptive_edge_thinning: bool = True
    merge_small_components: bool = True
    min_island_area: int = 200

    # ── Region merging (post k-means) ─────────────────────────────────────────
    #: When True, merge adjacent colour regions with low colour distance.
    merge_adjacent_regions: bool = True

    #: Maximum LAB ΔE between regions to trigger merging.
    region_merge_threshold: float = 15.0


@dataclass
class VectorConfig:
    """Parameters for bitmap-to-vector tracing via potrace."""

    turdsize: int = 10
    alphamax: float = 0.8
    opttolerance: float = 0.2
    min_path_length: float = 20.0

    # ── SVG canvas ────────────────────────────────────────────────────────────
    #: Cricut standard mat: 12 × 12 inches = 304.8 × 304.8 mm.
    canvas_width_mm: float = 304.8
    canvas_height_mm: float = 304.8
    dpi: float = 96.0


@dataclass
class StencilConfig:
    """Parameters for stencil-safety processing and layer construction."""

    bridge_width: int = 8
    outline_stroke_width: float = 2.0
    min_cut_width_mm: float = 0.8
    add_registration_marks: bool = True
    reg_mark_radius_mm: float = 3.0
    padding_mm: float = 10.0


@dataclass
class OutputConfig:
    """Output path configuration."""

    base_dir: Path = field(default_factory=lambda: Path("outputs"))

    images_dir: str = "images"
    simplified_dir: str = "simplified"
    masks_dir: str = "masks"
    svg_dir: str = "svg"

    def run_root(self, run_name: str) -> Path:
        return self.base_dir / run_name

    def images_path(self, run_name: str) -> Path:
        return self.base_dir / run_name / self.images_dir

    def simplified_path(self, run_name: str) -> Path:
        return self.base_dir / run_name / self.simplified_dir

    def masks_path(self, run_name: str) -> Path:
        return self.base_dir / run_name / self.masks_dir

    def svg_path(self, run_name: str) -> Path:
        return self.base_dir / run_name / self.svg_dir

    def ensure_run_dirs(self, run_name: str) -> None:
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

    count: int = 3
    workers: int = 1


@dataclass
class FactoryConfig:
    """Parameters for art factory mode.

    Per spec: generate 20 candidate images using seed/lighting/angle variation,
    score them, and return the best candidates.
    """

    #: Total candidates to generate per prompt (20 per spec).
    count: int = 20

    #: Number of top-scoring candidates to convert to stencil SVGs.
    top_k: int = 3

    #: Parallel worker processes.
    workers: int = 1

    #: Cosine similarity threshold for diversity filtering.
    diversity_threshold: float = 0.85


# ─────────────────────────────────────────────────────────────────────────────
# Top-level aggregate
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ProviderConfig:
    """Configuration for the generation backend (provider abstraction layer).

    provider choices:
        "auto"  — try fal.ai first (if FAL_KEY set), fall back to local SDXL.
        "fal"   — fal.ai FLUX.1 Dev.  Requires FAL_KEY env var + fal-client.
        "local" — local SDXL stack.  Requires torch + diffusers + GPU.

    fal.ai setup (fastest, no GPU needed):
        pip install fal-client
        $env:FAL_KEY = "your-key"  # PowerShell
        Get key: https://fal.ai/dashboard/keys

    n_candidates:
        How many images to generate per run.  Best n_top are kept.
        Lower for speed; higher for quality selection.
    """

    provider: str = "auto"          # "auto" | "fal" | "local"

    # ── Candidate generation ──────────────────────────────────────────────────
    n_candidates: int = 8
    n_top: int = 3

    # ── Generation parameters ─────────────────────────────────────────────────
    num_steps: int = 28
    guidance_scale: float = 3.5
    reference_strength: float = 0.80
    seed: Optional[int] = None

    # ── fal.ai specific ───────────────────────────────────────────────────────
    fal_model: str = "fal-ai/flux/dev"
    fal_img2img_model: str = "fal-ai/flux/dev/image-to-image"

    # ── Local SDXL specific ───────────────────────────────────────────────────
    local_lora_path: Optional[Path] = None
    local_lora_scale: float = 0.8
    local_use_ip_adapter: bool = True
    local_torch_dtype: str = "float32"   # "float32" (CPU) or "float16" (GPU)


@dataclass
class PipelineConfig:
    """Top-level configuration aggregating all pipeline sub-configs."""

    provider: ProviderConfig = field(default_factory=ProviderConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    vector: VectorConfig = field(default_factory=VectorConfig)
    stencil: StencilConfig = field(default_factory=StencilConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    batch: BatchConfig = field(default_factory=BatchConfig)
    factory: FactoryConfig = field(default_factory=FactoryConfig)

    #: When True, skip SD generation and use reference images directly.
    skip_generation: bool = False
