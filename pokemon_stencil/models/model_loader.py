"""
Centralised Stable Diffusion model resolution and loading.

All pipeline stages that need a StableDiffusionPipeline object should call
:func:`load_sd_pipeline` rather than instantiating one inline.  This gives
us a single place to control:

- **Model resolution** – local weights first, Hub ID as fallback.
- **Device / dtype** – always CPU + float32 for maximum compatibility.
- **Optimisations** – attention slicing enabled by default.
- **Caching** – a module-level registry keeps one loaded pipeline per
  model identifier so the 4+ GB weights are not loaded twice.

Resolution order
----------------
1. ``config.model_local_path`` (e.g. ``models/sd15/``) – used if the
   directory exists on disk.  Enables fully-offline operation once weights
   have been downloaded.
2. ``config.model_hub_id`` (e.g. ``"runwayml/stable-diffusion-v1-5"``) –
   downloaded from HuggingFace Hub on first run and cached by diffusers.

Usage::

    from pokemon_stencil.models.model_loader import load_sd_pipeline
    pipe = load_sd_pipeline(config)
    images = pipe(prompt="Pikachu stencil art").images
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Union

from pokemon_stencil.config import GenerationConfig

if TYPE_CHECKING:
    # diffusers is an optional heavy dependency; only import for type hints.
    from diffusers import StableDiffusionPipeline

logger = logging.getLogger(__name__)

# ── Module-level pipeline cache ───────────────────────────────────────────────
# Key: resolved model identifier (str or Path normalised to str).
# Value: loaded StableDiffusionPipeline instance.
_PIPELINE_CACHE: Dict[str, "StableDiffusionPipeline"] = {}


# ── Public API ────────────────────────────────────────────────────────────────

def load_sd_pipeline(config: GenerationConfig) -> "StableDiffusionPipeline":
    """
    Resolve, load, and cache a Stable Diffusion 1.5 pipeline.

    The pipeline is always loaded on CPU with float32 precision and has
    attention slicing enabled to minimise peak VRAM / RAM usage.

    Args:
        config: :class:`~pokemon_stencil.config.GenerationConfig` that
                specifies the model source, dtype, and device.

    Returns:
        A ready-to-call ``diffusers.StableDiffusionPipeline`` instance.

    Raises:
        ImportError: If ``torch`` or ``diffusers`` are not installed.
        OSError: If a local model path is explicitly set but cannot be read.
    """
    model_source = _resolve_model_source(config)
    cache_key = str(model_source)

    if cache_key in _PIPELINE_CACHE:
        logger.debug("Returning cached SD pipeline for '%s'.", cache_key)
        return _PIPELINE_CACHE[cache_key]

    pipe = _load_from_source(model_source, config)
    _PIPELINE_CACHE[cache_key] = pipe
    return pipe


def clear_pipeline_cache() -> None:
    """
    Evict all cached pipeline instances.

    Useful in tests or when you need to force a reload after updating
    model weights on disk.
    """
    count = len(_PIPELINE_CACHE)
    _PIPELINE_CACHE.clear()
    logger.debug("Pipeline cache cleared (%d entry/entries removed).", count)


def is_cached(config: GenerationConfig) -> bool:
    """Return ``True`` if a pipeline for *config* is already in the cache."""
    return str(_resolve_model_source(config)) in _PIPELINE_CACHE


# ── Internal helpers ──────────────────────────────────────────────────────────

def _resolve_model_source(config: GenerationConfig) -> Union[Path, str]:
    """
    Determine where to load the model from.

    Resolution order:

    1. ``config.model_local_path`` if the directory exists on disk.
    2. ``config.model_hub_id`` (HuggingFace Hub model ID).

    Args:
        config: Generation configuration containing model path/ID fields.

    Returns:
        A :class:`~pathlib.Path` for a local directory, or a ``str``
        Hub model ID.
    """
    local = config.model_local_path
    if local.is_dir() and any(local.iterdir()):
        # Directory exists and is non-empty – treat as local weights.
        logger.info("Model resolved to local path: %s", local)
        return local

    hub_id = config.model_hub_id
    logger.info(
        "Local model path '%s' not found or empty; resolving to Hub ID '%s'.",
        local,
        hub_id,
    )
    return hub_id


def _load_from_source(
    source: Union[Path, str],
    config: GenerationConfig,
) -> "StableDiffusionPipeline":
    """
    Load a ``StableDiffusionPipeline`` from *source* and apply CPU
    optimisations.

    Args:
        source: Local :class:`~pathlib.Path` or HuggingFace Hub model ID.
        config: Configuration supplying dtype and device strings.

    Returns:
        Configured and optimised ``StableDiffusionPipeline``.

    Raises:
        ImportError: If ``torch`` or ``diffusers`` are unavailable.
    """
    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError as exc:
        raise ImportError(
            "torch and diffusers are required for image generation. "
            "Install them with:\n"
            "  pip install torch diffusers transformers accelerate safetensors"
        ) from exc

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(config.torch_dtype, torch.float32)

    logger.info(
        "Loading StableDiffusionPipeline from '%s' on %s (dtype=%s) …",
        source,
        config.device,
        config.torch_dtype,
    )

    pipe: StableDiffusionPipeline = StableDiffusionPipeline.from_pretrained(
        source,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(config.device)

    # ── CPU optimisations ─────────────────────────────────────────────────────
    # Attention slicing reduces peak RAM by recomputing attention in slices.
    pipe.enable_attention_slicing()

    logger.info("StableDiffusionPipeline loaded and optimised.")
    return pipe
