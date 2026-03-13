"""
Centralised model resolution, loading, and caching.

All pipeline stages that need a diffusers pipeline should call the
appropriate loader function here rather than instantiating one inline.

This module provides two pipeline types:

1. **Standard SD pipeline** (``load_sd_pipeline``) – legacy SD 1.5 pipeline
   used when ControlNet is disabled.

2. **ControlNet pipeline** (``load_controlnet_pipeline``) – the upgraded
   ``StableDiffusionControlNetPipeline`` using the DreamShaper base model
   with dual ControlNet (OpenPose + Canny) conditioning.

Resolution order (applies to all loaders)
------------------------------------------
1. ``config.model_local_path`` – used if the directory exists and is non-empty.
2. ``config.model_hub_id`` – downloaded from HuggingFace Hub on first run.

Usage::

    from pokemon_stencil.models.model_loader import load_controlnet_pipeline
    pipe = load_controlnet_pipeline(config)
    images = pipe(prompt="Pikachu", image=[ctrl_img], ...).images
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Union

from pokemon_stencil.config import GenerationConfig

if TYPE_CHECKING:
    from diffusers import StableDiffusionControlNetPipeline, StableDiffusionPipeline

logger = logging.getLogger(__name__)

# ── Module-level pipeline caches ──────────────────────────────────────────────
# Standard SD pipeline cache (legacy / ControlNet-disabled path).
_PIPELINE_CACHE: Dict[str, "StableDiffusionPipeline"] = {}

# ControlNet pipeline cache – key encodes base + both ControlNet IDs.
_CONTROLNET_CACHE: Dict[str, "StableDiffusionControlNetPipeline"] = {}


# ── Public API ─────────────────────────────────────────────────────────────────

def load_sd_pipeline(config: GenerationConfig) -> "StableDiffusionPipeline":
    """
    Resolve, load, and cache a standard StableDiffusionPipeline.

    Used when ``config.use_controlnet`` is ``False`` or as a fallback.
    Always runs on CPU with float32 and has attention slicing enabled.

    Args:
        config: :class:`~pokemon_stencil.config.GenerationConfig`.

    Returns:
        A ready-to-call ``StableDiffusionPipeline`` instance.

    Raises:
        ImportError: If ``torch`` or ``diffusers`` are not installed.
        OSError: If a local model path is set but cannot be read.
    """
    model_source = _resolve_model_source(
        config.model_local_path, config.model_hub_id
    )
    cache_key = str(model_source)

    if cache_key in _PIPELINE_CACHE:
        logger.debug("Returning cached SD pipeline for '%s'.", cache_key)
        return _PIPELINE_CACHE[cache_key]

    pipe = _load_sd_from_source(model_source, config)
    _PIPELINE_CACHE[cache_key] = pipe
    return pipe


def load_controlnet_pipeline(
    config: GenerationConfig,
) -> "StableDiffusionControlNetPipeline":
    """
    Load and cache a ``StableDiffusionControlNetPipeline`` using the
    DreamShaper base model with dual ControlNet (OpenPose + Canny).

    LoRA weights are applied after loading if ``config.lora_path`` is set.

    Args:
        config: :class:`~pokemon_stencil.config.GenerationConfig`.

    Returns:
        A ready-to-call ``StableDiffusionControlNetPipeline``.

    Raises:
        ImportError: If ``torch`` or ``diffusers`` are not installed.
    """
    base_source = _resolve_model_source(config.model_local_path, config.model_hub_id)
    openpose_source = _resolve_model_source(
        config.controlnet_openpose_local_path,
        config.controlnet_openpose_hub_id,
    )
    canny_source = _resolve_model_source(
        config.controlnet_canny_local_path,
        config.controlnet_canny_hub_id,
    )

    cache_key = f"{base_source}|{openpose_source}|{canny_source}"
    if config.lora_path:
        cache_key += f"|lora:{config.lora_path}"

    if cache_key in _CONTROLNET_CACHE:
        logger.debug("Returning cached ControlNet pipeline (%s).", cache_key[:80])
        return _CONTROLNET_CACHE[cache_key]

    pipe = _load_controlnet_from_sources(
        base_source, openpose_source, canny_source, config
    )
    _CONTROLNET_CACHE[cache_key] = pipe
    return pipe


def clear_pipeline_cache() -> None:
    """Evict all cached pipeline instances (standard and ControlNet)."""
    n_sd = len(_PIPELINE_CACHE)
    n_cn = len(_CONTROLNET_CACHE)
    _PIPELINE_CACHE.clear()
    _CONTROLNET_CACHE.clear()
    logger.debug(
        "Pipeline cache cleared (%d SD + %d ControlNet entries removed).",
        n_sd,
        n_cn,
    )


def is_cached(config: GenerationConfig) -> bool:
    """Return ``True`` if a pipeline for *config* is already in the cache."""
    if config.use_controlnet:
        base = str(
            _resolve_model_source(config.model_local_path, config.model_hub_id)
        )
        op = str(
            _resolve_model_source(
                config.controlnet_openpose_local_path,
                config.controlnet_openpose_hub_id,
            )
        )
        ca = str(
            _resolve_model_source(
                config.controlnet_canny_local_path,
                config.controlnet_canny_hub_id,
            )
        )
        key = f"{base}|{op}|{ca}"
        if config.lora_path:
            key += f"|lora:{config.lora_path}"
        return key in _CONTROLNET_CACHE
    return (
        str(_resolve_model_source(config.model_local_path, config.model_hub_id))
        in _PIPELINE_CACHE
    )


# ── Internal helpers ───────────────────────────────────────────────────────────

def _resolve_model_source(
    local_path: Path, hub_id: str
) -> Union[Path, str]:
    """
    Determine where to load the model from.

    Prefers *local_path* when the directory exists and is non-empty;
    falls back to *hub_id*.
    """
    if local_path.is_dir() and any(local_path.iterdir()):
        logger.info("Model resolved to local path: %s", local_path)
        return local_path
    logger.info(
        "Local path '%s' not found or empty; resolving to Hub ID '%s'.",
        local_path,
        hub_id,
    )
    return hub_id


def _make_torch_dtype(config: GenerationConfig):
    """Map config dtype string to a ``torch`` dtype object."""
    import torch
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return dtype_map.get(config.torch_dtype, torch.float32)


def _load_sd_from_source(
    source: Union[Path, str],
    config: GenerationConfig,
) -> "StableDiffusionPipeline":
    """Load a standard StableDiffusionPipeline and apply CPU optimisations."""
    try:
        import torch
        from diffusers import StableDiffusionPipeline
    except ImportError as exc:
        raise ImportError(
            "torch and diffusers are required for image generation. "
            "Install with:  pip install torch diffusers transformers accelerate safetensors"
        ) from exc

    torch_dtype = _make_torch_dtype(config)
    logger.info(
        "Loading StableDiffusionPipeline from '%s' on %s (dtype=%s) …",
        source,
        config.device,
        config.torch_dtype,
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        source,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(config.device)
    pipe.enable_attention_slicing()
    logger.info("StableDiffusionPipeline loaded and optimised.")
    return pipe


def _load_controlnet_from_sources(
    base_source: Union[Path, str],
    openpose_source: Union[Path, str],
    canny_source: Union[Path, str],
    config: GenerationConfig,
) -> "StableDiffusionControlNetPipeline":
    """
    Load the ControlNet pipeline with DreamShaper base + dual ControlNets.

    Also applies LoRA weights if ``config.lora_path`` is set.
    """
    try:
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
    except ImportError as exc:
        raise ImportError(
            "torch and diffusers are required. "
            "Install with:  pip install torch diffusers transformers accelerate safetensors"
        ) from exc

    torch_dtype = _make_torch_dtype(config)

    logger.info("Loading ControlNet OpenPose from '%s' …", openpose_source)
    controlnet_openpose = ControlNetModel.from_pretrained(
        openpose_source,
        torch_dtype=torch_dtype,
    )

    logger.info("Loading ControlNet Canny from '%s' …", canny_source)
    controlnet_canny = ControlNetModel.from_pretrained(
        canny_source,
        torch_dtype=torch_dtype,
    )

    logger.info(
        "Loading DreamShaper pipeline from '%s' on %s (dtype=%s) …",
        base_source,
        config.device,
        config.torch_dtype,
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        base_source,
        controlnet=[controlnet_openpose, controlnet_canny],
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(config.device)
    pipe.enable_attention_slicing()

    # ── Optional LoRA ──────────────────────────────────────────────────────────
    if config.lora_path and Path(config.lora_path).is_file():
        logger.info("Applying LoRA weights from '%s' (scale=%.2f).", config.lora_path, config.lora_scale)
        try:
            pipe.load_lora_weights(str(config.lora_path))
            pipe.fuse_lora(lora_scale=config.lora_scale)
            logger.info("LoRA applied successfully.")
        except Exception as exc:
            logger.warning("LoRA loading failed (%s); continuing without LoRA.", exc)

    logger.info("StableDiffusionControlNetPipeline loaded and optimised.")
    return pipe
