"""
Centralised model resolution, loading, and caching.

All pipeline stages that need a diffusers pipeline should call the
appropriate loader function here rather than instantiating one inline.

This module provides three pipeline types:

1. **SDXL ControlNet pipeline** (``load_sdxl_controlnet_pipeline``) – the primary
   upgraded pipeline using ``StableDiffusionXLControlNetPipeline`` with SDXL base,
   dual ControlNet (OpenPose + Canny), and optional IP-Adapter conditioning.

2. **Standard SD pipeline** (``load_sd_pipeline``) – legacy SD 1.5 pipeline used
   when ControlNet is disabled or as a fallback.

3. **ControlNet pipeline** (``load_controlnet_pipeline``) – SD 1.5-based
   ``StableDiffusionControlNetPipeline`` kept for backward compatibility.

Resolution order (applies to all loaders)
------------------------------------------
1. ``config.model_local_path`` – used if the directory exists and is non-empty.
2. ``config.model_hub_id`` – downloaded from HuggingFace Hub on first run.

Usage::

    from pokemon_stencil.models.model_loader import load_sdxl_controlnet_pipeline
    pipe = load_sdxl_controlnet_pipeline(config)
    images = pipe(prompt="Pikachu", image=[ctrl_img], ...).images
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from pokemon_stencil.config import GenerationConfig

if TYPE_CHECKING:
    from diffusers import (
        StableDiffusionControlNetPipeline,
        StableDiffusionPipeline,
        StableDiffusionXLControlNetPipeline,
    )

logger = logging.getLogger(__name__)

# ── Module-level pipeline caches ──────────────────────────────────────────────
_PIPELINE_CACHE: Dict[str, "StableDiffusionPipeline"] = {}
_CONTROLNET_CACHE: Dict[str, "StableDiffusionControlNetPipeline"] = {}
_SDXL_CACHE: Dict[str, "StableDiffusionXLControlNetPipeline"] = {}


# ── Public API ─────────────────────────────────────────────────────────────────

def load_sdxl_controlnet_pipeline(
    config: GenerationConfig,
) -> "StableDiffusionXLControlNetPipeline":
    """
    Load and cache a ``StableDiffusionXLControlNetPipeline`` using the SDXL
    base model with dual ControlNet (OpenPose + Canny).

    IP-Adapter conditioning is loaded after pipeline init if
    ``config.use_ip_adapter`` is ``True``.
    LoRA weights are applied if ``config.lora_path`` is set.

    Args:
        config: :class:`~pokemon_stencil.config.GenerationConfig`.

    Returns:
        A ready-to-call ``StableDiffusionXLControlNetPipeline``.

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

    cache_key = f"sdxl|{base_source}|{openpose_source}|{canny_source}"
    if config.lora_path:
        cache_key += f"|lora:{config.lora_path}"
    if config.use_ip_adapter:
        cache_key += f"|ipa:{config.ip_adapter_hub_id}"

    if cache_key in _SDXL_CACHE:
        logger.debug("Returning cached SDXL pipeline (%s).", cache_key[:80])
        return _SDXL_CACHE[cache_key]

    pipe = _load_sdxl_from_sources(
        base_source, openpose_source, canny_source, config
    )
    _SDXL_CACHE[cache_key] = pipe
    return pipe


def load_controlnet_pipeline(
    config: GenerationConfig,
) -> "StableDiffusionControlNetPipeline":
    """
    Load and cache a ``StableDiffusionControlNetPipeline`` (SD 1.5-based).

    Legacy path kept for backward compatibility with DreamShaper configs.

    Args:
        config: :class:`~pokemon_stencil.config.GenerationConfig`.

    Returns:
        A ready-to-call ``StableDiffusionControlNetPipeline``.
    """
    base_source = _resolve_model_source(
        config.legacy_model_local_path, config.legacy_model_hub_id
    )
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


def load_sd_pipeline(config: GenerationConfig) -> "StableDiffusionPipeline":
    """
    Resolve, load, and cache a standard StableDiffusionPipeline (legacy).

    Args:
        config: :class:`~pokemon_stencil.config.GenerationConfig`.

    Returns:
        A ready-to-call ``StableDiffusionPipeline`` instance.
    """
    model_source = _resolve_model_source(
        config.legacy_model_local_path, config.legacy_model_hub_id
    )
    cache_key = str(model_source)

    if cache_key in _PIPELINE_CACHE:
        logger.debug("Returning cached SD pipeline for '%s'.", cache_key)
        return _PIPELINE_CACHE[cache_key]

    pipe = _load_sd_from_source(model_source, config)
    _PIPELINE_CACHE[cache_key] = pipe
    return pipe


def clear_pipeline_cache() -> None:
    """Evict all cached pipeline instances."""
    n_sd = len(_PIPELINE_CACHE)
    n_cn = len(_CONTROLNET_CACHE)
    n_sdxl = len(_SDXL_CACHE)
    _PIPELINE_CACHE.clear()
    _CONTROLNET_CACHE.clear()
    _SDXL_CACHE.clear()
    logger.debug(
        "Pipeline cache cleared (%d SD + %d ControlNet + %d SDXL entries removed).",
        n_sd, n_cn, n_sdxl,
    )


def is_cached(config: GenerationConfig) -> bool:
    """Return ``True`` if a pipeline for *config* is already in the cache."""
    base = str(_resolve_model_source(config.model_local_path, config.model_hub_id))
    op = str(_resolve_model_source(
        config.controlnet_openpose_local_path,
        config.controlnet_openpose_hub_id,
    ))
    ca = str(_resolve_model_source(
        config.controlnet_canny_local_path,
        config.controlnet_canny_hub_id,
    ))
    key = f"sdxl|{base}|{op}|{ca}"
    if config.lora_path:
        key += f"|lora:{config.lora_path}"
    if config.use_ip_adapter:
        key += f"|ipa:{config.ip_adapter_hub_id}"
    return key in _SDXL_CACHE


# ── Internal helpers ───────────────────────────────────────────────────────────

def _resolve_model_source(
    local_path: Path, hub_id: str
) -> Union[Path, str]:
    """Prefer local path when non-empty; fall back to hub_id."""
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


def _load_sdxl_from_sources(
    base_source: Union[Path, str],
    openpose_source: Union[Path, str],
    canny_source: Union[Path, str],
    config: GenerationConfig,
) -> "StableDiffusionXLControlNetPipeline":
    """
    Load the SDXL ControlNet pipeline with dual ControlNets, optional
    IP-Adapter conditioning, and optional LoRA.

    Architecture per spec:
      Text Prompt → Prompt Optimisation → Reference Encoding (IP-Adapter)
      → ControlNet Pose Conditioning → ControlNet Edge Conditioning
      → SDXL Base Diffusion → Optional Character LoRA → Image Generation
    """
    try:
        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
    except ImportError as exc:
        raise ImportError(
            "torch and diffusers>=0.24.0 are required for SDXL generation. "
            "Install with:  pip install torch 'diffusers>=0.24.0' transformers "
            "accelerate safetensors"
        ) from exc

    torch_dtype = _make_torch_dtype(config)

    logger.info("Loading ControlNet OpenPose (SDXL) from '%s' …", openpose_source)
    controlnet_openpose = ControlNetModel.from_pretrained(
        openpose_source,
        torch_dtype=torch_dtype,
    )

    logger.info("Loading ControlNet Canny (SDXL) from '%s' …", canny_source)
    controlnet_canny = ControlNetModel.from_pretrained(
        canny_source,
        torch_dtype=torch_dtype,
    )

    logger.info(
        "Loading StableDiffusionXLControlNetPipeline from '%s' on %s (dtype=%s) …",
        base_source,
        config.device,
        config.torch_dtype,
    )

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        base_source,
        controlnet=[controlnet_openpose, controlnet_canny],
        torch_dtype=torch_dtype,
    )
    pipe = pipe.to(config.device)
    pipe.enable_attention_slicing()

    # ── IP-Adapter (character reference conditioning) ─────────────────────────
    if config.use_ip_adapter:
        _load_ip_adapter(pipe, config)

    # ── Optional LoRA ──────────────────────────────────────────────────────────
    if config.lora_path and Path(config.lora_path).is_file():
        _apply_lora(pipe, config)

    logger.info("StableDiffusionXLControlNetPipeline loaded and optimised.")
    return pipe


def _load_ip_adapter(pipe, config: GenerationConfig) -> None:
    """
    Load IP-Adapter weights onto *pipe* for reference image conditioning.

    Uses the SDXL-compatible IP-Adapter from ``h94/IP-Adapter``.
    The adapter sub-folder ``sdxl_models`` is used automatically by diffusers
    when the base pipeline is SDXL.

    Args:
        pipe: Loaded ``StableDiffusionXLControlNetPipeline``.
        config: Generation config with IP-Adapter hub/local paths.
    """
    ip_source = _resolve_model_source(
        config.ip_adapter_local_path, config.ip_adapter_hub_id
    )
    try:
        logger.info("Loading IP-Adapter from '%s' …", ip_source)
        # diffusers >= 0.24 supports load_ip_adapter natively on XL pipelines.
        # The SDXL variant lives in the 'sdxl_models' subfolder.
        pipe.load_ip_adapter(
            str(ip_source),
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl.bin",
        )
        pipe.set_ip_adapter_scale(config.ip_adapter_scale)
        logger.info(
            "IP-Adapter loaded (scale=%.2f).", config.ip_adapter_scale
        )
    except Exception as exc:
        logger.warning(
            "IP-Adapter loading failed (%s); "
            "continuing without reference conditioning.",
            exc,
        )


def _apply_lora(pipe, config: GenerationConfig) -> None:
    """Apply and fuse LoRA weights onto *pipe*."""
    lora_path = Path(config.lora_path)
    logger.info(
        "Applying LoRA weights from '%s' (scale=%.2f).",
        lora_path,
        config.lora_scale,
    )
    try:
        pipe.load_lora_weights(str(lora_path))
        pipe.fuse_lora(lora_scale=config.lora_scale)
        logger.info("LoRA applied and fused successfully.")
    except Exception as exc:
        logger.warning("LoRA loading failed (%s); continuing without LoRA.", exc)


def _load_sd_from_source(
    source: Union[Path, str],
    config: GenerationConfig,
) -> "StableDiffusionPipeline":
    """Load a standard StableDiffusionPipeline (legacy path)."""
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
        "Loading StableDiffusionPipeline (legacy) from '%s' on %s (dtype=%s) …",
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
    logger.info("StableDiffusionPipeline loaded.")
    return pipe


def _load_controlnet_from_sources(
    base_source: Union[Path, str],
    openpose_source: Union[Path, str],
    canny_source: Union[Path, str],
    config: GenerationConfig,
) -> "StableDiffusionControlNetPipeline":
    """Load SD 1.5 ControlNet pipeline (legacy path)."""
    try:
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetPipeline
    except ImportError as exc:
        raise ImportError(
            "torch and diffusers are required. "
            "Install with:  pip install torch diffusers transformers accelerate safetensors"
        ) from exc

    torch_dtype = _make_torch_dtype(config)

    logger.info("Loading SD15 ControlNet OpenPose from '%s' …", openpose_source)
    controlnet_openpose = ControlNetModel.from_pretrained(
        openpose_source, torch_dtype=torch_dtype,
    )

    logger.info("Loading SD15 ControlNet Canny from '%s' …", canny_source)
    controlnet_canny = ControlNetModel.from_pretrained(
        canny_source, torch_dtype=torch_dtype,
    )

    logger.info(
        "Loading SD15 pipeline from '%s' on %s (dtype=%s) …",
        base_source, config.device, config.torch_dtype,
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

    if config.lora_path and Path(config.lora_path).is_file():
        _apply_lora(pipe, config)

    logger.info("StableDiffusionControlNetPipeline (SD15) loaded.")
    return pipe
