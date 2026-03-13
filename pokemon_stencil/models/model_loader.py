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

Compatibility notes
-------------------
- Requires ``diffusers>=0.27.0`` for stable ``load_ip_adapter()`` and
  ``StableDiffusionXLControlNetPipeline``.
- Requires ``torch>=2.1.0`` for reliable SDXL float32/float16 support.
- The ``ip-adapter`` PyPI package must NOT be installed; it imports the
  removed ``huggingface_hub.cached_download`` and will break the import
  chain.  IP-Adapter support is built into diffusers via ``load_ip_adapter()``.
- ``fuse_lora(lora_scale=...)`` was removed in diffusers 0.27; this module
  handles both the old and new API transparently.

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
from pokemon_stencil.models.lora_validator import validate_lora_sdxl_compatible

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
    if key in _SDXL_CACHE:
        return True
    # Also check legacy SD pipeline cache.
    legacy_source = str(_resolve_model_source(
        config.legacy_model_local_path, config.legacy_model_hub_id
    ))
    return legacy_source in _PIPELINE_CACHE


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


def _source_str(source: Union[Path, str]) -> str:
    """Return a string suitable for passing to a diffusers ``from_pretrained`` call."""
    if isinstance(source, Path):
        # Always use absolute path so diffusers can locate sub-directories.
        return str(source.resolve())
    return source


def _apply_memory_optimizations(pipe, device: str) -> None:
    """
    Apply memory and throughput optimisations to *pipe*.

    Priority order:
    1. xformers memory-efficient attention (GPU only, fastest).
    2. Attention slicing (CPU + GPU, always safe).

    Errors are caught and logged as debug so neither optimisation is a
    hard requirement.
    """
    if device != "cpu":
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.debug("xformers memory efficient attention enabled.")
            return
        except Exception as exc:
            logger.debug("xformers unavailable (%s).", exc)

    try:
        pipe.enable_attention_slicing()
        logger.debug("Attention slicing enabled.")
    except Exception as exc:
        logger.debug("enable_attention_slicing not supported (%s).", exc)


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

    fp16 variant:
      Used only when ``torch_dtype="float16"`` and device is not CPU.
      Hub repos expose fp16 safetensors under the ``variant="fp16"`` key;
      using it on CPU causes a download of redundant weights.
    """
    try:
        import torch
        from diffusers import ControlNetModel, StableDiffusionXLControlNetPipeline
    except ImportError as exc:
        raise ImportError(
            "torch>=2.1.0 and diffusers>=0.27.0 are required for SDXL generation.\n"
            "Install with:\n"
            "  pip install 'torch>=2.1.0,<2.3.0' 'diffusers>=0.27.0,<0.29.0' "
            "transformers accelerate safetensors"
        ) from exc

    torch_dtype = _make_torch_dtype(config)
    use_fp16_variant = (config.torch_dtype == "float16" and config.device != "cpu")

    logger.info("Loading ControlNet OpenPose (SDXL) from '%s' …", openpose_source)
    controlnet_openpose = ControlNetModel.from_pretrained(
        _source_str(openpose_source),
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )

    logger.info("Loading ControlNet Canny (SDXL) from '%s' …", canny_source)
    controlnet_canny = ControlNetModel.from_pretrained(
        _source_str(canny_source),
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )

    logger.info(
        "Loading StableDiffusionXLControlNetPipeline from '%s' "
        "on %s (dtype=%s, fp16_variant=%s) …",
        base_source,
        config.device,
        config.torch_dtype,
        use_fp16_variant,
    )

    load_kwargs: dict = dict(
        controlnet=[controlnet_openpose, controlnet_canny],
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )
    if use_fp16_variant:
        load_kwargs["variant"] = "fp16"

    pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
        _source_str(base_source),
        **load_kwargs,
    )
    pipe = pipe.to(config.device)
    _apply_memory_optimizations(pipe, config.device)

    # ── IP-Adapter (character reference conditioning) ─────────────────────────
    if config.use_ip_adapter:
        _load_ip_adapter(pipe, config)

    # ── Optional LoRA ──────────────────────────────────────────────────────────
    if config.lora_path:
        _apply_lora(pipe, config)

    logger.info("StableDiffusionXLControlNetPipeline loaded and optimised.")
    return pipe


def _load_ip_adapter(pipe, config: GenerationConfig) -> None:
    """
    Load IP-Adapter SDXL weights onto *pipe* for reference image conditioning.

    Uses diffusers' built-in ``load_ip_adapter()`` (available since 0.24).
    The ``ip-adapter`` PyPI package must NOT be installed as it imports the
    removed ``huggingface_hub.cached_download``.

    Resolution strategy:
    - Local directory populated → resolved to absolute path on disk.
    - Otherwise → Hub repo ID string (diffusers auto-downloads on first use).

    The SDXL IP-Adapter weights live in the ``sdxl_models/`` subfolder of the
    ``h94/IP-Adapter`` repository.
    """
    local_path = config.ip_adapter_local_path
    hub_id = config.ip_adapter_hub_id

    if local_path.is_dir() and any(local_path.iterdir()):
        pretrained = str(local_path.resolve())
        logger.info("Loading IP-Adapter from local path '%s' …", pretrained)
    else:
        pretrained = hub_id
        logger.info("Loading IP-Adapter from Hub '%s' …", hub_id)

    try:
        pipe.load_ip_adapter(
            pretrained,
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl.bin",
        )
        pipe.set_ip_adapter_scale(config.ip_adapter_scale)
        logger.info("IP-Adapter loaded (scale=%.2f).", config.ip_adapter_scale)
    except Exception as exc:
        logger.warning(
            "IP-Adapter loading failed (%s); continuing without reference conditioning.",
            exc,
        )


def _apply_lora(pipe, config: GenerationConfig) -> None:
    """
    Validate and apply LoRA weights onto *pipe*, then fuse them.

    Validation
    ----------
    Before loading, the LoRA file is inspected via :mod:`lora_validator` to
    confirm it was trained for SDXL architecture.  Loading an SD1.5 LoRA onto
    an SDXL pipeline causes cross-attention dimension mismatches (768 vs 2048)
    that manifest as ``RuntimeError`` during generation.  This function raises
    a clear ``ValueError`` instead, preventing silent or cryptic failures.

    diffusers API compatibility
    ---------------------------
    * diffusers >=0.27: ``pipe.set_adapters([name], [scale])`` then
      ``pipe.fuse_lora()`` (``lora_scale`` kwarg was removed from fuse_lora).
    * diffusers <=0.26: ``pipe.fuse_lora(lora_scale=scale)``.
    * Last resort: ``pipe.fuse_lora()`` without scale.

    Raises:
        ValueError: LoRA fails SDXL compatibility validation.
        RuntimeError: Unexpected loading or fusion failure.
    """
    lora_path = Path(config.lora_path)
    scale = config.lora_scale

    # ── SDXL compatibility check — must pass before any load attempt ──────────
    ok, validation_msg = validate_lora_sdxl_compatible(lora_path)
    if not ok:
        raise ValueError(validation_msg)

    logger.info(
        "LoRA validated: %s — applying (scale=%.2f).", validation_msg, scale
    )

    # ── Load and fuse ─────────────────────────────────────────────────────────
    try:
        pipe.load_lora_weights(str(lora_path))

        # diffusers >= 0.27: set_adapters + fuse_lora (no lora_scale kwarg)
        try:
            pipe.set_adapters(["default_0"], adapter_weights=[scale])
            pipe.fuse_lora()
            logger.info("LoRA fused via set_adapters + fuse_lora (diffusers>=0.27).")
            return
        except (TypeError, AttributeError):
            pass

        # diffusers <= 0.26: fuse_lora(lora_scale=scale)
        try:
            pipe.fuse_lora(lora_scale=scale)
            logger.info("LoRA fused via fuse_lora(lora_scale=…) (diffusers<=0.26).")
            return
        except TypeError:
            pass

        # Last resort: fuse without scale.
        pipe.fuse_lora()
        logger.warning(
            "LoRA fused without scale adjustment (API incompatible with scale=%.2f).",
            scale,
        )

    except ValueError:
        raise  # Re-raise validation errors unchanged.
    except Exception as exc:
        raise RuntimeError(
            f"LoRA loading failed for '{lora_path.name}'. "
            "Ensure the LoRA is a valid SDXL-compatible .safetensors file. "
            f"Error: {exc}"
        ) from exc


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
        _source_str(source),
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
        use_safetensors=True,
    )
    pipe = pipe.to(config.device)
    _apply_memory_optimizations(pipe, config.device)
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
        _source_str(openpose_source),
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )

    logger.info("Loading SD15 ControlNet Canny from '%s' …", canny_source)
    controlnet_canny = ControlNetModel.from_pretrained(
        _source_str(canny_source),
        torch_dtype=torch_dtype,
        use_safetensors=True,
    )

    logger.info(
        "Loading SD15 pipeline from '%s' on %s (dtype=%s) …",
        base_source, config.device, config.torch_dtype,
    )

    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        _source_str(base_source),
        controlnet=[controlnet_openpose, controlnet_canny],
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False,
        use_safetensors=True,
    )
    pipe = pipe.to(config.device)
    _apply_memory_optimizations(pipe, config.device)

    if config.lora_path:
        _apply_lora(pipe, config)

    logger.info("StableDiffusionControlNetPipeline (SD15) loaded.")
    return pipe
