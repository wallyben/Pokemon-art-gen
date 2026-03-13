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

LoRA safety contract
---------------------
- ``config.lora_path = None``   →  NO LoRA path is ever executed
- Invalid file on disk          →  validation fails → lora_path treated as None
- Failed SDXL validation        →  LoRA blocked, generation continues in pure SDXL
- Failed load_lora_weights()    →  LoRA unloaded, generation continues in pure SDXL
- Failed fuse_lora()            →  LoRA unfused + unloaded, pure SDXL mode
- Any LoRA error                →  pipeline cached WITHOUT LoRA (clean state)

Cache segregation
-----------------
Pipeline cache keys include: model, controlnet, ip-adapter, AND lora_path.
A pipeline cached WITH a LoRA is never returned when lora_path=None (and vice
versa).  If a LoRA fails to load, the pipeline is cached under the NO-LORA key
to prevent re-attempting a broken load without poisoning the LoRA-keyed slot.

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
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

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
    LoRA weights are applied if ``config.lora_path`` is set AND the file
    passes SDXL compatibility validation.  On any LoRA failure the pipeline
    continues in pure SDXL mode (no LoRA) rather than crashing.

    IMPORTANT: The effective LoRA state of the returned pipeline is logged.
    Callers should check the log to confirm LoRA was applied when expected.

    Args:
        config: :class:`~pokemon_stencil.config.GenerationConfig`.

    Returns:
        A ready-to-call ``StableDiffusionXLControlNetPipeline``.

    Raises:
        ImportError: If ``torch`` or ``diffusers`` are not installed.
    """
    # ── Guard: normalise lora_path ────────────────────────────────────────────
    # Ensure lora_path is truly None when not set or when the file is missing.
    effective_lora_path = _resolve_lora_path(config.lora_path)

    base_source = _resolve_model_source(config.model_local_path, config.model_hub_id)
    openpose_source = _resolve_model_source(
        config.controlnet_openpose_local_path,
        config.controlnet_openpose_hub_id,
    )
    canny_source = _resolve_model_source(
        config.controlnet_canny_local_path,
        config.controlnet_canny_hub_id,
    )

    # ── Build cache key (segregated by lora_path) ─────────────────────────────
    base_cache_key = f"sdxl|{base_source}|{openpose_source}|{canny_source}"
    if config.use_ip_adapter:
        base_cache_key += f"|ipa:{config.ip_adapter_hub_id}"

    # Two possible keys: one with LoRA, one without.
    # If LoRA is requested we try the lora key first; on failure we fall back
    # to the no-lora key.  We NEVER mix a lora-keyed entry with a no-lora run.
    no_lora_key = base_cache_key
    lora_cache_key = (
        f"{base_cache_key}|lora:{effective_lora_path}"
        if effective_lora_path
        else None
    )

    # ── Cache lookup ──────────────────────────────────────────────────────────
    # If a LoRA is requested, try the lora-specific key first.
    if lora_cache_key and lora_cache_key in _SDXL_CACHE:
        logger.debug("Returning cached SDXL+LoRA pipeline (%s).", str(lora_cache_key)[:80])
        return _SDXL_CACHE[lora_cache_key]

    # If no LoRA requested (or LoRA key missed), try the no-lora key.
    if effective_lora_path is None and no_lora_key in _SDXL_CACHE:
        logger.debug("Returning cached SDXL pipeline (no LoRA) (%s).", str(no_lora_key)[:80])
        return _SDXL_CACHE[no_lora_key]

    # ── Load base pipeline ────────────────────────────────────────────────────
    pipe = _load_sdxl_from_sources(
        base_source, openpose_source, canny_source, config
    )

    # ── Apply LoRA (safe, non-crashing) ───────────────────────────────────────
    lora_applied = False
    if effective_lora_path:
        lora_applied = _apply_lora_safe(pipe, effective_lora_path, config.lora_scale)
        if lora_applied:
            logger.info(
                "LoRA applied successfully: %s (scale=%.2f). "
                "Pipeline cached under lora-specific key.",
                effective_lora_path.name,
                config.lora_scale,
            )
            _SDXL_CACHE[lora_cache_key] = pipe  # type: ignore[index]
        else:
            logger.warning(
                "LoRA '%s' could not be applied — pipeline continues in pure SDXL mode. "
                "Pipeline cached under no-lora key to prevent re-attempting broken load.",
                effective_lora_path.name,
            )
            # Cache under the NO-LORA key so subsequent no-lora requests get this clean pipe.
            # Do NOT cache under lora_cache_key (would be misleading).
            _SDXL_CACHE[no_lora_key] = pipe
    else:
        logger.info(
            "No LoRA requested — pure SDXL mode. "
            "Pipeline cached under no-lora key."
        )
        _SDXL_CACHE[no_lora_key] = pipe

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
    logger.info(
        "Pipeline cache cleared (%d SD + %d ControlNet + %d SDXL entries removed).",
        n_sd, n_cn, n_sdxl,
    )


def clear_lora_pipeline_cache() -> None:
    """
    Evict only SDXL pipeline cache entries that have a LoRA baked in.

    Use this when the user changes LoRA settings without restarting the app,
    to force a fresh pipeline load with the new LoRA.
    """
    lora_keys = [k for k in _SDXL_CACHE if "|lora:" in k]
    for k in lora_keys:
        del _SDXL_CACHE[k]
    if lora_keys:
        logger.info(
            "Evicted %d LoRA-keyed SDXL pipeline cache entries.", len(lora_keys)
        )
    else:
        logger.debug("No LoRA-keyed cache entries to evict.")


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
    base_key = f"sdxl|{base}|{op}|{ca}"
    if config.use_ip_adapter:
        base_key += f"|ipa:{config.ip_adapter_hub_id}"

    effective_lora = _resolve_lora_path(config.lora_path)
    if effective_lora:
        lora_key = f"{base_key}|lora:{effective_lora}"
        if lora_key in _SDXL_CACHE:
            return True
    if base_key in _SDXL_CACHE:
        return True

    # Also check legacy SD pipeline cache.
    legacy_source = str(_resolve_model_source(
        config.legacy_model_local_path, config.legacy_model_hub_id
    ))
    return legacy_source in _PIPELINE_CACHE


# ── Internal helpers ───────────────────────────────────────────────────────────

def _resolve_lora_path(lora_path: Optional[Path]) -> Optional[Path]:
    """
    Normalise a LoRA path.

    Returns ``None`` in all cases where LoRA should not be applied:
    - ``lora_path`` is ``None``
    - The file does not exist on disk
    - The file is not a ``.safetensors`` file

    This is the single source-of-truth gate that ensures
    ``lora_path = None`` is always honoured and stale/missing paths
    never trigger a load attempt.
    """
    if lora_path is None:
        return None
    p = Path(lora_path)
    if not p.exists():
        logger.debug(
            "LoRA path '%s' does not exist — treating as no LoRA.", p
        )
        return None
    if p.suffix.lower() != ".safetensors":
        logger.debug(
            "LoRA path '%s' is not a .safetensors file — treating as no LoRA.", p
        )
        return None
    return p


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
    Load the SDXL ControlNet pipeline with dual ControlNets and optional
    IP-Adapter conditioning.

    LoRA is NOT applied here — it is handled by the caller
    (``load_sdxl_controlnet_pipeline``) after caching decisions.
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

    logger.info("StableDiffusionXLControlNetPipeline base loaded and optimised.")
    return pipe


def _load_ip_adapter(pipe, config: GenerationConfig) -> None:
    """
    Load IP-Adapter SDXL weights onto *pipe* for reference image conditioning.

    Uses diffusers' built-in ``load_ip_adapter()`` (available since 0.24).
    The ``ip-adapter`` PyPI package must NOT be installed as it imports the
    removed ``huggingface_hub.cached_download``.
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


def _apply_lora_safe(pipe, lora_path: Path, scale: float) -> bool:
    """
    Validate, load, and fuse a LoRA onto *pipe* — in a FULLY SAFE manner.

    This function NEVER raises.  On any failure it:
    1. Attempts to unload any partially-applied LoRA weights from the pipe.
    2. Logs a detailed warning.
    3. Returns ``False`` so the caller can fall back to pure SDXL mode.

    Returns:
        ``True``  — LoRA was validated, loaded, and fused successfully.
        ``False`` — LoRA was blocked or failed; pipe is in clean SDXL state.

    Safety guarantees
    -----------------
    * ``lora_path`` is always a real, existing ``.safetensors`` file (enforced
      by ``_resolve_lora_path`` before this function is called).
    * Validation is the first step; ``load_lora_weights`` is only called when
      validation passes.
    * If ``load_lora_weights`` or ``fuse_lora`` fail after loading, we attempt
      ``unload_lora_weights`` to restore the pipe to a clean state.
    * The pipe object is NEVER partially-fused and returned; it is either fully
      fused (return True) or fully unloaded (return False).
    """
    # ── Step 1: SDXL compatibility validation ─────────────────────────────────
    ok, validation_msg = validate_lora_sdxl_compatible(lora_path)
    if not ok:
        logger.warning(
            "LoRA '%s' failed SDXL validation — blocked. %s",
            lora_path.name,
            validation_msg,
        )
        return False

    logger.info(
        "LoRA validated: %s — loading (scale=%.2f).", validation_msg, scale
    )

    # ── Step 2: Load weights ───────────────────────────────────────────────────
    try:
        pipe.load_lora_weights(str(lora_path))
        logger.debug("load_lora_weights succeeded for '%s'.", lora_path.name)
    except Exception as exc:
        logger.warning(
            "load_lora_weights failed for '%s': %s — falling back to pure SDXL.",
            lora_path.name,
            exc,
        )
        # Try to ensure clean state
        _try_unload_lora(pipe, lora_path.name)
        return False

    # ── Step 3: Set adapter scale + fuse ──────────────────────────────────────
    # diffusers >= 0.27: set_adapters + fuse_lora (no lora_scale kwarg)
    # diffusers <= 0.26: fuse_lora(lora_scale=scale)
    fused = False
    try:
        try:
            pipe.set_adapters(["default_0"], adapter_weights=[scale])
            pipe.fuse_lora()
            fused = True
            logger.info("LoRA fused via set_adapters + fuse_lora (diffusers>=0.27).")
        except (TypeError, AttributeError):
            pass

        if not fused:
            try:
                pipe.fuse_lora(lora_scale=scale)
                fused = True
                logger.info("LoRA fused via fuse_lora(lora_scale=…) (diffusers<=0.26).")
            except TypeError:
                pass

        if not fused:
            pipe.fuse_lora()
            fused = True
            logger.warning(
                "LoRA fused without scale adjustment (API incompatible with scale=%.2f).",
                scale,
            )

    except Exception as exc:
        logger.warning(
            "fuse_lora failed for '%s': %s — unloading LoRA, falling back to pure SDXL.",
            lora_path.name,
            exc,
        )
        _try_unload_lora(pipe, lora_path.name)
        return False

    return fused


def _try_unload_lora(pipe, lora_name: str) -> None:
    """
    Attempt to unload LoRA weights from *pipe*, restoring it to clean state.

    Errors are silently swallowed — this is a best-effort cleanup.
    """
    try:
        pipe.unload_lora_weights()
        logger.debug("unload_lora_weights() succeeded for '%s'.", lora_name)
    except Exception as exc:
        logger.debug(
            "unload_lora_weights() not available or failed for '%s': %s "
            "(pipeline may still be in unfused state).",
            lora_name,
            exc,
        )


def _apply_lora(pipe, config: GenerationConfig) -> None:
    """
    LEGACY wrapper kept for backward compatibility.

    Prefer ``_apply_lora_safe`` for new code — it never raises.
    This function raises ``ValueError`` on validation failure and
    ``RuntimeError`` on loading/fusing failure.

    Raises:
        ValueError: LoRA fails SDXL compatibility validation.
        RuntimeError: Unexpected loading or fusion failure.
    """
    lora_path = Path(config.lora_path)
    scale = config.lora_scale

    ok, validation_msg = validate_lora_sdxl_compatible(lora_path)
    if not ok:
        raise ValueError(validation_msg)

    logger.info(
        "LoRA validated: %s — applying (scale=%.2f).", validation_msg, scale
    )

    try:
        pipe.load_lora_weights(str(lora_path))

        try:
            pipe.set_adapters(["default_0"], adapter_weights=[scale])
            pipe.fuse_lora()
            logger.info("LoRA fused via set_adapters + fuse_lora (diffusers>=0.27).")
            return
        except (TypeError, AttributeError):
            pass

        try:
            pipe.fuse_lora(lora_scale=scale)
            logger.info("LoRA fused via fuse_lora(lora_scale=…) (diffusers<=0.26).")
            return
        except TypeError:
            pass

        pipe.fuse_lora()
        logger.warning(
            "LoRA fused without scale adjustment (API incompatible with scale=%.2f).",
            scale,
        )

    except ValueError:
        raise
    except Exception as exc:
        _try_unload_lora(pipe, Path(config.lora_path).name)
        raise RuntimeError(
            f"LoRA loading failed for '{Path(config.lora_path).name}'. "
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
