"""
SDXL LoRA compatibility validator.

Validates that a supplied LoRA ``.safetensors`` file is intended for the SDXL
architecture before loading it into the pipeline.

Why this matters
----------------
Loading an SD1.5 LoRA onto an SDXL pipeline causes silent weight mismatches or
``RuntimeError: shape mismatch`` during inference because SDXL's cross-attention
key/value projections expect a 2048-dim text embedding (concatenated CLIP-L 768
+ OpenCLIP-bigG 1280), while SD1.5 LoRAs were trained against the 768-dim
CLIP-L embedding only.

Detection heuristics (applied in order)
-----------------------------------------
1. **Metadata** — ``ss_base_model_version`` key written by popular trainers
   (kohya_ss, OneTrainer, etc.).  Contains strings like ``sd_xl_base_1.0`` for
   SDXL or ``sd-1-5`` for SD 1.5.
2. **Cross-attention projection dimension** — the ``lora_down`` weight for
   ``attn2.to_k`` / ``attn2.to_v`` layers has an input dimension equal to the
   text-conditioning dimension:
   * SDXL → 2048 (CLIP-L 768 + OpenCLIP-bigG 1280)
   * SD1.5 → 768 (CLIP-L only)
3. **Max-dimension heuristic (strict)** — if no cross-attention keys are found,
   the maximum tensor dimension is used as a strict fallback signal:
   * dim >= 2048 → accepted as SDXL-compatible
   * dim <= 1280 → rejected (SD1.5 UNet max channel is 1280)
   * 1280 < dim < 2048 → rejected conservatively (ambiguous range)

Usage::

    from pokemon_stencil.models.lora_validator import validate_lora_sdxl_compatible
    ok, msg = validate_lora_sdxl_compatible(Path("models/lora/pokemon_xl.safetensors"))
    if not ok:
        raise ValueError(msg)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

logger = logging.getLogger(__name__)

# ── Architecture constants ─────────────────────────────────────────────────────

#: Text-conditioning dimension for SDXL (CLIP-L 768 + OpenCLIP-bigG 1280).
_SDXL_TEXT_DIM: int = 2048

#: Text-conditioning dimension for SD 1.5 (CLIP-L only).
_SD15_TEXT_DIM: int = 768

#: Maximum UNet channel dimension for SD 1.5 (mid-block).
#: SD1.5 LoRAs that don't hit the cross-attn path will still have max_dim ≤ 1280.
_SD15_MAX_UNET_DIM: int = 1280

#: Substrings that appear in metadata when the base model is SDXL.
_SDXL_METADATA_KEYWORDS = ("sd_xl", "sdxl", "xl_base", "stable_diffusion_xl", "xl-base")

#: Substrings that appear in metadata when the base model is SD 1.5.
_SD15_METADATA_KEYWORDS = (
    "sd-1", "sd1", "v1-5", "v1.5", "stable_diffusion_v1", "stable-diffusion-v1",
    "sd_1", "sd15",
)

#: Cross-attention key patterns for auto-detection (multiple naming conventions).
#: These cover kohya_ss, diffusers, and CompVis naming formats.
_CROSS_ATTN_PATTERNS = (
    # kohya_ss format
    ("attn2", "to_k", "lora_down"),
    ("attn2", "to_v", "lora_down"),
    # transformer_blocks format
    ("transformer_blocks", "attn2", "to_k"),
    ("transformer_blocks", "attn2", "to_v"),
)


# ── Public API ──────────────────────────────────────────────────────────────────

def validate_lora_sdxl_compatible(lora_path: Path) -> Tuple[bool, str]:
    """
    Validate that *lora_path* points to an SDXL-compatible LoRA file.

    Args:
        lora_path: Path to the ``.safetensors`` LoRA weight file.

    Returns:
        ``(True, success_message)`` if the LoRA is SDXL-compatible.
        ``(False, error_message)`` if the file is absent, has a wrong extension,
        or contains SD1.5 weights.

    This function never raises — all failure paths are returned as
    ``(False, message)`` tuples so callers can decide whether to raise or log.
    """
    # ── 1. Existence & extension checks ──────────────────────────────────────
    if not lora_path.exists():
        return False, f"LoRA file not found: {lora_path}"

    if lora_path.suffix.lower() != ".safetensors":
        return (
            False,
            f"LoRA must be a .safetensors file, got: '{lora_path.suffix}'. "
            "Only safetensors LoRA files are supported.",
        )

    # ── 2. Introspect file via safetensors ───────────────────────────────────
    try:
        from safetensors import safe_open  # type: ignore[import]
    except ImportError:
        return (
            False,
            "safetensors package not installed; cannot validate LoRA architecture. "
            "Install with:  pip install 'safetensors>=0.4.2,<0.5.0'",
        )

    try:
        with safe_open(str(lora_path), framework="pt", device="cpu") as f:
            metadata: dict = f.metadata() or {}
            keys = list(f.keys())

        if not keys:
            return False, f"LoRA file '{lora_path.name}' appears empty or corrupt."

        # ── 2a. Metadata-based detection (most reliable) ─────────────────
        base_ver = metadata.get("ss_base_model_version", "").lower().strip()
        if base_ver:
            logger.debug(
                "LoRA '%s' metadata ss_base_model_version='%s'",
                lora_path.name, base_ver,
            )
            if any(kw in base_ver for kw in _SDXL_METADATA_KEYWORDS):
                return (
                    True,
                    f"LoRA '{lora_path.name}' is SDXL-compatible "
                    f"(metadata base_model='{base_ver}').",
                )
            if any(kw in base_ver for kw in _SD15_METADATA_KEYWORDS):
                return (
                    False,
                    "Loaded LoRA is not SDXL-compatible. Please provide an SDXL LoRA. "
                    f"(Detected base model: '{base_ver}' — this is an SD1.5 LoRA.)",
                )

        # ── 2b. Cross-attention projection dimension (shape-based) ────────
        #
        # In a LoRA, the ``lora_down`` weight for the cross-attention
        # key/value projection reflects the text-conditioning input dimension:
        #   SDXL: attn2.to_k.lora_down  → shape [rank, 2048]
        #   SD15: attn2.to_k.lora_down  → shape [rank, 768]
        cross_attn_dim = _detect_cross_attn_dim(keys, lora_path)
        if cross_attn_dim is not None:
            if cross_attn_dim >= _SDXL_TEXT_DIM:
                return (
                    True,
                    f"LoRA '{lora_path.name}' is SDXL-compatible "
                    f"(cross-attention input dim={cross_attn_dim}).",
                )
            if cross_attn_dim == _SD15_TEXT_DIM:
                return (
                    False,
                    "Loaded LoRA is not SDXL-compatible. Please provide an SDXL LoRA. "
                    f"(Cross-attention input dim={cross_attn_dim} matches SD1.5, "
                    "not SDXL which requires dim=2048.)",
                )
            # Cross-attn dim found but neither 768 nor 2048 — reject conservatively
            return (
                False,
                "Loaded LoRA is not SDXL-compatible. Please provide an SDXL LoRA. "
                f"(Unexpected cross-attention input dim={cross_attn_dim}; "
                f"SDXL requires 2048, SD1.5 uses 768.)",
            )

        # ── 2c. Max-dimension heuristic — STRICT fallback ─────────────────
        #
        # SDXL cross-attention requires 2048-dim text embeddings. Any genuine
        # SDXL LoRA that touches text cross-attention will have a 2048-dim
        # tensor. SD1.5's largest UNet channel is 1280.
        #
        # Decision boundaries:
        #   max_dim >= 2048  → SDXL-compatible (required cross-attn dim present)
        #   max_dim <= 1280  → SD1.5 (max SD1.5 UNet channel is 1280)
        #   1280 < max_dim < 2048 → ambiguous; reject conservatively to prevent
        #                            matmul errors at inference time
        max_dim = _max_tensor_dim(keys, lora_path)
        if max_dim == 0:
            return False, f"LoRA file '{lora_path.name}' contains no readable tensors."

        if max_dim >= _SDXL_TEXT_DIM:
            return (
                True,
                f"LoRA '{lora_path.name}' likely SDXL-compatible "
                f"(max tensor dim={max_dim} ≥ {_SDXL_TEXT_DIM}).",
            )

        if max_dim <= _SD15_MAX_UNET_DIM:
            # max_dim ≤ 1280 is consistent with SD1.5 architecture only.
            return (
                False,
                "Loaded LoRA is not SDXL-compatible. Please provide an SDXL LoRA. "
                f"(Max tensor dimension {max_dim} ≤ {_SD15_MAX_UNET_DIM}, "
                f"consistent with SD1.5 architecture. "
                f"SDXL LoRAs have cross-attention tensors with dim={_SDXL_TEXT_DIM}.)",
            )

        # 1280 < max_dim < 2048: ambiguous range — reject conservatively.
        # This prevents false positives from LoRAs with unusual tensor shapes
        # that could cause matmul errors at inference time.
        logger.warning(
            "LoRA '%s': architecture undetermined (max_dim=%d, range 1280–2048). "
            "Rejecting conservatively to prevent inference-time matmul errors. "
            "Provide an SDXL LoRA with metadata or 2048-dim cross-attention tensors.",
            lora_path.name,
            max_dim,
        )
        return (
            False,
            "Loaded LoRA is not SDXL-compatible. Please provide an SDXL LoRA. "
            f"(Architecture undetermined: max tensor dim={max_dim} is in the "
            f"ambiguous range 1280–2048. Rejecting conservatively to prevent "
            f"inference-time matmul errors.)",
        )

    except Exception as exc:  # noqa: BLE001
        return (
            False,
            f"Failed to validate LoRA '{lora_path.name}': {exc}",
        )


# ── Internal helpers ────────────────────────────────────────────────────────────

def _detect_cross_attn_dim(keys: list, lora_path: Path) -> int | None:
    """
    Scan LoRA keys for cross-attention projection tensors and return the
    input dimension, or ``None`` if no such keys are found.

    Handles multiple naming conventions:
    - kohya_ss:  ``...attn2.to_k.lora_down.weight``
    - diffusers: ``...attn2_to_k.lora_down.weight``
    - flat:      ``lora_unet_..._attn2_to_k.lora_down.weight``
    """
    try:
        from safetensors import safe_open

        with safe_open(str(lora_path), framework="pt", device="cpu") as f:
            for key in keys:
                if not _is_cross_attn_lora_down_key(key):
                    continue
                tensor = f.get_tensor(key)
                if len(tensor.shape) >= 2:
                    in_dim = int(tensor.shape[-1])
                    logger.debug(
                        "LoRA cross-attn key '%s' → in_dim=%d", key, in_dim
                    )
                    return in_dim
    except Exception as exc:  # noqa: BLE001
        logger.debug("Cross-attn dimension probe failed: %s", exc)
    return None


def _is_cross_attn_lora_down_key(key: str) -> bool:
    """
    Return True if *key* corresponds to a cross-attention lora_down weight.

    Matches keys from multiple training frameworks (kohya_ss, diffusers,
    CompVis) that encode the text→UNet cross-attention projection weights.
    """
    key_lower = key.lower()
    # Must be a lora_down weight
    if "lora_down" not in key_lower:
        return False
    # Must be a cross-attention (attn2) key/value projection
    if "attn2" not in key_lower:
        return False
    # Must target a projection layer
    if "to_k" not in key_lower and "to_v" not in key_lower:
        return False
    return True


def _max_tensor_dim(keys: list, lora_path: Path) -> int:
    """Return the maximum single dimension seen across all tensors."""
    max_dim = 0
    try:
        from safetensors import safe_open

        with safe_open(str(lora_path), framework="pt", device="cpu") as f:
            for key in keys:
                tensor = f.get_tensor(key)
                for dim in tensor.shape:
                    if dim > max_dim:
                        max_dim = dim
    except Exception as exc:  # noqa: BLE001
        logger.debug("Max-dim probe failed: %s", exc)
    return max_dim
