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
3. **Max-dimension heuristic** — if no cross-attention keys are found, the
   maximum tensor dimension is used as a fallback signal.

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

#: Substrings that appear in metadata when the base model is SDXL.
_SDXL_METADATA_KEYWORDS = ("sd_xl", "sdxl", "xl_base", "stable_diffusion_xl", "xl-base")

#: Substrings that appear in metadata when the base model is SD 1.5.
_SD15_METADATA_KEYWORDS = (
    "sd-1", "sd1", "v1-5", "v1.5", "stable_diffusion_v1", "stable-diffusion-v1",
    "sd_1", "sd15",
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

        # ── 2c. Max-dimension heuristic (fallback) ────────────────────────
        max_dim = _max_tensor_dim(keys, lora_path)
        if max_dim == 0:
            return False, f"LoRA file '{lora_path.name}' contains no readable tensors."

        if max_dim >= _SDXL_TEXT_DIM:
            return (
                True,
                f"LoRA '{lora_path.name}' likely SDXL-compatible "
                f"(max tensor dim={max_dim} ≥ {_SDXL_TEXT_DIM}).",
            )
        if max_dim <= _SD15_TEXT_DIM:
            return (
                False,
                "Loaded LoRA is not SDXL-compatible. Please provide an SDXL LoRA. "
                f"(Max tensor dimension {max_dim} is consistent with SD1.5 "
                f"architecture, not SDXL which uses dims up to {_SDXL_TEXT_DIM}.)",
            )

        # Cannot determine — accept with a warning
        logger.warning(
            "LoRA '%s': architecture could not be determined (max_dim=%d). "
            "Proceeding cautiously — dimension mismatch errors may occur at runtime.",
            lora_path.name,
            max_dim,
        )
        return (
            True,
            f"LoRA '{lora_path.name}' architecture undetermined "
            f"(max_dim={max_dim}); proceeding with caution.",
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
    """
    try:
        from safetensors import safe_open

        with safe_open(str(lora_path), framework="pt", device="cpu") as f:
            for key in keys:
                # Target: cross-attention key or value projections, lora_down side
                if "attn2" in key and ("to_k" in key or "to_v" in key) and "lora_down" in key:
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
