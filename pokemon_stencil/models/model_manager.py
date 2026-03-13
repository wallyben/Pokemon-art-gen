"""
Model manager for the Pokemon Stencil Art Factory (v3 – SDXL).

Handles automatic downloading, local caching, integrity verification, and
directory structure management for all models in the SDXL generation stack:

Directory structure (per spec)
-------------------------------
    models/
        sdxl/               stabilityai/stable-diffusion-xl-base-1.0
        controlnet_pose/    thibaud/controlnet-openpose-sdxl-1.0
        controlnet_canny/   diffusers/controlnet-canny-sdxl-1.0
        ip_adapter/         h94/IP-Adapter
        lora/               user-supplied .safetensors files

Models are downloaded automatically on first use via ``huggingface_hub``.

Usage::

    from pokemon_stencil.models.model_manager import ModelManager
    manager = ModelManager()
    manager.ensure_models()          # download any missing models
    info = manager.status()          # check which are locally available
    loras = manager.list_loras()     # enumerate available LoRA files
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pokemon_stencil.config import (
    DEFAULT_CONTROLNET_CANNY_HUB_ID,
    DEFAULT_CONTROLNET_CANNY_LOCAL_PATH,
    DEFAULT_CONTROLNET_OPENPOSE_HUB_ID,
    DEFAULT_CONTROLNET_OPENPOSE_LOCAL_PATH,
    DEFAULT_IP_ADAPTER_HUB_ID,
    DEFAULT_IP_ADAPTER_LOCAL_PATH,
    DEFAULT_LORA_DIR,
    DEFAULT_SDXL_HUB_ID,
    DEFAULT_SDXL_LOCAL_PATH,
)

logger = logging.getLogger(__name__)

# ── Model registry ─────────────────────────────────────────────────────────────

#: Ordered list of (local_path, hub_id, human_label) tuples.
_MODEL_REGISTRY: List[Tuple[Path, str, str]] = [
    (
        DEFAULT_SDXL_LOCAL_PATH,
        DEFAULT_SDXL_HUB_ID,
        "SDXL base model",
    ),
    (
        DEFAULT_CONTROLNET_OPENPOSE_LOCAL_PATH,
        DEFAULT_CONTROLNET_OPENPOSE_HUB_ID,
        "ControlNet OpenPose (SDXL)",
    ),
    (
        DEFAULT_CONTROLNET_CANNY_LOCAL_PATH,
        DEFAULT_CONTROLNET_CANNY_HUB_ID,
        "ControlNet Canny (SDXL)",
    ),
    (
        DEFAULT_IP_ADAPTER_LOCAL_PATH,
        DEFAULT_IP_ADAPTER_HUB_ID,
        "IP-Adapter (SDXL reference conditioning)",
    ),
]

#: Required sub-paths within the IP-Adapter repo (integrity check).
_IP_ADAPTER_REQUIRED_FILES = [
    "sdxl_models/ip-adapter_sdxl.bin",
    "sdxl_models/ip-adapter-plus_sdxl_vit-h.bin",
]


class ModelManager:
    """
    Manages model downloads, local caching, and verification.

    Args:
        models_root: Override the root directory.  When set, all local
            paths from the registry are re-rooted under this directory.
    """

    def __init__(self, models_root: Optional[Path] = None) -> None:
        self._models_root = models_root

    # ── Public API ─────────────────────────────────────────────────────────────

    def ensure_models(self, download_missing: bool = True) -> Dict[str, bool]:
        """
        Check all required models are present locally; download missing ones.

        Args:
            download_missing: When ``True`` (default) fetch missing models
                from HuggingFace Hub.

        Returns:
            Mapping of ``hub_id → is_locally_available``.
        """
        availability: Dict[str, bool] = {}
        for local_path, hub_id, label in self._registry():
            present = _is_populated(local_path)
            if not present and download_missing:
                logger.info(
                    "Downloading %s from Hub (%s) → %s",
                    label, hub_id, local_path,
                )
                self._download(hub_id, local_path)
                present = _is_populated(local_path)
            availability[hub_id] = present
            _log_status(label, hub_id, local_path, present)
        return availability

    def ensure_directories(self) -> None:
        """Create all model directories including the lora/ subdirectory."""
        dirs = [local_path for local_path, _, _ in self._registry()]
        dirs.append(self._lora_dir())
        for path in dirs:
            path.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured directory: %s", path)

    def status(self) -> Dict[str, bool]:
        """
        Return current local availability without downloading.

        Returns:
            Mapping of ``hub_id → True/False``.
        """
        return {
            hub_id: _is_populated(local_path)
            for local_path, hub_id, _ in self._registry()
        }

    def verify_integrity(self) -> Dict[str, bool]:
        """
        Verify that critical model files exist in each local directory.

        Performs a shallow check (file existence, not checksum) to quickly
        detect incomplete downloads.

        Returns:
            Mapping of ``hub_id → integrity_ok``.
        """
        results: Dict[str, bool] = {}
        for local_path, hub_id, label in self._registry():
            ok = _check_integrity(local_path, hub_id)
            results[hub_id] = ok
            if not ok:
                logger.warning(
                    "Integrity check FAILED for %s (%s). "
                    "Re-run ensure_models() to re-download.",
                    label, hub_id,
                )
            else:
                logger.debug("Integrity OK: %s", label)
        return results

    def list_loras(self) -> List[Path]:
        """
        Return all top-level ``.safetensors`` files in the LoRA directory.

        Only direct children of the lora directory are returned — cache
        subdirectories (``.cache/``), lock files (``.lock``), and metadata
        files (``.meta``) are excluded by design (non-recursive glob).
        """
        lora_dir = self._lora_dir()
        if not lora_dir.is_dir():
            return []
        # Non-recursive: only direct children with .safetensors extension.
        # This excludes .cache/, .lock, .meta and other non-model files.
        return sorted(
            p for p in lora_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".safetensors"
        )

    def lora_names(self) -> List[str]:
        """Return human-readable names (stems) of available LoRA files."""
        return [p.stem for p in self.list_loras()]

    def list_loras_validated(self) -> List[Path]:
        """
        Return only LoRA files that pass SDXL compatibility validation.

        This is a strict subset of ``list_loras()``: any SD1.5, ambiguous,
        or corrupt LoRA files are silently excluded.  Use this when you want
        to populate a UI dropdown with only safe, usable LoRAs.

        Returns:
            Sorted list of ``Path`` objects for SDXL-compatible LoRAs.
        """
        try:
            from pokemon_stencil.models.lora_validator import validate_lora_sdxl_compatible
        except ImportError:
            logger.warning(
                "lora_validator not available; returning all LoRAs unvalidated."
            )
            return self.list_loras()

        validated: List[Path] = []
        for lora_path in self.list_loras():
            ok, msg = validate_lora_sdxl_compatible(lora_path)
            if ok:
                validated.append(lora_path)
            else:
                logger.debug(
                    "LoRA '%s' excluded from validated list: %s",
                    lora_path.name, msg,
                )
        return validated

    def lora_names_validated(self) -> List[str]:
        """Return stems of SDXL-compatible LoRA files only."""
        return [p.stem for p in self.list_loras_validated()]

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _registry(self) -> List[Tuple[Path, str, str]]:
        """Return registry, optionally re-rooted under *models_root*."""
        if self._models_root is None:
            return _MODEL_REGISTRY
        re_rooted = []
        for local_path, hub_id, label in _MODEL_REGISTRY:
            new_path = self._models_root / local_path.name
            re_rooted.append((new_path, hub_id, label))
        return re_rooted

    def _lora_dir(self) -> Path:
        if self._models_root is not None:
            return self._models_root / DEFAULT_LORA_DIR.name
        return DEFAULT_LORA_DIR

    @staticmethod
    def _download(hub_id: str, local_path: Path) -> None:
        """
        Download a model repository from HuggingFace Hub to *local_path*.

        Uses ``snapshot_download`` to fetch the complete repository including
        weights, config files, and tokeniser.

        Args:
            hub_id: HuggingFace Hub model ID.
            local_path: Target directory on disk.

        Raises:
            ImportError: If ``huggingface_hub`` is not installed.
            RuntimeError: If the download fails.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ImportError(
                "huggingface_hub is required for automatic model downloads.\n"
                "Install with:  pip install huggingface_hub"
            ) from exc

        local_path.mkdir(parents=True, exist_ok=True)
        try:
            # ``local_dir_use_symlinks`` was deprecated in huggingface_hub 0.21
            # and removed in 0.23.  Try the old kwarg first; fall back to the
            # kwarg-free call so the code works across the full 0.21–0.22 range.
            try:
                snapshot_download(
                    repo_id=hub_id,
                    local_dir=str(local_path),
                    local_dir_use_symlinks=False,
                )
            except TypeError:
                snapshot_download(
                    repo_id=hub_id,
                    local_dir=str(local_path),
                )
            logger.info("Downloaded %s to %s", hub_id, local_path)
        except Exception as exc:
            logger.error("Failed to download %s: %s", hub_id, exc)
            raise RuntimeError(
                f"Model download failed for {hub_id}: {exc}"
            ) from exc


# ── Utilities ──────────────────────────────────────────────────────────────────

def _is_populated(path: Path) -> bool:
    """Return ``True`` if *path* is a non-empty directory."""
    return path.is_dir() and any(path.iterdir())


def _check_integrity(local_path: Path, hub_id: str) -> bool:
    """
    Shallow integrity check – verify expected config files exist.

    For most models we just check that ``config.json`` is present.
    For IP-Adapter we additionally check the SDXL weight files.
    """
    if not local_path.is_dir():
        return False

    # Every HF model repo should have a config.json or model_index.json.
    has_config = (
        (local_path / "config.json").exists()
        or (local_path / "model_index.json").exists()
    )
    if not has_config:
        return False

    # Extra check for IP-Adapter SDXL weights.
    if "IP-Adapter" in hub_id or "ip-adapter" in hub_id.lower():
        for rel_path in _IP_ADAPTER_REQUIRED_FILES:
            if not (local_path / rel_path).exists():
                logger.debug(
                    "IP-Adapter integrity: missing %s in %s", rel_path, local_path
                )
                return False

    return True


def _log_status(label: str, hub_id: str, local_path: Path, present: bool) -> None:
    if present:
        logger.info("✓ %s found at %s", label, local_path)
    else:
        logger.warning(
            "✗ %s not found locally (hub: %s). "
            "Run ModelManager().ensure_models() to download.",
            label,
            hub_id,
        )
