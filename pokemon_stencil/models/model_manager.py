"""
Model manager for the Pokemon Stencil Art Factory.

Handles automatic downloading, local caching, and directory structure
management for all models in the generation stack:

- DreamShaper base model          (``models/dreamshaper/``)
- ControlNet OpenPose              (``models/controlnet_openpose/``)
- ControlNet Canny                 (``models/controlnet_canny/``)

Usage::

    from pokemon_stencil.models.model_manager import ModelManager
    manager = ModelManager()
    manager.ensure_models()          # download any missing models
    info = manager.status()          # check which are locally available
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

from pokemon_stencil.config import (
    DEFAULT_CONTROLNET_CANNY_HUB_ID,
    DEFAULT_CONTROLNET_CANNY_LOCAL_PATH,
    DEFAULT_CONTROLNET_OPENPOSE_HUB_ID,
    DEFAULT_CONTROLNET_OPENPOSE_LOCAL_PATH,
    DEFAULT_DREAMSHAPER_HUB_ID,
    DEFAULT_DREAMSHAPER_LOCAL_PATH,
)

logger = logging.getLogger(__name__)


# ── Model registry ─────────────────────────────────────────────────────────────

#: Ordered list of (local_path, hub_id, human_label) tuples describing every
#: model in the required stack.
_MODEL_REGISTRY: List[tuple[Path, str, str]] = [
    (
        DEFAULT_DREAMSHAPER_LOCAL_PATH,
        DEFAULT_DREAMSHAPER_HUB_ID,
        "DreamShaper base model",
    ),
    (
        DEFAULT_CONTROLNET_OPENPOSE_LOCAL_PATH,
        DEFAULT_CONTROLNET_OPENPOSE_HUB_ID,
        "ControlNet OpenPose",
    ),
    (
        DEFAULT_CONTROLNET_CANNY_LOCAL_PATH,
        DEFAULT_CONTROLNET_CANNY_HUB_ID,
        "ControlNet Canny",
    ),
]


class ModelManager:
    """
    Manages model downloads and local caching for the generation stack.

    Args:
        models_root: Override the root directory under which all model
            sub-directories are created.  Defaults to the values defined in
            :mod:`~pokemon_stencil.config`.
    """

    def __init__(self, models_root: Optional[Path] = None) -> None:
        self._models_root = models_root

    # ── Public API ─────────────────────────────────────────────────────────────

    def ensure_models(self, download_missing: bool = True) -> Dict[str, bool]:
        """
        Check that all required models are present locally; download any
        that are missing if *download_missing* is ``True``.

        Args:
            download_missing: When ``True`` (default) missing models are
                fetched from HuggingFace Hub using ``huggingface_hub``.

        Returns:
            Mapping of ``hub_id → is_locally_available`` after the method
            completes.
        """
        availability: Dict[str, bool] = {}
        for local_path, hub_id, label in self._registry():
            present = _is_populated(local_path)
            if not present and download_missing:
                logger.info("Downloading %s from Hub (%s) → %s", label, hub_id, local_path)
                self._download(hub_id, local_path)
                present = _is_populated(local_path)
            availability[hub_id] = present
            _log_status(label, hub_id, local_path, present)
        return availability

    def status(self) -> Dict[str, bool]:
        """
        Return the current local availability of each model without
        downloading anything.

        Returns:
            Mapping of ``hub_id → True/False``.
        """
        return {
            hub_id: _is_populated(local_path)
            for local_path, hub_id, _ in self._registry()
        }

    def ensure_directories(self) -> None:
        """Create local model directories if they do not yet exist."""
        for local_path, _, label in self._registry():
            local_path.mkdir(parents=True, exist_ok=True)
            logger.debug("Ensured directory for %s: %s", label, local_path)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _registry(self) -> List[tuple[Path, str, str]]:
        """Return the model registry, optionally re-rooted under *models_root*."""
        if self._models_root is None:
            return _MODEL_REGISTRY
        # Re-root: replace the first component of each local path.
        re_rooted = []
        for local_path, hub_id, label in _MODEL_REGISTRY:
            new_path = self._models_root / local_path.name
            re_rooted.append((new_path, hub_id, label))
        return re_rooted

    @staticmethod
    def _download(hub_id: str, local_path: Path) -> None:
        """
        Download a model from HuggingFace Hub to *local_path*.

        Uses ``huggingface_hub.snapshot_download`` so the entire model
        repository (weights, config, tokeniser) is cached locally.

        Args:
            hub_id: HuggingFace Hub model ID (e.g. ``"Lykon/DreamShaper"``).
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
            snapshot_download(
                repo_id=hub_id,
                local_dir=str(local_path),
                local_dir_use_symlinks=False,
            )
            logger.info("Downloaded %s to %s", hub_id, local_path)
        except Exception as exc:
            logger.error("Failed to download %s: %s", hub_id, exc)
            raise RuntimeError(f"Model download failed for {hub_id}: {exc}") from exc


# ── Utilities ──────────────────────────────────────────────────────────────────

def _is_populated(path: Path) -> bool:
    """Return ``True`` if *path* is a non-empty directory."""
    return path.is_dir() and any(path.iterdir())


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
