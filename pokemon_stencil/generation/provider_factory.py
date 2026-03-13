"""
pokemon_stencil.generation.provider_factory — select the right generation backend.

Provider selection order for provider="auto":
    1. fal.ai  — if FAL_KEY is set and fal-client is installed
    2. local_sdxl — if torch + diffusers are available

Usage::

    from pokemon_stencil.generation import get_provider
    provider = get_provider()           # auto-select
    provider = get_provider("fal")      # force fal.ai
    provider = get_provider("local")    # force local SDXL
"""

from __future__ import annotations

import logging
from typing import Optional

from pokemon_stencil.generation.base import AbstractGenerationProvider

logger = logging.getLogger(__name__)


def get_provider(
    provider: str = "auto",
) -> AbstractGenerationProvider:
    """Return the best available generation provider.

    Args:
        provider: One of "auto", "fal", "local".

    Returns:
        A concrete provider instance.

    Raises:
        RuntimeError if the requested provider is unavailable and no fallback
        exists.
    """
    from pokemon_stencil.generation.fal_provider import FalProvider
    from pokemon_stencil.generation.local_provider import LocalSDXLProvider

    fal = FalProvider()
    local = LocalSDXLProvider()

    if provider == "fal":
        if not fal.is_available():
            raise RuntimeError(
                "fal provider requested but FAL_KEY is not set or fal-client is not "
                "installed.\n"
                "  Set key: $env:FAL_KEY = 'your-key'  (PowerShell)\n"
                "  Install:  pip install fal-client"
            )
        logger.info("Using fal.ai provider.")
        return fal

    if provider == "local":
        if not local.is_available():
            raise RuntimeError(
                "local_sdxl provider requested but torch/diffusers are not installed."
            )
        logger.info("Using local SDXL provider.")
        return local

    # auto — try fal first, then local
    if fal.is_available():
        logger.info("Auto-selected fal.ai provider (FAL_KEY found).")
        return fal

    if local.is_available():
        logger.warning(
            "No FAL_KEY found — falling back to local SDXL provider. "
            "Set FAL_KEY for faster, GPU-free generation via fal.ai."
        )
        return local

    raise RuntimeError(
        "No generation provider is available.\n"
        "Option A — Cloud (recommended, no GPU needed):\n"
        "  1. pip install fal-client\n"
        "  2. Set FAL_KEY environment variable (get key at https://fal.ai/dashboard/keys)\n"
        "\n"
        "Option B — Local GPU:\n"
        "  pip install torch diffusers transformers accelerate peft\n"
        "  (requires CUDA GPU; CPU inference takes hours per image)"
    )
