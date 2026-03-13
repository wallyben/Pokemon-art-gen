"""
pokemon_stencil.generation.fal_provider — fal.ai generation backend.

Primary production backend.  Uses FLUX.1 Dev via fal.ai's hosted inference
API: no local GPU, no dependency hell, Windows-friendly (pip install fal-client).

Setup::

    pip install fal-client
    # Then set your key — choose ONE method:
    $env:FAL_KEY = "your-key-here"        # PowerShell (current session)
    setx FAL_KEY "your-key-here"          # PowerShell (persistent, Windows)

Get your key at: https://fal.ai/dashboard/keys

Models used:
    fal-ai/flux/dev              — text-to-image when no reference supplied
    fal-ai/flux/dev/image-to-image — reference-conditioned generation

Reference conditioning:
    When reference_images are provided, the first reference is base64-encoded
    and sent to the img2img model.  This is the primary mechanism for
    character accuracy.  For highest accuracy provide a clean front-facing
    reference image.

Cost estimate (fal.ai pricing as of 2026):
    ~$0.003–0.05 per image depending on model/steps.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import urllib.request
from pathlib import Path
from typing import List, Optional

from pokemon_stencil.generation.base import (
    AbstractGenerationProvider,
    GenerationRequest,
    GenerationResult,
)

logger = logging.getLogger(__name__)

_FAL_T2I_MODEL = "fal-ai/flux/dev"
_FAL_I2I_MODEL = "fal-ai/flux/dev/image-to-image"

# fal.ai size identifiers supported by FLUX/dev
_SIZE_MAP = {
    (512, 512): "square",
    (768, 768): "square_hd",
    (1024, 1024): "square_hd",
    (768, 1360): "portrait_16_9",
    (1360, 768): "landscape_16_9",
}


def _fal_image_size(width: int, height: int) -> str:
    return _SIZE_MAP.get((width, height), "square_hd")


def _encode_image_b64(path: Path) -> str:
    """Return a data-URI suitable for fal.ai image_url fields."""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return f"data:image/png;base64,{data}"


def _download_image(url: str, dest: Path) -> bool:
    """Download *url* to *dest*.  Returns True on success."""
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            dest.write_bytes(resp.read())
        return True
    except Exception as exc:
        logger.warning("Failed to download image from fal CDN: %s", exc)
        return False


class FalProvider(AbstractGenerationProvider):
    """Generation backend powered by fal.ai FLUX.1 Dev.

    Character accuracy strategy:
    - With reference images: use img2img (flux/dev/image-to-image) so the
      output closely follows the character's shape and colour scheme.
    - Without references: use text-to-image with a detailed character
      description built by PokemonPromptBuilder.
    - reference_strength (default 0.80) balances faithfulness to the
      reference vs. prompt-driven creativity.
    """

    @property
    def name(self) -> str:
        return "fal"

    def is_available(self) -> bool:
        """Return True if FAL_KEY is set and fal_client is importable."""
        if not os.environ.get("FAL_KEY"):
            return False
        try:
            import fal_client  # noqa: F401
            return True
        except ImportError:
            return False

    def generate(
        self,
        request: GenerationRequest,
        output_dir: Path,
    ) -> GenerationResult:
        """Call fal.ai API and download result images to *output_dir*."""
        self._ensure_output_dir(output_dir)

        try:
            import fal_client
        except ImportError:
            logger.error(
                "fal-client not installed. Run: pip install fal-client"
            )
            return GenerationResult(images=[], provider=self.name)

        if not os.environ.get("FAL_KEY"):
            logger.error("FAL_KEY environment variable not set.")
            return GenerationResult(images=[], provider=self.name)

        # ── Choose model & arguments ──────────────────────────────────────────
        has_refs = bool(request.reference_images)
        valid_refs = [r for r in request.reference_images if r.exists()]

        saved: List[Path] = []
        metadata: dict = {"model": "", "provider": "fal"}

        for img_idx in range(request.num_images):
            seed = (request.seed + img_idx) if request.seed is not None else None

            if valid_refs:
                # ── Image-to-image with reference conditioning ────────────────
                ref_path = valid_refs[0]
                ref_data_uri = _encode_image_b64(ref_path)
                model = _FAL_I2I_MODEL
                args = {
                    "prompt": request.prompt,
                    "image_url": ref_data_uri,
                    "strength": float(request.reference_strength),
                    "num_inference_steps": request.num_steps,
                    "guidance_scale": request.guidance_scale,
                    "num_images": 1,
                }
                if seed is not None:
                    args["seed"] = seed
                logger.info(
                    "[fal] img2img with reference '%s', strength=%.2f",
                    ref_path.name,
                    request.reference_strength,
                )
            else:
                # ── Text-to-image ─────────────────────────────────────────────
                model = _FAL_T2I_MODEL
                args = {
                    "prompt": request.prompt,
                    "image_size": _fal_image_size(request.width, request.height),
                    "num_inference_steps": request.num_steps,
                    "guidance_scale": request.guidance_scale,
                    "num_images": 1,
                    "enable_safety_checker": False,
                }
                if seed is not None:
                    args["seed"] = seed
                logger.info("[fal] text-to-image, prompt='%s...'", request.prompt[:60])

            # ── Submit to fal.ai ──────────────────────────────────────────────
            try:
                result = fal_client.submit(model, arguments=args).get()
                metadata["model"] = model

                img_list = result.images if hasattr(result, "images") else result.get("images", [])
                if not img_list:
                    logger.warning("[fal] No images returned for index %d", img_idx)
                    continue

                img_info = img_list[0]
                img_url = img_info.url if hasattr(img_info, "url") else img_info.get("url")

                dest = self._image_path(output_dir, img_idx)
                if _download_image(img_url, dest):
                    saved.append(dest)
                    logger.info("[fal] Saved image to %s", dest)

            except Exception as exc:
                logger.error("[fal] Generation failed for index %d: %s", img_idx, exc)

        return GenerationResult(images=saved, provider=self.name, metadata=metadata)
