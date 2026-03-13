"""
Character Reference Encoder for the Pokemon Stencil Art Factory.

Encodes a set of reference images into a single averaged latent tensor
using the Stable Diffusion VAE encoder.  The resulting latent can be
used as the ``image`` argument of an img2img pipeline to preserve
character appearance across diverse scene compositions.

Workflow
--------
1. Load up to *max_images* reference PNG/JPG images.
2. Resize each to model resolution (default 768×768).
3. Encode every image through the VAE encoder to obtain a latent tensor.
4. Average all latent tensors element-wise into a single *reference latent*.
5. Decode the averaged latent back to a PIL Image for visual inspection.

If VAE encoding fails (e.g. diffusers / torch not available) the module
falls back to pixel-space averaging of the reference images.

Usage::

    from pokemon_stencil.image_gen.reference_encoder import CharacterReferenceEncoder

    encoder = CharacterReferenceEncoder(vae=pipe.vae, image_size=(768, 768))
    ref_image = encoder.encode_references(ref_images)   # → PIL Image
    # Use ref_image as img2img conditioning
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class CharacterReferenceEncoder:
    """
    Encodes multiple reference images into a single representative image
    by averaging in VAE latent space (with pixel-space fallback).

    Args:
        vae: A ``diffusers`` VAE model (e.g. ``pipe.vae``).  When ``None``
            the encoder always uses pixel-space blending.
        image_size: ``(width, height)`` to resize references to before
            encoding.  Should match the generation resolution.
        max_images: Maximum number of reference images to load and average.
            Additional images beyond this limit are ignored.
    """

    def __init__(
        self,
        vae=None,
        image_size: Tuple[int, int] = (768, 768),
        max_images: int = 30,
    ) -> None:
        self._vae = vae
        self._image_size = image_size
        self._max_images = max_images

    # ── Public API ─────────────────────────────────────────────────────────────

    def encode_references(self, images: List[Image.Image]) -> Optional[Image.Image]:
        """
        Encode *images* and return an averaged reference conditioning image.

        When fewer than 1 image is provided, returns ``None``.

        Args:
            images: Pre-loaded reference PIL Images (any size, any mode).

        Returns:
            A single ``RGB`` PIL Image representing the averaged reference, or
            ``None`` if *images* is empty.
        """
        if not images:
            logger.warning("CharacterReferenceEncoder: no reference images provided.")
            return None

        resized = self._resize_images(images[: self._max_images])
        logger.info(
            "CharacterReferenceEncoder: encoding %d reference image(s) at %s.",
            len(resized),
            self._image_size,
        )

        if self._vae is not None:
            result = self._encode_via_vae(resized)
            if result is not None:
                return result
            logger.warning(
                "VAE encoding failed; falling back to pixel-space blending."
            )

        return self._pixel_blend(resized)

    def encode_from_directory(self, directory: Path) -> Optional[Image.Image]:
        """
        Load reference images from *directory* and call :meth:`encode_references`.

        Args:
            directory: Path to a directory containing PNG/JPG reference files.

        Returns:
            Averaged reference image, or ``None`` if the directory is empty.
        """
        images = _load_images_from_dir(directory, self._max_images)
        if not images:
            logger.warning(
                "CharacterReferenceEncoder: no images found in '%s'.", directory
            )
            return None
        return self.encode_references(images)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _resize_images(self, images: List[Image.Image]) -> List[Image.Image]:
        """Resize all images to *self._image_size* using Lanczos resampling."""
        return [img.convert("RGB").resize(self._image_size, Image.LANCZOS) for img in images]

    def _encode_via_vae(self, images: List[Image.Image]) -> Optional[Image.Image]:
        """
        Encode *images* through the VAE encoder, average the latents, and
        decode back to a PIL Image.

        Returns ``None`` on any failure so the caller can fall back gracefully.
        """
        try:
            import torch

            latents: List = []
            for img in images:
                tensor = _pil_to_tensor(img).unsqueeze(0)  # (1, C, H, W)
                with torch.no_grad():
                    dist = self._vae.encode(tensor).latent_dist
                    latent = dist.sample()
                    latent = latent * self._vae.config.scaling_factor
                latents.append(latent)

            avg_latent = torch.stack(latents, dim=0).mean(dim=0)  # (1, C, h, w)

            # Decode back to pixel space.
            avg_latent = avg_latent / self._vae.config.scaling_factor
            with torch.no_grad():
                decoded = self._vae.decode(avg_latent).sample  # (1, C, H, W)

            decoded_np = (
                (decoded.squeeze(0).permute(1, 2, 0).numpy() * 0.5 + 0.5)
                .clip(0, 1)
            )
            decoded_np = (decoded_np * 255).astype(np.uint8)
            return Image.fromarray(decoded_np)

        except Exception as exc:
            logger.debug("VAE latent encoding error: %s", exc, exc_info=True)
            return None

    @staticmethod
    def _pixel_blend(images: List[Image.Image]) -> Image.Image:
        """
        Average *images* in pixel space.

        Simple mean across all images; each image is assumed to be the
        same size (already resized by the caller).

        Args:
            images: List of equal-sized RGB PIL Images.

        Returns:
            Averaged RGB PIL Image.
        """
        arrays = [np.array(img, dtype=np.float32) for img in images]
        avg = np.mean(np.stack(arrays, axis=0), axis=0).astype(np.uint8)
        result = Image.fromarray(avg)
        logger.debug("Pixel-space blend of %d image(s) complete.", len(images))
        return result


# ── Utilities ──────────────────────────────────────────────────────────────────

def _pil_to_tensor(image: Image.Image):
    """Convert a PIL Image to a normalised float32 torch Tensor (C, H, W)."""
    import torch
    arr = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0  # normalise to [-1, 1]
    return torch.from_numpy(arr.transpose(2, 0, 1))  # (C, H, W)


def _load_images_from_dir(directory: Path, max_images: int) -> List[Image.Image]:
    """Load up to *max_images* PNG/JPG images from *directory*."""
    if not directory.is_dir():
        return []
    suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    paths = sorted(
        (p for p in directory.iterdir() if p.suffix.lower() in suffixes),
        key=lambda p: p.name,
    )[:max_images]
    images: List[Image.Image] = []
    for p in paths:
        try:
            images.append(Image.open(p).convert("RGB"))
        except Exception as exc:
            logger.warning("Could not load reference image %s: %s", p, exc)
    return images
