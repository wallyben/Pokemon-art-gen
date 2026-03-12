"""
Multi-reference conditioning encoder.

Combines multiple Pokémon reference images into a single conditioning input
for image generation, improving identity / character consistency across
generated candidates.

Two strategies are supported (selected automatically at runtime):

1. **VAE latent averaging** – If a cached Stable Diffusion VAE is accessible
   via the model-loader cache, each reference is encoded to a latent vector
   and the vectors are averaged.  The mean latent is returned as a tensor
   that can be fed directly as ``latents`` in an img2img pipeline call.

2. **Pixel-blend fallback** – When the VAE is unavailable (e.g. model weights
   not yet downloaded), reference images are resized to the target resolution
   and averaged at the pixel level.  The blended image is returned as a PIL
   Image that can be used as the ``image`` argument in an img2img pipeline.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class ReferenceEncoder:
    """
    Encodes a list of reference images into a conditioning representation.

    Args:
        target_size: ``(width, height)`` in pixels.  All references are
            resized to this resolution before encoding.
    """

    def __init__(self, target_size: Tuple[int, int] = (512, 512)) -> None:
        self.target_size = target_size

    # ── Public API ────────────────────────────────────────────────────────────

    def encode(
        self,
        reference_images: List[Image.Image],
        vae: Optional[object] = None,
    ) -> Union["torch.Tensor", Image.Image]:  # noqa: F821
        """
        Produce a conditioning representation from *reference_images*.

        Tries the VAE latent strategy first; falls back to pixel blending.

        Args:
            reference_images: One or more reference PIL Images (any mode).
            vae: Optional Stable Diffusion ``AutoencoderKL`` instance.
                 When ``None`` the pixel-blend fallback is used.

        Returns:
            A ``torch.Tensor`` of averaged latents when the VAE path succeeds,
            or a PIL Image (pixel blend) when it does not.

        Raises:
            ValueError: If *reference_images* is empty.
        """
        if not reference_images:
            raise ValueError("reference_images must not be empty.")

        resized = self._resize_all(reference_images)

        if vae is not None:
            try:
                return self._encode_vae(resized, vae)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "VAE encoding failed (%s); falling back to pixel blend.",
                    exc,
                )

        return self._pixel_blend(resized)

    def blend(self, reference_images: List[Image.Image]) -> Image.Image:
        """
        Compute a pixel-level average of *reference_images*.

        Useful as a lightweight conditioning input or as a preview of the
        combined reference identity.

        Args:
            reference_images: One or more PIL Images (any mode).

        Returns:
            Averaged PIL Image in RGB mode, resized to ``self.target_size``.

        Raises:
            ValueError: If *reference_images* is empty.
        """
        if not reference_images:
            raise ValueError("reference_images must not be empty.")
        resized = self._resize_all(reference_images)
        return self._pixel_blend(resized)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _resize_all(self, images: List[Image.Image]) -> List[Image.Image]:
        """Resize every image to ``self.target_size`` in RGB mode."""
        return [
            img.convert("RGB").resize(self.target_size, Image.LANCZOS)
            for img in images
        ]

    @staticmethod
    def _pixel_blend(images: List[Image.Image]) -> Image.Image:
        """
        Average a list of same-size RGB images at the pixel level.

        Args:
            images: RGB PIL Images of identical size.

        Returns:
            Averaged PIL Image in RGB mode.
        """
        arrays = [np.array(img, dtype=np.float32) for img in images]
        mean_arr = np.mean(arrays, axis=0).astype(np.uint8)
        return Image.fromarray(mean_arr, mode="RGB")

    @staticmethod
    def _encode_vae(
        images: List[Image.Image],
        vae: object,
    ) -> "torch.Tensor":  # noqa: F821
        """
        Encode images using a Stable Diffusion VAE and return the mean latent.

        Args:
            images: RGB PIL Images pre-resized to the target resolution.
            vae: An ``AutoencoderKL`` instance from the ``diffusers`` library.

        Returns:
            Averaged latent tensor, shape ``(1, 4, H//8, W//8)``.
        """
        import torch

        latents: List["torch.Tensor"] = []
        for img in images:
            arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
            tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                dist = vae.encode(tensor)
                # diffusers VAEOutput exposes either .latent_dist.mean or .latent_dist
                if hasattr(dist, "latent_dist"):
                    latent = dist.latent_dist.mean
                else:
                    latent = dist.mean
            latents.append(latent)

        mean_latent = torch.stack(latents).mean(dim=0)
        logger.debug(
            "VAE encoded %d reference(s) → mean latent shape %s.",
            len(images),
            tuple(mean_latent.shape),
        )
        return mean_latent
