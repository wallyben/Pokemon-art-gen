"""
Style transfer utilities for enforcing stencil-friendly visual properties.

Rather than neural style transfer (which is expensive on CPU), this module
applies a series of deterministic image-processing steps that push generated
or reference images toward the flat, high-contrast look required for clean
stencil cutting:

  1. Palette quantisation  – reduce to N dominant colours
  2. Posterisation         – round each channel to discrete steps
  3. Contrast enhancement  – CLAHE on the L channel
  4. Edge sharpening       – unsharp mask

These transforms are intentionally lightweight so they run well on a
Ryzen 7 CPU with integrated graphics.
"""

import logging
from typing import Tuple

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

from pokemon_stencil.config import ProcessingConfig

logger = logging.getLogger(__name__)


class StencilStyleTransfer:
    """
    Applies stencil-friendly style adjustments to a PIL Image.

    All operations are CPU-only and use OpenCV / Pillow.
    """

    def __init__(self, config: ProcessingConfig) -> None:
        """
        Args:
            config: ProcessingConfig controlling filter parameters.
        """
        self.config = config

    def apply(self, image: Image.Image) -> Image.Image:
        """
        Run the full style-transfer pipeline on *image*.

        Args:
            image: Input PIL Image (RGB).

        Returns:
            Processed PIL Image with stencil-friendly aesthetics.
        """
        img = image.convert("RGB")
        img = self._resize(img)
        img = self._bilateral_smooth(img)
        img = self._enhance_contrast(img)
        img = self._posterise(img, levels=4)
        img = self._sharpen_edges(img)
        logger.debug("Style transfer complete.")
        return img

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _resize(self, img: Image.Image) -> Image.Image:
        """Resize to the configured output size using Lanczos resampling."""
        target = self.config.output_size
        if img.size != target:
            img = img.resize(target, Image.LANCZOS)
        return img

    def _bilateral_smooth(self, img: Image.Image) -> Image.Image:
        """
        Apply repeated bilateral filtering to smooth within regions while
        preserving hard edges – the key to clean stencil colour areas.
        """
        arr = np.array(img, dtype=np.uint8)
        for _ in range(self.config.bilateral_passes):
            arr = cv2.bilateralFilter(
                arr,
                d=self.config.bilateral_d,
                sigmaColor=self.config.bilateral_sigma_color,
                sigmaSpace=self.config.bilateral_sigma_space,
            )
        return Image.fromarray(arr)

    def _enhance_contrast(self, img: Image.Image) -> Image.Image:
        """Apply CLAHE to the L channel to boost local contrast."""
        arr = np.array(img, dtype=np.uint8)
        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        lab = cv2.merge([l_ch, a_ch, b_ch])
        arr = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        return Image.fromarray(arr)

    @staticmethod
    def _posterise(img: Image.Image, levels: int = 4) -> Image.Image:
        """
        Reduce each channel to *levels* discrete values.

        This approximates the look of cel-shaded / stencil art by flattening
        tonal gradients into broad, solid colour areas.
        """
        arr = np.array(img, dtype=np.float32)
        step = 255.0 / (levels - 1)
        arr = np.round(arr / step) * step
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr)

    @staticmethod
    def _sharpen_edges(img: Image.Image) -> Image.Image:
        """Apply an unsharp mask to crisp up boundary lines."""
        return img.filter(ImageFilter.UnsharpMask(radius=1, percent=150, threshold=3))
