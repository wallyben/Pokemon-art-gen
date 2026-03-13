"""
Image simplification for stencil-ready artwork.

Converts a generated or reference Pokémon image into a clean, simplified
bitmap suitable for colour segmentation and SVG vector tracing.

Responsibilities
----------------
- Remove tiny shape islands (components below a minimum area threshold).
- Reduce colour noise (non-local means denoising on the LAB luminance channel).
- Enforce stencil-safe shapes (morphological cleanup, edge thinning).
- Preserve edges (bilateral filtering, Canny-guided thinning).

Uses OpenCV for all pixel-level operations and scikit-image utilities for
connected-component analysis.

Usage::

    from pokemon_stencil.image_gen.image_simplifier import ImageSimplifier
    from pokemon_stencil.config import ProcessingConfig

    simplifier = ImageSimplifier(ProcessingConfig())
    clean = simplifier.simplify(pil_image)
    edges = simplifier.extract_edges(pil_image)
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from pokemon_stencil.config import ProcessingConfig

logger = logging.getLogger(__name__)


class ImageSimplifier:
    """
    Reduces image complexity to stencil-ready flat colour regions.

    The simplification pipeline runs entirely on CPU using OpenCV and
    optionally scikit-image for component analysis.

    Args:
        config: :class:`~pokemon_stencil.config.ProcessingConfig` controlling
            filter parameters, morphology settings, and stencil safety thresholds.
    """

    def __init__(self, config: ProcessingConfig) -> None:
        self.config = config

    # ── Public API ─────────────────────────────────────────────────────────────

    def simplify(self, image: Image.Image) -> Image.Image:
        """
        Run the full stencil simplification pipeline on *image*.

        Pipeline stages:
        1. Bilateral smooth  – edge-preserving noise reduction.
        2. CLAHE contrast    – boost local contrast for cleaner segmentation.
        3. Posterisation     – quantise tones to flat colour bands.
        4. Morphological clean – open/close to remove speckling.
        5. Remove tiny islands – absorb small floating components.
        6. Reduce colour noise – NLM denoising on luminance channel.
        7. Median blur       – final smoothing pass.

        Args:
            image: Input PIL Image (RGB or RGBA).

        Returns:
            Simplified RGB PIL Image.
        """
        arr = self._to_bgr(image)
        arr = self._bilateral_smooth(arr)
        arr = self._clahe_contrast(arr)
        arr = self._posterise(arr, levels=self.config.n_colors * 2)
        arr = self._morphological_clean(arr)
        arr = self.remove_tiny_islands(arr)
        arr = self.reduce_colour_noise(arr)
        arr = cv2.medianBlur(arr, 5)
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

    def extract_edges(self, image: Image.Image) -> np.ndarray:
        """
        Extract a binary Canny edge map from *image*.

        Args:
            image: Input PIL Image (RGB or RGBA).

        Returns:
            Binary uint8 ndarray (0 or 255) representing detected edges.
        """
        arr = self._to_bgr(image)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        return cv2.Canny(blurred, self.config.canny_low, self.config.canny_high)

    def remove_tiny_islands(self, arr: np.ndarray) -> np.ndarray:
        """
        Absorb small isolated colour islands into their nearest neighbour.

        Components whose area falls below ``config.min_island_area`` pixels
        are flood-filled with the median colour of the surrounding border
        pixels, eliminating hairline-cut-inducing micro-regions.

        Args:
            arr: OpenCV BGR uint8 image.

        Returns:
            BGR image with small components merged into neighbours.
        """
        min_area = self.config.min_island_area
        result = arr.copy()
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        _, quant = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            quant, connectivity=8
        )
        merged = 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        for idx in range(1, num_labels):
            if stats[idx, cv2.CC_STAT_AREA] >= min_area:
                continue
            mask = (labels == idx).astype(np.uint8) * 255
            dilated = cv2.dilate(mask, kernel, iterations=2)
            border = cv2.bitwise_and(dilated, cv2.bitwise_not(mask))
            border_pixels = arr[border > 0]
            if border_pixels.size == 0:
                continue
            fill = np.median(border_pixels, axis=0).astype(np.uint8)
            result[mask > 0] = fill
            merged += 1
        if merged:
            logger.debug("Removed %d tiny island(s) (min_area=%d px).", merged, min_area)
        return result

    @staticmethod
    def reduce_colour_noise(arr: np.ndarray) -> np.ndarray:
        """
        Suppress isolated colour speckle on the LAB luminance channel.

        Applies fast non-local means denoising to the L channel only so
        hue boundaries remain sharp while flat fill regions become cleaner.

        Args:
            arr: OpenCV BGR uint8 image.

        Returns:
            Colour-denoised BGR image.
        """
        try:
            lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
            l_ch, a_ch, b_ch = cv2.split(lab)
            l_denoised = cv2.fastNlMeansDenoising(
                l_ch, h=7, templateWindowSize=7, searchWindowSize=21
            )
            return cv2.cvtColor(cv2.merge([l_denoised, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
        except Exception:
            return arr

    def enforce_stencil_shapes(self, image: Image.Image) -> Image.Image:
        """
        Apply stencil-safety morphological cleanup to *image*.

        Runs morphological open and close operations to remove thin bridges,
        close small holes, and ensure all regions are cuttable with a
        standard Cricut blade.

        Args:
            image: Input PIL Image (RGB or RGBA).

        Returns:
            Cleaned RGB PIL Image.
        """
        arr = self._to_bgr(image)
        arr = self._morphological_clean(arr)
        return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

    def preserve_edges(self, image: Image.Image) -> Image.Image:
        """
        Sharpen and preserve prominent edges while smoothing fill regions.

        Detects edge zones via Sobel magnitude and applies targeted bilateral
        smoothing only outside the edge mask, keeping boundaries crisp.

        Args:
            image: Input PIL Image (RGB or RGBA).

        Returns:
            Edge-preserved RGB PIL Image.
        """
        arr = self._to_bgr(image)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        sx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(sx, sy)
        mag_norm = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        _, edge_mask = cv2.threshold(
            mag_norm.astype(np.uint8), 30, 255, cv2.THRESH_BINARY
        )
        edge_mask3 = cv2.cvtColor(edge_mask, cv2.COLOR_GRAY2BGR)
        non_edge = cv2.bitwise_not(edge_mask3)
        smoothed = cv2.bilateralFilter(arr, d=5, sigmaColor=40, sigmaSpace=40)
        result = cv2.bitwise_and(arr, edge_mask3) | cv2.bitwise_and(smoothed, non_edge)
        return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

    # ── Internal pipeline helpers ──────────────────────────────────────────────

    def _bilateral_smooth(self, arr: np.ndarray) -> np.ndarray:
        for _ in range(self.config.bilateral_passes):
            arr = cv2.bilateralFilter(
                arr,
                d=self.config.bilateral_d,
                sigmaColor=self.config.bilateral_sigma_color,
                sigmaSpace=self.config.bilateral_sigma_space,
            )
        return arr

    @staticmethod
    def _clahe_contrast(arr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.merge([clahe.apply(l_ch), a_ch, b_ch])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _posterise(arr: np.ndarray, levels: int = 8) -> np.ndarray:
        step = 255.0 / max(levels - 1, 1)
        quantised = np.round(arr.astype(np.float32) / step) * step
        return np.clip(quantised, 0, 255).astype(np.uint8)

    def _morphological_clean(self, arr: np.ndarray) -> np.ndarray:
        k = self.config.morph_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        arr = cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel)
        return cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)

    @staticmethod
    def _to_bgr(image: Image.Image) -> np.ndarray:
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
