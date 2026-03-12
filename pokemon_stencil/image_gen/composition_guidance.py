"""
Composition guidance for line-based structural conditioning.

Provides two modes for generating a structural guide (composition map) from
a reference image or sketch, which can be fed to the image generator to
improve silhouette clarity and stencil-friendliness.

Modes
-----
EDGE
    Uses Canny edge detection on the input image to extract structural
    boundaries.  Steps:

    1. Convert image to grayscale.
    2. Apply Gaussian blur to suppress noise.
    3. Run ``cv2.Canny`` edge detection.
    4. Optionally dilate the resulting edges for a stronger structural guide.

SKETCH
    Generates a synthetic sketch using Sobel + Laplacian edge detection,
    capturing both directional gradients and fine structural detail.

Public API
----------
:func:`generate_composition_map` is the single entry point.  It accepts a
``PIL.Image`` and returns a ``numpy.ndarray`` suitable for use as a
structural conditioning map inside the generation pipeline.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Optional

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Public types
# ─────────────────────────────────────────────────────────────────────────────

class CompositionMode(str, Enum):
    """Mode used to derive the structural composition map."""

    EDGE = "edge"
    """Canny-based hard edge detection (recommended for clean reference art)."""

    SKETCH = "sketch"
    """Sobel + Laplacian gradient sketch (better for detailed source images)."""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_composition_map(
    image: Image.Image,
    mode: CompositionMode = CompositionMode.EDGE,
    canny_low: int = 50,
    canny_high: int = 150,
    blur_kernel_size: int = 5,
    dilate: bool = True,
    dilate_kernel_size: int = 3,
) -> np.ndarray:
    """
    Generate a structural composition map from a PIL Image.

    The returned map is a 2-D grayscale array where edges / structural lines
    are represented as bright pixels (255) on a black (0) background.

    Args:
        image: Source image in any PIL mode; converted to grayscale internally.
        mode: Algorithm to use for edge extraction.
            - :attr:`CompositionMode.EDGE` – Gaussian blur + Canny.
            - :attr:`CompositionMode.SKETCH` – Sobel + Laplacian gradient.
        canny_low: Lower hysteresis threshold for Canny (EDGE mode only).
        canny_high: Upper hysteresis threshold for Canny (EDGE mode only).
        blur_kernel_size: Gaussian blur kernel size applied before detection.
            Must be a *positive odd* integer (e.g. 3, 5, 7).
        dilate: When ``True``, dilate the Canny edge map to produce thicker
            guide lines.  Only used in EDGE mode.
        dilate_kernel_size: Dilation kernel size (pixels).  Ignored when
            ``dilate=False``.

    Returns:
        ``numpy.ndarray`` of shape ``(H, W)`` and dtype ``uint8``.  Edges are
        255; background is 0.

    Raises:
        ValueError: If ``blur_kernel_size`` is not a positive odd integer.
        ValueError: If an unknown ``mode`` value is supplied.
    """
    if blur_kernel_size < 1 or blur_kernel_size % 2 == 0:
        raise ValueError(
            f"blur_kernel_size must be a positive odd integer, "
            f"got {blur_kernel_size!r}."
        )

    # Convert to grayscale uint8 array.
    gray = np.array(image.convert("L"), dtype=np.uint8)

    if mode == CompositionMode.EDGE:
        composition_map = _edge_mode(
            gray,
            canny_low=canny_low,
            canny_high=canny_high,
            blur_kernel_size=blur_kernel_size,
            dilate=dilate,
            dilate_kernel_size=dilate_kernel_size,
        )
    elif mode == CompositionMode.SKETCH:
        composition_map = _sketch_mode(gray, blur_kernel_size=blur_kernel_size)
    else:
        raise ValueError(f"Unknown CompositionMode: {mode!r}")

    logger.debug(
        "Generated composition map | mode=%s | shape=%s | nonzero_pct=%.1f%%",
        mode,
        composition_map.shape,
        100.0 * np.count_nonzero(composition_map) / max(composition_map.size, 1),
    )
    return composition_map


# ─────────────────────────────────────────────────────────────────────────────
# Private helpers
# ─────────────────────────────────────────────────────────────────────────────

def _edge_mode(
    gray: np.ndarray,
    canny_low: int,
    canny_high: int,
    blur_kernel_size: int,
    dilate: bool,
    dilate_kernel_size: int,
) -> np.ndarray:
    """
    Apply Gaussian blur then Canny edge detection, with optional dilation.

    Args:
        gray: Grayscale image array of shape ``(H, W)`` and dtype ``uint8``.
        canny_low: Lower Canny threshold.
        canny_high: Upper Canny threshold.
        blur_kernel_size: Gaussian blur kernel size.
        dilate: Whether to dilate the edge map.
        dilate_kernel_size: Size of the dilation kernel.

    Returns:
        Binary edge map; dtype ``uint8`` (255 = edge, 0 = background).
    """
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)
    edges = cv2.Canny(blurred, canny_low, canny_high)

    if dilate and dilate_kernel_size > 0:
        kernel = np.ones((dilate_kernel_size, dilate_kernel_size), dtype=np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

    return edges


def _sketch_mode(
    gray: np.ndarray,
    blur_kernel_size: int,
) -> np.ndarray:
    """
    Generate a synthetic sketch using Sobel and Laplacian edge detection.

    Combines horizontal/vertical Sobel gradient magnitude with the absolute
    Laplacian, weighted 70 % / 30 %, then normalises to [0, 255].

    Args:
        gray: Grayscale image array of shape ``(H, W)`` and dtype ``uint8``.
        blur_kernel_size: Gaussian blur kernel size applied before detection.

    Returns:
        Normalised edge magnitude map; dtype ``uint8``
        (255 = strong edge, 0 = background).
    """
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)

    # Sobel gradient magnitude.
    sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

    # Laplacian for fine structural detail.
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    laplacian_abs = np.abs(laplacian)

    # Weighted combination.
    combined = 0.7 * sobel_mag + 0.3 * laplacian_abs

    # Normalise to [0, 255].
    max_val = combined.max()
    if max_val > 0:
        combined = combined / max_val * 255.0

    return combined.astype(np.uint8)
