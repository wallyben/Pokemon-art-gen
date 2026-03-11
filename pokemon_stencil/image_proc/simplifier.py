"""
Image simplification pipeline.

Converts a generated or reference Pokemon image into a clean, simplified
bitmap with broad colour regions and sharp edges – the ideal input for
colour segmentation and vector tracing.

Pipeline stages
---------------
1. Bilateral smooth   – preserve edges, smooth within regions
2. CLAHE contrast     – bring out local structure
3. Posterisation      – reduce tonal gradients to flat bands
4. Morphological open – remove speckling
5. Median blur        – final smoothing pass
"""

import logging
from typing import Tuple

import cv2
import numpy as np
from PIL import Image

from pokemon_stencil.config import ProcessingConfig

logger = logging.getLogger(__name__)


class ImageSimplifier:
    """
    Reduces image complexity to stencil-ready flat colour regions.

    All operations are CPU-only OpenCV / NumPy.
    """

    def __init__(self, config: ProcessingConfig) -> None:
        """
        Args:
            config: ProcessingConfig with filter and morphology parameters.
        """
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def simplify(self, image: Image.Image) -> Image.Image:
        """
        Run the full simplification pipeline on *image*.

        Args:
            image: Input PIL Image (RGB).

        Returns:
            Simplified PIL Image ready for colour segmentation.
        """
        arr = self._to_cv2(image)
        arr = self._bilateral_smooth(arr)
        arr = self._clahe_contrast(arr)
        arr = self._posterise(arr, levels=self.config.n_colors * 2)
        arr = self._morphological_clean(arr)
        arr = self._median_blur(arr)
        result = Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))
        logger.debug("Simplification complete; output size: %s", result.size)
        return result

    def extract_edges(self, image: Image.Image) -> np.ndarray:
        """
        Extract a binary edge map from *image* using Canny.

        Args:
            image: Input PIL Image (RGB).

        Returns:
            Binary uint8 ndarray (0 or 255) with the edge map.
        """
        arr = self._to_cv2(image)
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(
            blurred,
            threshold1=self.config.canny_low,
            threshold2=self.config.canny_high,
        )
        return edges

    # ------------------------------------------------------------------
    # Pipeline steps
    # ------------------------------------------------------------------

    def _bilateral_smooth(self, arr: np.ndarray) -> np.ndarray:
        """Repeated bilateral filtering for edge-preserving smoothing."""
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
        """Boost local contrast via CLAHE on the L channel (LAB colour space)."""
        lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_ch = clahe.apply(l_ch)
        lab = cv2.merge([l_ch, a_ch, b_ch])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _posterise(arr: np.ndarray, levels: int = 8) -> np.ndarray:
        """Quantise pixel values to *levels* discrete steps per channel."""
        arr_f = arr.astype(np.float32)
        step = 255.0 / max(levels - 1, 1)
        arr_f = np.round(arr_f / step) * step
        return np.clip(arr_f, 0, 255).astype(np.uint8)

    def _morphological_clean(self, arr: np.ndarray) -> np.ndarray:
        """Remove small noise blobs with morphological open then close."""
        k = self.config.morph_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        arr = cv2.morphologyEx(arr, cv2.MORPH_OPEN, kernel)
        arr = cv2.morphologyEx(arr, cv2.MORPH_CLOSE, kernel)
        return arr

    @staticmethod
    def _median_blur(arr: np.ndarray, ksize: int = 5) -> np.ndarray:
        """Final median blur pass to flatten residual speckle."""
        return cv2.medianBlur(arr, ksize)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _to_cv2(image: Image.Image) -> np.ndarray:
        """Convert PIL Image (RGB) to OpenCV BGR ndarray."""
        rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
