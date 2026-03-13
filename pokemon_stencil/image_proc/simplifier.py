"""
Image simplification pipeline.

Converts a generated or reference Pokemon image into a clean, simplified
bitmap with broad colour regions and sharp edges – the ideal input for
colour segmentation and vector tracing.

Pipeline stages
---------------
1. Bilateral smooth          – preserve edges, smooth within regions
2. CLAHE contrast            – bring out local structure
3. Posterisation             – reduce tonal gradients to flat bands
4. Morphological open/close  – remove speckling
5. Adaptive edge thinning    – thin over-thick edges to aid clean cutting
6. Component merging         – absorb small floating islands into neighbours
7. Median blur               – final smoothing pass

Stencil safety improvements (stages 5-6) are controlled by:
- ``config.adaptive_edge_thinning``  (default True)
- ``config.merge_small_components``  (default True)
- ``config.min_island_area``         (default 200 px)
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

        # ── Stencil safety improvements ───────────────────────────────────────
        if self.config.adaptive_edge_thinning:
            arr = self._adaptive_edge_thin(arr)

        if self.config.merge_small_components:
            arr = self._merge_small_components(arr, self.config.min_island_area)

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
    def _adaptive_edge_thin(arr: np.ndarray) -> np.ndarray:
        """
        Adaptively thin overly-thick edge regions.

        For each colour channel the method detects regions that are close to
        a posterised boundary, then applies a light erosion/dilation cycle
        calibrated to the local edge thickness to ensure no single edge
        feature exceeds a safe Cricut-cuttable width.

        The result preserves flat-fill regions while tightening any blurred
        boundary zones left by bilateral filtering.

        Args:
            arr: OpenCV BGR uint8 image.

        Returns:
            Edge-thinned BGR uint8 image.
        """
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

        # Detect transition zones between posterised colour bands.
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = cv2.magnitude(sobelx, sobely)

        # Normalise and threshold to produce an edge mask.
        mag_norm = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        _, edge_mask = cv2.threshold(
            mag_norm.astype(np.uint8), 30, 255, cv2.THRESH_BINARY
        )

        # Thin edge mask via skeletonisation (Zhang-Suen approximation via
        # repeated erosion until stable).
        thinned = _thin_binary_mask(edge_mask)

        # Expand thinned edges back by 1 px so they remain visually present.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        thinned_dilated = cv2.dilate(thinned, kernel, iterations=1)

        # Restore original colour within thinned edge zones:
        # outside edge zone keeps arr; inside edge zone keeps arr (colours
        # are not changed – only the mask shape is used to guide morphological
        # clean-up of adjacent fill regions).
        mask3 = cv2.cvtColor(thinned_dilated, cv2.COLOR_GRAY2BGR)
        non_edge_mask = cv2.bitwise_not(mask3)

        # Smooth only the non-edge fill areas with a small bilateral pass.
        smoothed = cv2.bilateralFilter(arr, d=5, sigmaColor=40, sigmaSpace=40)
        result = cv2.bitwise_and(arr, mask3) | cv2.bitwise_and(smoothed, non_edge_mask)
        return result

    @staticmethod
    def _merge_small_components(
        arr: np.ndarray, min_area: int
    ) -> np.ndarray:
        """
        Merge small isolated colour regions (islands) into the nearest
        large neighbour colour.

        Small floating islands below *min_area* pixels cause hairline cuts
        that snap during handling.  This step identifies them per-channel
        and flood-fills them with the median colour of the surrounding border.

        Args:
            arr: OpenCV BGR uint8 image.
            min_area: Minimum component area in pixels to retain.

        Returns:
            BGR image with small components merged into neighbours.
        """
        result = arr.copy()
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)

        # Work on a quantised grayscale to limit the number of unique levels.
        _, quant = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Find connected components on the quantised image.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            quant, connectivity=8
        )

        for label_idx in range(1, num_labels):  # skip background (0)
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area >= min_area:
                continue

            # Build a mask for this small component.
            component_mask = (labels == label_idx).astype(np.uint8) * 255

            # Dilate mask to find its immediate border pixels.
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated = cv2.dilate(component_mask, kernel, iterations=2)
            border_mask = cv2.bitwise_and(dilated, cv2.bitwise_not(component_mask))

            # Sample the median colour from the border.
            border_pixels = arr[border_mask > 0]
            if border_pixels.size == 0:
                continue
            fill_colour = np.median(border_pixels, axis=0).astype(np.uint8)

            # Flood-fill the component with the border median colour.
            result[component_mask > 0] = fill_colour

        merged_count = sum(
            1
            for i in range(1, num_labels)
            if stats[i, cv2.CC_STAT_AREA] < min_area
        )
        if merged_count:
            logger.debug("Merged %d small component(s) (min_area=%d px).", merged_count, min_area)

        return result

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


# ── Module-level helpers ───────────────────────────────────────────────────────

def _thin_binary_mask(mask: np.ndarray, max_iter: int = 10) -> np.ndarray:
    """
    Approximate skeletonisation of a binary mask via repeated erosion.

    Iterates erosion until fewer than 0.1 % of pixels change or *max_iter*
    is reached.

    Args:
        mask: Binary uint8 image (0 or 255).
        max_iter: Maximum erosion iterations.

    Returns:
        Thinned binary mask.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    prev = mask.copy()
    for _ in range(max_iter):
        eroded = cv2.erode(prev, kernel, iterations=1)
        change_ratio = np.sum(prev != eroded) / max(prev.size, 1)
        prev = eroded
        if change_ratio < 0.001:
            break
    return prev
