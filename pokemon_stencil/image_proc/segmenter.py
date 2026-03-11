"""
Colour region segmentation.

Uses K-Means clustering in LAB colour space to partition a simplified image
into N distinct colour layers.  Each layer is returned as a binary mask
(uint8 ndarray, 0 = not in layer, 255 = in layer) alongside its representative
RGB colour.

LAB space is used because Euclidean distance in LAB correlates closely with
human-perceived colour difference (ΔE), giving perceptually even clusters.
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
from PIL import Image
from sklearn.cluster import KMeans

from pokemon_stencil.config import ProcessingConfig

logger = logging.getLogger(__name__)


@dataclass
class ColourLayer:
    """A single colour region extracted from the image."""

    index: int
    """Zero-based layer index (0 = most dominant colour)."""

    colour_rgb: Tuple[int, int, int]
    """Representative RGB colour of this layer."""

    mask: np.ndarray
    """Binary mask (uint8, 0 or 255) where this colour is present."""

    pixel_count: int
    """Number of pixels assigned to this layer."""

    @property
    def coverage_fraction(self) -> float:
        """Fraction of total image pixels covered by this layer."""
        return self.pixel_count / max(self.mask.size, 1)


class ColourSegmenter:
    """
    Segments an image into K colour layers using K-Means in LAB space.

    Layers are sorted by descending pixel count (most dominant first).
    Layers below *min_region_area* pixels are discarded.
    """

    def __init__(self, config: ProcessingConfig) -> None:
        """
        Args:
            config: ProcessingConfig with n_colors and min_region_area.
        """
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def segment(self, image: Image.Image) -> List[ColourLayer]:
        """
        Segment *image* into colour layers.

        Args:
            image: Simplified PIL Image in RGB mode.

        Returns:
            List of ColourLayer objects, sorted by descending pixel count.
            Layers smaller than config.min_region_area are excluded.
        """
        arr_rgb = np.array(image.convert("RGB"), dtype=np.uint8)
        arr_lab = self._rgb_to_lab(arr_rgb)

        h, w = arr_rgb.shape[:2]
        pixels_lab = arr_lab.reshape(-1, 3).astype(np.float32)
        pixels_rgb = arr_rgb.reshape(-1, 3)

        labels, centres_lab = self._cluster(pixels_lab)

        layers: List[ColourLayer] = []
        for k in range(self.config.n_colors):
            pixel_mask = labels == k
            count = int(pixel_mask.sum())

            if count < self.config.min_region_area:
                logger.debug(
                    "Layer %d skipped (only %d px < min %d)",
                    k, count, self.config.min_region_area,
                )
                continue

            # Representative colour: median of cluster pixels in RGB
            cluster_rgb = pixels_rgb[pixel_mask]
            colour_rgb = tuple(np.median(cluster_rgb, axis=0).astype(int).tolist())

            # Binary mask reshaped to image dimensions
            binary = pixel_mask.reshape(h, w).astype(np.uint8) * 255
            binary = self._clean_mask(binary)

            layers.append(
                ColourLayer(
                    index=k,
                    colour_rgb=colour_rgb,
                    mask=binary,
                    pixel_count=count,
                )
            )
            logger.debug(
                "Layer %d | colour=%s | pixels=%d (%.1f%%)",
                k, colour_rgb, count, 100 * count / (h * w),
            )

        # Sort by coverage, most dominant first
        layers.sort(key=lambda l: l.pixel_count, reverse=True)

        # Re-index after filtering and sorting
        for new_idx, layer in enumerate(layers):
            layer.index = new_idx

        logger.info("Segmentation produced %d layer(s).", len(layers))
        return layers

    def visualise(
        self, layers: List[ColourLayer], image_size: Tuple[int, int]
    ) -> Image.Image:
        """
        Render all colour layers onto a white canvas for inspection.

        Args:
            layers: Output of :meth:`segment`.
            image_size: (width, height) of the output image.

        Returns:
            PIL Image showing each layer in its representative colour.
        """
        w, h = image_size
        canvas = np.full((h, w, 3), 255, dtype=np.uint8)
        for layer in layers:
            colour = np.array(layer.colour_rgb, dtype=np.uint8)
            canvas[layer.mask == 255] = colour
        return Image.fromarray(canvas)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _cluster(
        self, pixels_lab: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run K-Means and return per-pixel labels and cluster centres."""
        kmeans = KMeans(
            n_clusters=self.config.n_colors,
            init="k-means++",
            n_init=5,
            max_iter=200,
            random_state=42,
        )
        kmeans.fit(pixels_lab)
        labels = kmeans.labels_.astype(np.int32)
        centres = kmeans.cluster_centers_
        return labels, centres

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological operations to clean up the binary mask.

        Removes thin slivers and fills small holes so each layer forms
        solid, cuttable regions.
        """
        k = self.config.morph_kernel_size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    @staticmethod
    def _rgb_to_lab(arr_rgb: np.ndarray) -> np.ndarray:
        """Convert an HxWx3 uint8 RGB array to LAB float32."""
        bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        return lab.astype(np.float32)
