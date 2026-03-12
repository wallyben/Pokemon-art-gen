"""
Stencil suitability scorer.

Ranks generated artwork candidates by how well they suit physical stencil
cutting.  Three complementary metrics are combined into a single normalised
score in [0, 1].

Metrics
-------
1. **Silhouette clarity** – Canny edge density.
   Mid-range density (≈10 % of pixels) is ideal: too few edges produces a
   featureless blob; too many edges indicate noisy detail that will be lost
   when the stencil is physically cut.

2. **Connected-component score** – penalises excessive small isolated
   regions.  Many tiny islands are physically un-cuttable and will fall out
   of the stencil material.

3. **Contrast score** – grayscale standard deviation.
   Higher contrast produces cleaner K-Means colour segmentation and more
   distinct stencil layers.
"""

from __future__ import annotations

import logging
from typing import List

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class StencilScorer:
    """
    Scores a PIL Image for stencil-cutting suitability.

    Args:
        weight_clarity: Weight applied to the silhouette clarity metric.
        weight_component: Weight applied to the connected-component metric.
        weight_contrast: Weight applied to the contrast metric.
        target_edge_density: Ideal Canny edge-pixel fraction.  Score is
            maximised when edge density equals this value.
        max_small_components: Number of small components at which the
            component score reaches 0.
        min_component_area_frac: Fraction of total image pixels below which
            a connected region is considered "small".
        contrast_normaliser: Grayscale standard deviation that maps to a
            contrast score of 1.0.
    """

    def __init__(
        self,
        weight_clarity: float = 0.4,
        weight_component: float = 0.3,
        weight_contrast: float = 0.3,
        target_edge_density: float = 0.10,
        max_small_components: int = 20,
        min_component_area_frac: float = 0.002,
        contrast_normaliser: float = 64.0,
    ) -> None:
        self.weight_clarity = weight_clarity
        self.weight_component = weight_component
        self.weight_contrast = weight_contrast
        self.target_edge_density = target_edge_density
        self.max_small_components = max_small_components
        self.min_component_area_frac = min_component_area_frac
        self.contrast_normaliser = contrast_normaliser

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def score(self, image: Image.Image) -> float:
        """
        Compute a stencil-suitability score for *image*.

        Args:
            image: PIL Image (any mode; converted to RGB internally).

        Returns:
            Score in [0.0, 1.0].  Higher is better.
        """
        arr = np.array(image.convert("RGB"), dtype=np.uint8)
        s_clarity = self._silhouette_clarity(arr)
        s_component = self._component_score(arr)
        s_contrast = self._contrast_score(arr)
        total = (
            self.weight_clarity * s_clarity
            + self.weight_component * s_component
            + self.weight_contrast * s_contrast
        )
        logger.debug(
            "Score: clarity=%.3f component=%.3f contrast=%.3f → %.3f",
            s_clarity, s_component, s_contrast, total,
        )
        return float(total)

    def score_all(self, images: List[Image.Image]) -> List[float]:
        """
        Score every image in *images* and return a parallel list of scores.

        Args:
            images: List of PIL Images to score.

        Returns:
            Scores in the same order as *images*.
        """
        return [self.score(img) for img in images]

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    def _silhouette_clarity(self, arr: np.ndarray) -> float:
        """
        Score based on Canny edge density.

        The score is 1.0 when edge density equals ``target_edge_density``
        and decays linearly to 0 as density diverges from the target.  This
        penalises both featureless blobs (too few edges) and overly noisy
        designs (too many edges).

        Returns:
            Float in [0.0, 1.0].
        """
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        density = float(np.count_nonzero(edges)) / float(edges.size)
        target = self.target_edge_density
        score = max(0.0, 1.0 - abs(density - target) / target)
        return score

    def _component_score(self, arr: np.ndarray) -> float:
        """
        Score based on the number of small connected regions.

        Many small isolated islands are physically un-cuttable and indicate
        a design that is too complex for stencil use.

        Returns:
            1.0 when no small components are present; 0.0 when
            ``max_small_components`` or more small components exist.
        """
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        total_px = int(arr.shape[0]) * int(arr.shape[1])
        min_area = max(1, int(total_px * self.min_component_area_frac))
        small_count = sum(
            1
            for i in range(1, n_labels)
            if int(stats[i, cv2.CC_STAT_AREA]) < min_area
        )
        score = max(0.0, 1.0 - small_count / self.max_small_components)
        return score

    def _contrast_score(self, arr: np.ndarray) -> float:
        """
        Score based on grayscale standard deviation.

        Higher contrast produces cleaner K-Means segmentation and bolder
        stencil layers.

        Returns:
            Score in [0.0, 1.0]; reaches 1.0 at ``contrast_normaliser``
            standard deviation.
        """
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        std = float(np.std(gray.astype(np.float32)))
        return min(1.0, std / self.contrast_normaliser)
