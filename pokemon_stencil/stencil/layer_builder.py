"""
Stencil-safety layer builder.

Processes colour layers from segmentation to make them physically safe for
cutting on a Cricut or similar die-cutting machine.

Operations applied to each mask
--------------------------------
1. **Bridge insertion** – isolated foreground islands (regions that do not
   touch any image border) are connected to the nearest border with a thin
   rectangular bridge of width ``config.bridge_width``.  Without bridges,
   floating islands would fall out of the stencil when it is cut.

2. **Registration mark stamping** – small filled circles are drawn at the
   four corners of every layer mask so that layers can be aligned precisely
   when applying the multi-layer design.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import cv2
import numpy as np

from pokemon_stencil.config import StencilConfig
from pokemon_stencil.image_proc.segmenter import ColourLayer

logger = logging.getLogger(__name__)


class StencilLayerBuilder:
    """
    Applies stencil-safety processing to a list of colour layers.

    Args:
        config: ``StencilConfig`` with bridge and registration parameters.
    """

    def __init__(self, config: StencilConfig) -> None:
        self.config = config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        layers: List[ColourLayer],
        image_size: Tuple[int, int],
    ) -> List[ColourLayer]:
        """
        Process all colour layers for stencil safety.

        Args:
            layers: Colour layers produced by ``ColourSegmenter``.
            image_size: ``(width, height)`` of the masks in pixels.

        Returns:
            New list of ``ColourLayer`` objects with processed masks.
            Layer ordering and indices are preserved.
        """
        processed: List[ColourLayer] = []
        for layer in layers:
            new_mask = self._process_mask(layer.mask, image_size)
            processed.append(
                ColourLayer(
                    index=layer.index,
                    colour_rgb=layer.colour_rgb,
                    mask=new_mask,
                    pixel_count=int(np.sum(new_mask == 255)),
                )
            )
            logger.debug("Layer %d: stencil safety applied.", layer.index)
        return processed

    # ------------------------------------------------------------------
    # Internal per-mask processing
    # ------------------------------------------------------------------

    def _process_mask(
        self,
        mask: np.ndarray,
        image_size: Tuple[int, int],
    ) -> np.ndarray:
        """Apply bridge insertion and (optionally) registration marks."""
        mask = mask.copy()
        mask = self._add_bridges(mask)
        if self.config.add_registration_marks:
            mask = self._stamp_registration_marks(mask, image_size)
        return mask

    def _add_bridges(self, mask: np.ndarray) -> np.ndarray:
        """
        Connect isolated foreground islands to the image border.

        Uses connected-component analysis to identify regions that do not
        touch any image edge.  For each such island the centroid is located
        and a thin rectangle of width ``config.bridge_width`` is drawn from
        the centroid to the nearest border.

        Args:
            mask: Binary uint8 mask (0 or 255), shape (H, W).

        Returns:
            Modified mask with bridge pixels set to 255.
        """
        h, w = mask.shape
        bw = self.config.bridge_width

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        for label_id in range(1, num_labels):  # label 0 = background
            comp_mask = (labels == label_id).astype(np.uint8)

            # Check whether this component touches any image border
            touches_border = (
                comp_mask[0, :].any()
                or comp_mask[-1, :].any()
                or comp_mask[:, 0].any()
                or comp_mask[:, -1].any()
            )
            if touches_border:
                continue

            # Draw a bridge from the centroid to the nearest border
            cx = int(centroids[label_id][0])
            cy = int(centroids[label_id][1])

            dist_top = cy
            dist_bottom = h - 1 - cy
            dist_left = cx
            dist_right = w - 1 - cx
            min_dist = min(dist_top, dist_bottom, dist_left, dist_right)

            half = max(1, bw // 2)

            if min_dist == dist_top:
                x1 = max(0, cx - half)
                x2 = min(w, cx + half)
                mask[0:cy, x1:x2] = 255
            elif min_dist == dist_bottom:
                x1 = max(0, cx - half)
                x2 = min(w, cx + half)
                mask[cy:h, x1:x2] = 255
            elif min_dist == dist_left:
                y1 = max(0, cy - half)
                y2 = min(h, cy + half)
                mask[y1:y2, 0:cx] = 255
            else:
                y1 = max(0, cy - half)
                y2 = min(h, cy + half)
                mask[y1:y2, cx:w] = 255

            logger.debug(
                "Bridge added for component %d (centroid=%d,%d)",
                label_id, cx, cy,
            )

        return mask

    def _stamp_registration_marks(
        self,
        mask: np.ndarray,
        image_size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Draw filled registration circles in the four corners of *mask*.

        Circles are always set to foreground (255) so they appear as
        identical cut-out holes on every layer, enabling precise alignment.

        Args:
            mask: Binary mask to stamp (modified in-place copy).
            image_size: ``(width, height)`` in pixels.

        Returns:
            Mask with registration marks stamped.
        """
        w, h = image_size
        # Convert mm radius → pixels using a fixed nominal DPI
        _dpi = 96.0
        radius_px = max(4, int(self.config.reg_mark_radius_mm * _dpi / 25.4))
        margin = radius_px + 2

        corners = [
            (margin, margin),
            (w - margin, margin),
            (margin, h - margin),
            (w - margin, h - margin),
        ]
        for cx, cy in corners:
            cv2.circle(mask, (cx, cy), radius_px, 255, thickness=-1)

        return mask
