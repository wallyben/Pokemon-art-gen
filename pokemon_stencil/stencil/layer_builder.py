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
        """
        Apply safety processing to a single layer mask.

        Pipeline (order is significant):
        1. Expand thin regions to ``config.min_cut_width_px``.
        2. Insert bridges so floating islands connect to the nearest large
           anchor region (or the image border as a fallback).
        3. Optionally stamp registration marks.
        """
        mask = mask.copy()
        mask = self._expand_thin_regions(mask)
        mask = self._add_bridges(mask)
        if self.config.add_registration_marks:
            mask = self._stamp_registration_marks(mask, image_size)
        return mask

    def _expand_thin_regions(self, mask: np.ndarray) -> np.ndarray:
        """
        Dilate features that are thinner than ``config.min_cut_width_px``.

        A Cricut (or similar cutter) cannot reliably cut strips narrower than
        a physical minimum.  Any foreground region that disappears after
        erosion by half the minimum width is considered "too thin" and the
        whole mask is dilated slightly to thicken it.

        Args:
            mask: Binary uint8 mask (0 or 255), shape (H, W).

        Returns:
            Mask with thin features expanded.  Unaffected when no thin
            features are detected, or when ``min_cut_width_px <= 1``.
        """
        min_w = self.config.min_cut_width_px
        if min_w <= 1 or not np.any(mask > 0):
            return mask

        # Use a small erosion to detect thin features.
        k = max(1, (min_w - 1) // 2)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        eroded = cv2.erode(mask, kernel, iterations=1)

        if np.any(eroded > 0):
            # There are thick features present; only dilate the thin parts.
            thin = cv2.bitwise_and(mask, cv2.bitwise_not(eroded))
            if np.any(thin > 0):
                mask = cv2.dilate(mask, kernel, iterations=1)
                logger.debug("Thin-region expansion applied (min_cut_width_px=%d).", min_w)
        else:
            # The entire mask is thin – dilate everything.
            mask = cv2.dilate(mask, kernel, iterations=1)
            logger.debug("Full mask expanded (thinner than min_cut_width_px=%d).", min_w)

        return mask

    def _add_bridges(self, mask: np.ndarray) -> np.ndarray:
        """
        Connect isolated foreground islands to the nearest anchor region.

        An **anchor** is any connected component that already touches at least
        one image border (and therefore does not need a bridge itself).

        For each isolated island (non-anchor component) the algorithm:
        1. Finds the anchor whose centroid is closest to the island centroid.
        2. Draws a thin filled rectangle of width ``config.bridge_width_px``
           between the two centroids.
        3. Falls back to bridging to the nearest image border when no anchor
           components exist (e.g. when every component is isolated).

        Bridges are drawn directly into *mask* (values set to 255).

        Args:
            mask: Binary uint8 mask (0 or 255), shape (H, W).

        Returns:
            Modified mask with bridge pixels set to 255.
        """
        h, w = mask.shape
        bw = max(1, self.config.bridge_width_px)

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )

        # Classify each foreground component as anchor (touches border) or island.
        anchor_ids: List[int] = []
        island_ids: List[int] = []
        for label_id in range(1, num_labels):  # label 0 = background
            comp_mask = (labels == label_id).astype(np.uint8)
            touches = (
                comp_mask[0, :].any()
                or comp_mask[-1, :].any()
                or comp_mask[:, 0].any()
                or comp_mask[:, -1].any()
            )
            if touches:
                anchor_ids.append(label_id)
            else:
                island_ids.append(label_id)

        for label_id in island_ids:
            cx = int(centroids[label_id][0])
            cy = int(centroids[label_id][1])

            if anchor_ids:
                # Bridge to the nearest anchor centroid.
                nearest = min(
                    anchor_ids,
                    key=lambda a: (
                        (centroids[a][0] - cx) ** 2
                        + (centroids[a][1] - cy) ** 2
                    ),
                )
                tx = int(centroids[nearest][0])
                ty = int(centroids[nearest][1])
                self._draw_bridge(mask, cx, cy, tx, ty, bw, h, w)
                logger.debug(
                    "Bridge: island %d (centroid=%d,%d) → anchor %d (%d,%d)",
                    label_id, cx, cy, nearest, tx, ty,
                )
            else:
                # Fallback: bridge to nearest image border.
                dist_top = cy
                dist_bottom = h - 1 - cy
                dist_left = cx
                dist_right = w - 1 - cx
                min_dist = min(dist_top, dist_bottom, dist_left, dist_right)
                half = max(1, bw // 2)

                if min_dist == dist_top:
                    x1, x2 = max(0, cx - half), min(w, cx + half)
                    mask[0:cy, x1:x2] = 255
                elif min_dist == dist_bottom:
                    x1, x2 = max(0, cx - half), min(w, cx + half)
                    mask[cy:h, x1:x2] = 255
                elif min_dist == dist_left:
                    y1, y2 = max(0, cy - half), min(h, cy + half)
                    mask[y1:y2, 0:cx] = 255
                else:
                    y1, y2 = max(0, cy - half), min(h, cy + half)
                    mask[y1:y2, cx:w] = 255

                logger.debug(
                    "Bridge (border fallback) for island %d (centroid=%d,%d)",
                    label_id, cx, cy,
                )

        return mask

    @staticmethod
    def _draw_bridge(
        mask: np.ndarray,
        x0: int,
        y0: int,
        x1: int,
        y1: int,
        width: int,
        h: int,
        w: int,
    ) -> None:
        """
        Draw a filled rectangular bridge of *width* pixels between two points.

        The bridge is drawn along the axis-aligned dimension with the shorter
        span first (horizontal if |dx| >= |dy|, else vertical) so that the
        path stays tight and does not create a large filled block.

        Args:
            mask:  Binary mask to modify in-place.
            x0, y0: Start centroid (column, row).
            x1, y1: End centroid (column, row).
            width:  Bridge thickness in pixels.
            h, w:   Image dimensions for bounds clamping.
        """
        half = max(1, width // 2)
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)

        if dx >= dy:
            # Horizontal bridge
            col_lo = max(0, min(x0, x1))
            col_hi = min(w, max(x0, x1) + 1)
            row_lo = max(0, y0 - half)
            row_hi = min(h, y0 + half)
            mask[row_lo:row_hi, col_lo:col_hi] = 255
        else:
            # Vertical bridge
            row_lo = max(0, min(y0, y1))
            row_hi = min(h, max(y0, y1) + 1)
            col_lo = max(0, x0 - half)
            col_hi = min(w, x0 + half)
            mask[row_lo:row_hi, col_lo:col_hi] = 255

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
