"""
Bitmap-to-vector tracer using potrace.

Each binary mask from the colour segmentation stage is passed to potrace,
which fits cubic Bezier curves to the pixel boundaries and returns SVG path
data strings.

potrace operates on 1-bit bitmaps.  The strategy is:
  mask (uint8, 0/255) → threshold → potrace bitmap → SVG paths

The module wraps *pypotrace* (Python bindings for libpotrace).  If pypotrace
is not installed, a pure-OpenCV fallback approximates the contours as
polylines (less smooth but functional).
"""

import logging
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from pokemon_stencil.config import VectorConfig

logger = logging.getLogger(__name__)


class VectorTracer:
    """
    Converts binary masks into SVG path data strings.

    Prefers pypotrace for Bezier-smooth paths; falls back to OpenCV
    findContours → polyline approximation if pypotrace is unavailable.
    """

    def __init__(self, config: VectorConfig) -> None:
        """
        Args:
            config: VectorConfig with potrace and SVG canvas parameters.
        """
        self.config = config
        self._has_potrace = self._check_potrace()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def trace_mask(self, mask: np.ndarray) -> List[str]:
        """
        Trace a binary mask and return a list of SVG path ``d`` attribute strings.

        Each element of the returned list represents one closed contour (outer
        boundary or hole).

        Args:
            mask: Binary uint8 ndarray (0 or 255), shape (H, W).

        Returns:
            List of SVG path data strings suitable for ``<path d="…"/>`` elements.
        """
        if self._has_potrace:
            return self._trace_with_potrace(mask)
        else:
            logger.warning(
                "pypotrace not available – using OpenCV contour fallback."
            )
            return self._trace_with_opencv(mask)

    def px_to_mm(self, px: float) -> float:
        """Convert pixels to millimetres using the configured DPI."""
        return px * 25.4 / self.config.dpi

    def mm_to_px(self, mm: float) -> float:
        """Convert millimetres to pixels using the configured DPI."""
        return mm * self.config.dpi / 25.4

    # ------------------------------------------------------------------
    # potrace backend
    # ------------------------------------------------------------------

    def _trace_with_potrace(self, mask: np.ndarray) -> List[str]:
        """
        Use pypotrace to convert *mask* to cubic Bezier SVG paths.

        pypotrace expects a 2D array of booleans (True = foreground).
        """
        try:
            import potrace  # type: ignore[import]
        except ImportError:
            self._has_potrace = False
            return self._trace_with_opencv(mask)

        # potrace wants a packed bitmap; use a uint32 buffer
        binary = (mask > 127).astype(np.uint32)
        bm = potrace.Bitmap(binary)
        path = bm.trace(
            turdsize=self.config.turdsize,
            alphamax=self.config.alphamax,
            opttolerance=self.config.opttolerance,
        )

        paths: List[str] = []
        for curve in path:
            d = self._curve_to_svg_d(curve)
            if d:
                paths.append(d)
        return paths

    @staticmethod
    def _curve_to_svg_d(curve) -> str:
        """Convert a potrace Curve object to an SVG path d-string."""
        if not curve.segments:
            return ""

        parts: List[str] = []
        start = curve.start_point
        parts.append(f"M {start[0]:.3f},{start[1]:.3f}")

        for segment in curve.segments:
            if segment.is_corner:
                # Corner: two straight lines through the vertex
                c = segment.c
                end = segment.end_point
                parts.append(f"L {c[0]:.3f},{c[1]:.3f}")
                parts.append(f"L {end[0]:.3f},{end[1]:.3f}")
            else:
                # Bezier: cubic curve
                c1, c2, end = segment.c1, segment.c2, segment.end_point
                parts.append(
                    f"C {c1[0]:.3f},{c1[1]:.3f} "
                    f"{c2[0]:.3f},{c2[1]:.3f} "
                    f"{end[0]:.3f},{end[1]:.3f}"
                )
        parts.append("Z")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # OpenCV fallback backend
    # ------------------------------------------------------------------

    def _trace_with_opencv(self, mask: np.ndarray) -> List[str]:
        """
        Approximate vector tracing using OpenCV contour detection.

        Returns simplified polyline paths (no Bezier curves).  Less smooth
        than potrace but suitable when pypotrace is not installed.
        """
        binary = (mask > 127).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS
        )

        paths: List[str] = []
        min_len = self.config.min_path_length

        for contour in contours:
            # Simplify the contour
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) < 3:
                continue

            perimeter = cv2.arcLength(approx, True)
            if perimeter < min_len:
                continue

            points = approx.reshape(-1, 2)
            d_parts = [f"M {points[0][0]:.3f},{points[0][1]:.3f}"]
            for pt in points[1:]:
                d_parts.append(f"L {pt[0]:.3f},{pt[1]:.3f}")
            d_parts.append("Z")
            paths.append(" ".join(d_parts))

        return paths

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _check_potrace() -> bool:
        """Return True if pypotrace can be imported."""
        try:
            import potrace  # noqa: F401
            return True
        except ImportError:
            return False
