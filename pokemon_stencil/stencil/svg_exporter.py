"""
SVG exporter for Cricut-ready stencil files.

Writes two types of SVG files per pipeline run:

Per-layer SVG
    One file per colour layer containing only the cut paths for that layer.
    The fill is set to the layer's representative colour (useful for preview)
    and stroke is ``none`` so Cricut reads the filled shapes as cut regions.

Combined SVG
    A single file with all layers stacked in named ``<g>`` groups,
    colour-coded for visual verification.  Not intended for direct cutting.

Canvas dimensions default to the Cricut standard 12×12 inch mat
(304.8 × 304.8 mm) as configured in ``VectorConfig``.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import svgwrite  # type: ignore[import]

from pokemon_stencil.config import StencilConfig, VectorConfig
from pokemon_stencil.vector.path_builder import LayerPaths

logger = logging.getLogger(__name__)


def _rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    """Convert an ``(R, G, B)`` tuple to a CSS hex colour string."""
    return "#{:02x}{:02x}{:02x}".format(*rgb)


class SVGExporter:
    """
    Exports stencil path data to SVG files.

    Args:
        vector_config: Canvas size and DPI parameters.
        stencil_config: Stroke width and padding parameters.
    """

    def __init__(
        self,
        vector_config: VectorConfig,
        stencil_config: StencilConfig,
    ) -> None:
        self.vcfg = vector_config
        self.scfg = stencil_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(
        self,
        layer_paths: List[LayerPaths],
        output_dir: Path,
        prefix: str,
        image_size_px: Tuple[int, int],
    ) -> List[Path]:
        """
        Write per-layer and combined SVG files.

        Args:
            layer_paths: ``LayerPaths`` objects from ``StencilPathBuilder``.
            output_dir: Directory to write SVG files into (created if absent).
            prefix: Filename prefix, e.g. ``"img000"``.
            image_size_px: ``(width, height)`` of the source image in pixels,
                           used to set the SVG ``viewBox``.

        Returns:
            List of ``Path`` objects for every SVG written (per-layer files
            first, combined file last).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        written: List[Path] = []

        w_px, h_px = image_size_px
        w_mm = self.vcfg.canvas_width_mm
        h_mm = self.vcfg.canvas_height_mm

        for lp in layer_paths:
            svg_path = output_dir / f"{prefix}_layer{lp.index:02d}.svg"
            self._write_layer_svg(lp, svg_path, w_px, h_px, w_mm, h_mm)
            written.append(svg_path)
            logger.debug("Wrote layer SVG: %s", svg_path)

        if layer_paths:
            combined_path = output_dir / f"{prefix}_combined.svg"
            self._write_combined_svg(
                layer_paths, combined_path, w_px, h_px, w_mm, h_mm
            )
            written.append(combined_path)
            logger.debug("Wrote combined SVG: %s", combined_path)

        return written

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_drawing(
        self,
        path: Path,
        w_px: int,
        h_px: int,
        w_mm: float,
        h_mm: float,
    ) -> svgwrite.Drawing:
        """Return a ``svgwrite.Drawing`` with the correct canvas and viewBox."""
        dwg = svgwrite.Drawing(
            str(path),
            size=(f"{w_mm}mm", f"{h_mm}mm"),
            profile="full",
        )
        dwg.viewbox(0, 0, w_px, h_px)
        return dwg

    def _write_layer_svg(
        self,
        lp: LayerPaths,
        path: Path,
        w_px: int,
        h_px: int,
        w_mm: float,
        h_mm: float,
    ) -> None:
        """Write a single-layer SVG file for *lp*."""
        dwg = self._make_drawing(path, w_px, h_px, w_mm, h_mm)
        colour = _rgb_to_hex(lp.colour_rgb)
        group = dwg.g(
            id=f"layer_{lp.index:02d}",
            fill=colour,
            stroke="none",
        )
        for d_str in lp.paths:
            group.add(dwg.path(d=d_str))
        dwg.add(group)
        dwg.save()

    def _write_combined_svg(
        self,
        layer_paths: List[LayerPaths],
        path: Path,
        w_px: int,
        h_px: int,
        w_mm: float,
        h_mm: float,
    ) -> None:
        """Write a combined multi-layer SVG for visual inspection."""
        dwg = self._make_drawing(path, w_px, h_px, w_mm, h_mm)
        for lp in layer_paths:
            colour = _rgb_to_hex(lp.colour_rgb)
            group = dwg.g(
                id=f"layer_{lp.index:02d}",
                fill=colour,
                stroke="none",
                opacity="0.8",
            )
            for d_str in lp.paths:
                group.add(dwg.path(d=d_str))
            dwg.add(group)
        dwg.save()
