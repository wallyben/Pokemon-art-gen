"""
Stencil path builder.

Coordinates the ``VectorTracer`` to convert each colour-layer mask into a
list of SVG path data strings, then packages the results into ``LayerPaths``
objects ready for export by ``SVGExporter``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Tuple

from pokemon_stencil.image_proc.segmenter import ColourLayer
from pokemon_stencil.vector.tracer import VectorTracer

logger = logging.getLogger(__name__)


@dataclass
class LayerPaths:
    """SVG path data for a single colour layer."""

    index: int
    """Zero-based layer index matching ``ColourLayer.index``."""

    colour_rgb: Tuple[int, int, int]
    """Representative RGB fill colour for this layer."""

    paths: List[str] = field(default_factory=list)
    """SVG path ``d`` attribute strings – one per closed contour."""


class StencilPathBuilder:
    """
    Converts a list of ``ColourLayer`` objects into ``LayerPaths`` via
    ``VectorTracer``.

    Args:
        tracer: Configured ``VectorTracer`` instance used to trace each mask.
    """

    def __init__(self, tracer: VectorTracer) -> None:
        self.tracer = tracer

    def build_paths(self, layers: List[ColourLayer]) -> List[LayerPaths]:
        """
        Trace each layer mask and return a parallel list of ``LayerPaths``.

        Args:
            layers: Processed colour layers (typically from
                    ``StencilLayerBuilder.build``).

        Returns:
            List of ``LayerPaths``, one per input layer, preserving order.
        """
        result: List[LayerPaths] = []
        for layer in layers:
            paths = self.tracer.trace_mask(layer.mask)
            logger.debug(
                "Layer %d traced: %d path(s) found.", layer.index, len(paths)
            )
            result.append(
                LayerPaths(
                    index=layer.index,
                    colour_rgb=layer.colour_rgb,
                    paths=paths,
                )
            )
        return result
