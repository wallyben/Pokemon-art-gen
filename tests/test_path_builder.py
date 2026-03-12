"""
Tests for pokemon_stencil.vector.path_builder.

Covers both the ``LayerPaths`` dataclass and ``StencilPathBuilder``.
The ``VectorTracer`` dependency is mocked so tests are fast and isolated.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from pokemon_stencil.image_proc.segmenter import ColourLayer
from pokemon_stencil.vector.path_builder import LayerPaths, StencilPathBuilder
from pokemon_stencil.vector.tracer import VectorTracer


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures and helpers
# ─────────────────────────────────────────────────────────────────────────────

_SAMPLE_PATH = "M 10.000,10.000 L 50.000,10.000 L 50.000,50.000 L 10.000,50.000 Z"


@pytest.fixture()
def mock_tracer() -> MagicMock:
    t = MagicMock(spec=VectorTracer)
    t.trace_mask.return_value = [_SAMPLE_PATH]
    return t


@pytest.fixture()
def builder(mock_tracer) -> StencilPathBuilder:
    return StencilPathBuilder(mock_tracer)


def _make_layer(
    index: int,
    colour: tuple = (100, 150, 200),
    h: int = 32,
    w: int = 32,
) -> ColourLayer:
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[8:24, 8:24] = 255
    return ColourLayer(
        index=index,
        colour_rgb=colour,
        mask=mask,
        pixel_count=int(np.sum(mask == 255)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# LayerPaths dataclass
# ─────────────────────────────────────────────────────────────────────────────

class TestLayerPaths:
    def test_default_paths_is_empty_list(self):
        lp = LayerPaths(index=0, colour_rgb=(255, 0, 0))
        assert lp.paths == []

    def test_index_stored(self):
        lp = LayerPaths(index=7, colour_rgb=(0, 0, 0))
        assert lp.index == 7

    def test_colour_rgb_stored(self):
        lp = LayerPaths(index=0, colour_rgb=(10, 20, 30))
        assert lp.colour_rgb == (10, 20, 30)

    def test_paths_stored(self):
        paths = ["M 0,0 Z", "M 5,5 Z"]
        lp = LayerPaths(index=0, colour_rgb=(0, 0, 0), paths=paths)
        assert lp.paths == paths

    def test_paths_list_is_mutable(self):
        """The stored paths list can be appended to after construction."""
        lp = LayerPaths(index=0, colour_rgb=(0, 0, 0), paths=["M 0,0 Z"])
        lp.paths.append("M 1,1 Z")
        assert len(lp.paths) == 2


# ─────────────────────────────────────────────────────────────────────────────
# StencilPathBuilder.build_paths()
# ─────────────────────────────────────────────────────────────────────────────

class TestStencilPathBuilderContract:
    def test_returns_list(self, builder):
        result = builder.build_paths([_make_layer(0)])
        assert isinstance(result, list)

    def test_empty_layers_returns_empty(self, builder):
        assert builder.build_paths([]) == []

    def test_one_result_per_layer(self, builder):
        layers = [_make_layer(i) for i in range(5)]
        result = builder.build_paths(layers)
        assert len(result) == 5

    def test_result_items_are_layer_paths(self, builder):
        result = builder.build_paths([_make_layer(0)])
        assert isinstance(result[0], LayerPaths)

    def test_index_matches_layer(self, builder):
        layers = [_make_layer(i) for i in range(4)]
        result = builder.build_paths(layers)
        for lp, layer in zip(result, layers):
            assert lp.index == layer.index

    def test_colour_rgb_matches_layer(self, builder):
        colour = (11, 22, 33)
        layer = _make_layer(0, colour=colour)
        result = builder.build_paths([layer])
        assert result[0].colour_rgb == colour

    def test_paths_come_from_tracer(self, builder, mock_tracer):
        mock_tracer.trace_mask.return_value = ["M 0,0 Z", "M 1,1 Z"]
        result = builder.build_paths([_make_layer(0)])
        assert result[0].paths == ["M 0,0 Z", "M 1,1 Z"]

    def test_empty_tracer_result_stored(self, builder, mock_tracer):
        mock_tracer.trace_mask.return_value = []
        result = builder.build_paths([_make_layer(0)])
        assert result[0].paths == []


class TestStencilPathBuilderTracerInteraction:
    def test_trace_called_once_per_layer(self, builder, mock_tracer):
        layers = [_make_layer(i) for i in range(3)]
        builder.build_paths(layers)
        assert mock_tracer.trace_mask.call_count == 3

    def test_trace_called_with_layer_mask(self, builder, mock_tracer):
        layer = _make_layer(0)
        builder.build_paths([layer])
        called_mask = mock_tracer.trace_mask.call_args[0][0]
        np.testing.assert_array_equal(called_mask, layer.mask)

    def test_order_of_results_matches_order_of_layers(self, builder, mock_tracer):
        """Results must follow input layer order, not some other order."""
        colours = [(i * 10, i * 20, i * 30) for i in range(4)]
        layers = [_make_layer(i, colour=colours[i]) for i in range(4)]
        result = builder.build_paths(layers)
        for i, lp in enumerate(result):
            assert lp.colour_rgb == colours[i]

    def test_different_paths_per_layer(self, builder, mock_tracer):
        """Each layer should receive its own tracer output."""
        call_count = [0]

        def side_effect(mask):
            path = f"M {call_count[0]},0 Z"
            call_count[0] += 1
            return [path]

        mock_tracer.trace_mask.side_effect = side_effect
        layers = [_make_layer(i) for i in range(3)]
        result = builder.build_paths(layers)
        paths = [lp.paths[0] for lp in result]
        assert len(set(paths)) == 3  # all distinct
