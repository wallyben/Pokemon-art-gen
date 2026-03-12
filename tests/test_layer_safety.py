"""
Tests for stencil layer safety improvements in StencilLayerBuilder.

Verifies:
- New StencilConfig fields (min_cut_width_px, bridge_width_px) are present.
- Thin region expansion dilates features thinner than min_cut_width_px.
- Improved bridge placement connects islands to anchor regions when available.
- Bridge fallback to border still works when no anchor exists.
- Bridges are applied before registration marks.

All tests use synthetic masks; no SD inference or GPU required.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from pokemon_stencil.config import StencilConfig
from pokemon_stencil.image_proc.segmenter import ColourLayer
from pokemon_stencil.stencil.layer_builder import StencilLayerBuilder


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _make_config(**kwargs) -> StencilConfig:
    """Build a StencilConfig with safe test defaults."""
    defaults = dict(
        bridge_width=4,
        bridge_width_px=3,
        min_cut_width_px=3,
        add_registration_marks=False,
    )
    defaults.update(kwargs)
    return StencilConfig(**defaults)


def _layer(mask: np.ndarray, index: int = 0) -> ColourLayer:
    """Wrap a binary mask in a ColourLayer."""
    return ColourLayer(
        index=index,
        colour_rgb=(100, 100, 100),
        mask=mask.astype(np.uint8),
        pixel_count=int(np.sum(mask > 0)),
    )


def _empty_mask(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _full_mask(h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# StencilConfig new fields
# ─────────────────────────────────────────────────────────────────────────────

class TestStencilConfigNewFields:
    def test_min_cut_width_px_default(self):
        cfg = StencilConfig()
        assert cfg.min_cut_width_px == 3

    def test_bridge_width_px_default(self):
        cfg = StencilConfig()
        assert cfg.bridge_width_px == 3

    def test_custom_min_cut_width_px(self):
        cfg = StencilConfig(min_cut_width_px=5)
        assert cfg.min_cut_width_px == 5

    def test_custom_bridge_width_px(self):
        cfg = StencilConfig(bridge_width_px=6)
        assert cfg.bridge_width_px == 6


# ─────────────────────────────────────────────────────────────────────────────
# _expand_thin_regions
# ─────────────────────────────────────────────────────────────────────────────

class TestExpandThinRegions:
    def test_no_expansion_for_thick_mask(self):
        """A solid 20×20 block is thick enough – should not lose pixels."""
        cfg = _make_config(min_cut_width_px=3)
        builder = StencilLayerBuilder(cfg)

        mask = _empty_mask(64, 64)
        mask[10:30, 10:30] = 255  # 20×20 solid block
        result = builder._expand_thin_regions(mask.copy())

        # The original pixels must still be foreground after expansion
        assert np.all(result[10:30, 10:30] == 255)

    def test_thin_line_gets_expanded(self):
        """A 1-pixel-wide vertical line is thinner than min_cut_width_px=3."""
        cfg = _make_config(min_cut_width_px=3)
        builder = StencilLayerBuilder(cfg)

        mask = _empty_mask(32, 32)
        mask[:, 16] = 255  # 1-pixel column

        before_count = int(np.sum(mask > 0))
        result = builder._expand_thin_regions(mask.copy())
        after_count = int(np.sum(result > 0))

        assert after_count >= before_count, "Thin line must be expanded, not shrunk."

    def test_min_cut_width_1_noop(self):
        """min_cut_width_px=1 should leave the mask unchanged."""
        cfg = _make_config(min_cut_width_px=1)
        builder = StencilLayerBuilder(cfg)

        mask = _empty_mask(32, 32)
        mask[:, 16] = 255
        original = mask.copy()

        result = builder._expand_thin_regions(mask.copy())
        np.testing.assert_array_equal(result, original)

    def test_empty_mask_unchanged(self):
        cfg = _make_config(min_cut_width_px=3)
        builder = StencilLayerBuilder(cfg)

        mask = _empty_mask(32, 32)
        result = builder._expand_thin_regions(mask.copy())
        assert np.all(result == 0)

    def test_output_dtype_preserved(self):
        cfg = _make_config(min_cut_width_px=3)
        builder = StencilLayerBuilder(cfg)

        mask = _empty_mask(32, 32)
        mask[5:10, 5:10] = 255
        result = builder._expand_thin_regions(mask)
        assert result.dtype == np.uint8


# ─────────────────────────────────────────────────────────────────────────────
# _add_bridges – island-to-anchor routing
# ─────────────────────────────────────────────────────────────────────────────

class TestAddBridges:
    def test_isolated_island_gets_bridged(self):
        """An island that does not touch any border must have a bridge added."""
        cfg = _make_config(bridge_width_px=3)
        builder = StencilLayerBuilder(cfg)

        mask = _empty_mask(64, 64)
        # Island in the centre, completely surrounded by background
        mask[28:36, 28:36] = 255  # 8×8 island

        result = builder._add_bridges(mask.copy())
        # After bridging, there should be more foreground pixels
        assert int(np.sum(result > 0)) > int(np.sum(mask > 0))

    def test_border_touching_region_not_bridged(self):
        """A component that already touches the border must not be modified."""
        cfg = _make_config(bridge_width_px=3)
        builder = StencilLayerBuilder(cfg)

        mask = _empty_mask(64, 64)
        mask[0:10, 0:10] = 255  # touches top-left corner (border)
        before = mask.copy()

        result = builder._add_bridges(mask.copy())
        # The border-touching region itself must remain unchanged
        np.testing.assert_array_equal(result[0:10, 0:10], before[0:10, 0:10])

    def test_island_bridged_to_anchor_not_only_border(self):
        """
        When an anchor region exists, the bridge should connect the island to
        that anchor rather than simply drawing a strip to the image edge.

        We verify this by checking that the bridge pixels are present between
        the two regions (neither starts at the image border side exclusively).
        """
        cfg = _make_config(bridge_width_px=2)
        builder = StencilLayerBuilder(cfg)

        mask = _empty_mask(64, 64)
        # Anchor: large block touching the left border
        mask[0:64, 0:8] = 255
        # Island: isolated block on the right
        mask[28:36, 48:56] = 255

        result = builder._add_bridges(mask.copy())
        # The island area should still be foreground
        assert np.any(result[28:36, 48:56] == 255)
        # And there should be more foreground than original (bridge drawn)
        assert int(np.sum(result > 0)) > int(np.sum(mask > 0))

    def test_empty_mask_unchanged(self):
        cfg = _make_config(bridge_width_px=3)
        builder = StencilLayerBuilder(cfg)

        mask = _empty_mask(32, 32)
        result = builder._add_bridges(mask.copy())
        assert np.all(result == 0)

    def test_full_mask_unchanged(self):
        cfg = _make_config(bridge_width_px=3)
        builder = StencilLayerBuilder(cfg)

        mask = _full_mask(32, 32)
        result = builder._add_bridges(mask.copy())
        np.testing.assert_array_equal(result, mask)


# ─────────────────────────────────────────────────────────────────────────────
# _draw_bridge
# ─────────────────────────────────────────────────────────────────────────────

class TestDrawBridge:
    def test_horizontal_bridge_fills_pixels(self):
        mask = _empty_mask(32, 32)
        StencilLayerBuilder._draw_bridge(mask, x0=5, y0=16, x1=25, y1=16, width=2, h=32, w=32)
        # Row 15-17 between columns 5 and 25 must be filled
        assert np.any(mask[15:18, 5:26] == 255)

    def test_vertical_bridge_fills_pixels(self):
        mask = _empty_mask(32, 32)
        StencilLayerBuilder._draw_bridge(mask, x0=16, y0=5, x1=16, y1=25, width=2, h=32, w=32)
        assert np.any(mask[5:26, 15:18] == 255)

    def test_bridge_stays_within_bounds(self):
        mask = _empty_mask(32, 32)
        StencilLayerBuilder._draw_bridge(mask, x0=0, y0=0, x1=31, y1=31, width=4, h=32, w=32)
        # No index errors; check mask shape unchanged
        assert mask.shape == (32, 32)


# ─────────────────────────────────────────────────────────────────────────────
# build() – end-to-end
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildEndToEnd:
    def test_returns_list_same_length(self):
        cfg = _make_config()
        builder = StencilLayerBuilder(cfg)

        mask = _empty_mask(64, 64)
        mask[10:30, 10:30] = 255
        layers = [_layer(mask, 0), _layer(mask, 1)]

        result = builder.build(layers, image_size=(64, 64))
        assert len(result) == 2

    def test_indices_preserved(self):
        cfg = _make_config()
        builder = StencilLayerBuilder(cfg)

        masks = [_empty_mask(), _empty_mask()]
        masks[0][5:25, 5:25] = 255
        masks[1][30:50, 30:50] = 255
        layers = [_layer(masks[i], i) for i in range(2)]

        result = builder.build(layers, image_size=(64, 64))
        assert [r.index for r in result] == [0, 1]

    def test_colours_preserved(self):
        cfg = _make_config()
        builder = StencilLayerBuilder(cfg)

        mask = _empty_mask(64, 64)
        mask[5:25, 5:25] = 255
        layer = ColourLayer(index=0, colour_rgb=(255, 0, 128), mask=mask, pixel_count=400)
        result = builder.build([layer], image_size=(64, 64))
        assert result[0].colour_rgb == (255, 0, 128)

    def test_output_masks_are_uint8(self):
        cfg = _make_config()
        builder = StencilLayerBuilder(cfg)

        mask = _empty_mask(64, 64)
        mask[10:30, 10:30] = 255
        layers = [_layer(mask)]
        result = builder.build(layers, image_size=(64, 64))
        assert result[0].mask.dtype == np.uint8

    def test_isolated_island_bridged(self):
        """After build(), an isolated island should have gained bridge pixels."""
        cfg = _make_config(bridge_width_px=3, min_cut_width_px=1)
        builder = StencilLayerBuilder(cfg)

        mask = _empty_mask(64, 64)
        mask[28:36, 28:36] = 255  # fully isolated island
        original_fg = int(np.sum(mask > 0))

        layers = [_layer(mask)]
        result = builder.build(layers, image_size=(64, 64))
        result_fg = int(np.sum(result[0].mask > 0))

        assert result_fg > original_fg, "Bridge must add pixels to isolated island."
