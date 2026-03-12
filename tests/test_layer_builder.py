"""
Tests for pokemon_stencil.stencil.layer_builder.StencilLayerBuilder.

All tests use synthetic numpy arrays; no external files or GPU required.
"""

from __future__ import annotations

import numpy as np
import pytest

from pokemon_stencil.config import StencilConfig
from pokemon_stencil.image_proc.segmenter import ColourLayer
from pokemon_stencil.stencil.layer_builder import StencilLayerBuilder


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures and helpers
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def cfg() -> StencilConfig:
    """StencilConfig with small bridge and mark sizes for fast tests."""
    return StencilConfig(
        bridge_width=4,
        outline_stroke_width=2.0,
        min_cut_width_mm=0.8,
        add_registration_marks=True,
        reg_mark_radius_mm=2.0,
        padding_mm=5.0,
    )


@pytest.fixture()
def builder(cfg) -> StencilLayerBuilder:
    return StencilLayerBuilder(cfg)


def _blank_mask(h: int = 64, w: int = 64, value: int = 0) -> np.ndarray:
    """Return a uint8 mask filled with *value* (0 or 255)."""
    return np.full((h, w), value, dtype=np.uint8)


def _make_layer(
    index: int,
    mask: np.ndarray,
    colour: tuple = (100, 150, 200),
) -> ColourLayer:
    return ColourLayer(
        index=index,
        colour_rgb=colour,
        mask=mask,
        pixel_count=int(np.sum(mask == 255)),
    )


# ─────────────────────────────────────────────────────────────────────────────
# StencilLayerBuilder.build() – basic contract
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildContract:
    def test_returns_list(self, builder):
        result = builder.build([], (64, 64))
        assert isinstance(result, list)

    def test_empty_layers_returns_empty(self, builder):
        assert builder.build([], (64, 64)) == []

    def test_returns_same_count(self, builder):
        layers = [_make_layer(i, _blank_mask(value=255)) for i in range(3)]
        result = builder.build(layers, (64, 64))
        assert len(result) == 3

    def test_result_items_are_colour_layers(self, builder):
        layer = _make_layer(0, _blank_mask(value=255))
        result = builder.build([layer], (64, 64))
        assert isinstance(result[0], ColourLayer)

    def test_index_preserved(self, builder):
        layers = [_make_layer(i, _blank_mask()) for i in range(4)]
        result = builder.build(layers, (64, 64))
        assert [r.index for r in result] == [0, 1, 2, 3]

    def test_colour_rgb_preserved(self, builder):
        colour = (10, 20, 30)
        layer = _make_layer(0, _blank_mask(), colour=colour)
        result = builder.build([layer], (64, 64))
        assert result[0].colour_rgb == colour

    def test_mask_is_uint8(self, builder):
        layer = _make_layer(0, _blank_mask(value=255))
        result = builder.build([layer], (64, 64))
        assert result[0].mask.dtype == np.uint8

    def test_mask_shape_preserved(self, builder):
        mask = _blank_mask(h=64, w=64, value=255)
        result = builder.build([_make_layer(0, mask)], (64, 64))
        assert result[0].mask.shape == (64, 64)

    def test_pixel_count_updated(self, builder):
        mask = _blank_mask(value=0)
        result = builder.build([_make_layer(0, mask, (0, 0, 0))], (64, 64))
        # pixel_count should reflect the final mask state
        assert result[0].pixel_count == int(np.sum(result[0].mask == 255))

    def test_original_mask_not_mutated(self, builder):
        """build() should not modify the input layer masks."""
        mask = _blank_mask(value=0)
        original = mask.copy()
        builder.build([_make_layer(0, mask)], (64, 64))
        np.testing.assert_array_equal(mask, original)


# ─────────────────────────────────────────────────────────────────────────────
# Bridge insertion
# ─────────────────────────────────────────────────────────────────────────────

class TestBridgeInsertion:
    def test_isolated_island_expands_foreground(self, builder):
        """An isolated island should gain bridge pixels connecting to the border."""
        mask = _blank_mask(value=0)
        mask[20:44, 20:44] = 255  # solid square not touching any border
        layer = _make_layer(0, mask)
        builder.config.add_registration_marks = False
        result = builder.build([layer], (64, 64))[0]
        assert result.mask.sum() > mask.sum()

    def test_border_touching_region_not_changed(self, builder):
        """A region touching the border should not shrink (bridges not needed)."""
        mask = _blank_mask(value=0)
        mask[0:20, 10:50] = 255  # touches the top border
        original_sum = int(mask.sum())
        layer = _make_layer(0, mask)
        builder.config.add_registration_marks = False
        result = builder.build([layer], (64, 64))[0]
        assert int(result.mask.sum()) >= original_sum

    def test_all_background_mask_no_error(self, builder):
        """An all-zero mask must not raise an exception."""
        mask = _blank_mask(value=0)
        builder.config.add_registration_marks = False
        result = builder.build([_make_layer(0, mask)], (64, 64))
        assert isinstance(result[0].mask, np.ndarray)

    def test_all_foreground_mask_no_error(self, builder):
        """An all-foreground mask must not raise an exception."""
        mask = _blank_mask(value=255)
        builder.config.add_registration_marks = False
        result = builder.build([_make_layer(0, mask)], (64, 64))
        assert isinstance(result[0].mask, np.ndarray)

    def test_bridge_values_are_binary(self, builder):
        """After bridge insertion mask should only contain 0 and 255."""
        mask = _blank_mask(value=0)
        mask[20:44, 20:44] = 255
        builder.config.add_registration_marks = False
        result = builder.build([_make_layer(0, mask)], (64, 64))[0]
        unique = set(np.unique(result.mask).tolist())
        assert unique.issubset({0, 255})

    def test_multiple_islands_each_get_bridge(self, builder):
        """Two isolated islands should both receive bridges."""
        mask = _blank_mask(value=0)
        mask[10:20, 10:20] = 255   # first island
        mask[44:54, 44:54] = 255   # second island
        builder.config.add_registration_marks = False
        result = builder.build([_make_layer(0, mask)], (64, 64))[0]
        # More foreground than the two islands alone
        island_sum = int(mask.sum())
        assert int(result.mask.sum()) > island_sum


# ─────────────────────────────────────────────────────────────────────────────
# Registration marks
# ─────────────────────────────────────────────────────────────────────────────

class TestRegistrationMarks:
    def test_marks_add_foreground_pixels(self, cfg):
        """Registration marks should add foreground pixels to a blank mask."""
        builder = StencilLayerBuilder(cfg)
        mask = _blank_mask(value=0)
        result = builder.build([_make_layer(0, mask)], (64, 64))[0]
        assert result.mask.sum() > 0

    def test_marks_disabled_leaves_blank_mask_blank(self):
        """With marks disabled, a blank mask should stay blank (no bridges either)."""
        cfg = StencilConfig(add_registration_marks=False)
        builder = StencilLayerBuilder(cfg)
        mask = _blank_mask(value=0)
        result = builder.build([_make_layer(0, mask)], (64, 64))[0]
        assert result.mask.sum() == 0

    def test_marks_present_on_all_layers(self, cfg):
        """Every layer should have registration marks when enabled."""
        builder = StencilLayerBuilder(cfg)
        layers = [_make_layer(i, _blank_mask(value=0)) for i in range(3)]
        results = builder.build(layers, (64, 64))
        for r in results:
            assert r.mask.sum() > 0

    def test_marks_values_are_binary(self, cfg):
        """Mask with registration marks must only contain 0 and 255."""
        builder = StencilLayerBuilder(cfg)
        mask = _blank_mask(value=0)
        result = builder.build([_make_layer(0, mask)], (64, 64))[0]
        unique = set(np.unique(result.mask).tolist())
        assert unique.issubset({0, 255})

    def test_mark_radius_zero_still_works(self):
        """Very small registration marks should not raise an error."""
        cfg = StencilConfig(add_registration_marks=True, reg_mark_radius_mm=0.1)
        builder = StencilLayerBuilder(cfg)
        mask = _blank_mask(value=0)
        result = builder.build([_make_layer(0, mask)], (64, 64))
        assert isinstance(result[0].mask, np.ndarray)
