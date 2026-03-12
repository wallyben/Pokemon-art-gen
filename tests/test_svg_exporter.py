"""
Tests for pokemon_stencil.stencil.svg_exporter.SVGExporter.

All tests write to a temporary directory created by pytest; no permanent files
are created.  No GPU, internet, or heavy dependencies are required.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pokemon_stencil.config import StencilConfig, VectorConfig
from pokemon_stencil.stencil.svg_exporter import SVGExporter, _rgb_to_hex
from pokemon_stencil.vector.path_builder import LayerPaths


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures and helpers
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def vcfg() -> VectorConfig:
    return VectorConfig(
        canvas_width_mm=100.0,
        canvas_height_mm=100.0,
        dpi=96.0,
    )


@pytest.fixture()
def scfg() -> StencilConfig:
    return StencilConfig()


@pytest.fixture()
def exporter(vcfg, scfg) -> SVGExporter:
    return SVGExporter(vcfg, scfg)


def _lp(index: int, colour: tuple = (255, 100, 50),
        paths: list | None = None) -> LayerPaths:
    return LayerPaths(
        index=index,
        colour_rgb=colour,
        paths=paths or ["M 10,10 L 50,10 L 50,50 L 10,50 Z"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# _rgb_to_hex helper
# ─────────────────────────────────────────────────────────────────────────────

class TestRgbToHex:
    def test_black(self):
        assert _rgb_to_hex((0, 0, 0)) == "#000000"

    def test_white(self):
        assert _rgb_to_hex((255, 255, 255)) == "#ffffff"

    def test_pure_red(self):
        assert _rgb_to_hex((255, 0, 0)) == "#ff0000"

    def test_pure_green(self):
        assert _rgb_to_hex((0, 255, 0)) == "#00ff00"

    def test_pure_blue(self):
        assert _rgb_to_hex((0, 0, 255)) == "#0000ff"

    def test_custom_value(self):
        assert _rgb_to_hex((16, 32, 48)) == "#102030"

    def test_output_length(self):
        assert len(_rgb_to_hex((100, 200, 50))) == 7

    def test_starts_with_hash(self):
        assert _rgb_to_hex((1, 2, 3)).startswith("#")


# ─────────────────────────────────────────────────────────────────────────────
# SVGExporter.export() – output files
# ─────────────────────────────────────────────────────────────────────────────

class TestExportFilesCreated:
    def test_creates_output_dir(self, exporter, tmp_path):
        out = tmp_path / "svgs" / "nested"
        exporter.export([_lp(0)], out, "img000", (64, 64))
        assert out.is_dir()

    def test_writes_layer_svg(self, exporter, tmp_path):
        exporter.export([_lp(0)], tmp_path, "img000", (64, 64))
        assert (tmp_path / "img000_layer00.svg").exists()

    def test_writes_combined_svg_with_multiple_layers(self, exporter, tmp_path):
        exporter.export([_lp(0), _lp(1)], tmp_path, "img000", (64, 64))
        assert (tmp_path / "img000_combined.svg").exists()

    def test_writes_combined_svg_with_single_layer(self, exporter, tmp_path):
        exporter.export([_lp(0)], tmp_path, "img000", (64, 64))
        assert (tmp_path / "img000_combined.svg").exists()

    def test_no_files_for_empty_layers(self, exporter, tmp_path):
        result = exporter.export([], tmp_path, "prefix", (64, 64))
        assert result == []

    def test_layer_file_naming(self, exporter, tmp_path):
        exporter.export([_lp(0), _lp(1), _lp(2)], tmp_path, "p", (64, 64))
        assert (tmp_path / "p_layer00.svg").exists()
        assert (tmp_path / "p_layer01.svg").exists()
        assert (tmp_path / "p_layer02.svg").exists()

    def test_prefix_applied_correctly(self, exporter, tmp_path):
        exporter.export([_lp(0)], tmp_path, "run42", (64, 64))
        assert (tmp_path / "run42_layer00.svg").exists()
        assert (tmp_path / "run42_combined.svg").exists()

    def test_multiple_prefixes_do_not_overwrite(self, exporter, tmp_path):
        exporter.export([_lp(0)], tmp_path, "a", (64, 64))
        exporter.export([_lp(0)], tmp_path, "b", (64, 64))
        assert (tmp_path / "a_layer00.svg").exists()
        assert (tmp_path / "b_layer00.svg").exists()


# ─────────────────────────────────────────────────────────────────────────────
# SVGExporter.export() – return value
# ─────────────────────────────────────────────────────────────────────────────

class TestExportReturnValue:
    def test_returns_list(self, exporter, tmp_path):
        result = exporter.export([_lp(0)], tmp_path, "img000", (64, 64))
        assert isinstance(result, list)

    def test_all_returned_paths_are_path_objects(self, exporter, tmp_path):
        result = exporter.export([_lp(0), _lp(1)], tmp_path, "img000", (64, 64))
        assert all(isinstance(p, Path) for p in result)

    def test_count_layer_plus_combined(self, exporter, tmp_path):
        """N layers → N per-layer SVGs + 1 combined = N+1 files."""
        n = 4
        result = exporter.export([_lp(i) for i in range(n)], tmp_path, "x", (64, 64))
        assert len(result) == n + 1

    def test_all_returned_files_exist(self, exporter, tmp_path):
        result = exporter.export([_lp(0), _lp(1)], tmp_path, "img000", (64, 64))
        for p in result:
            assert p.exists()

    def test_empty_returns_empty_list(self, exporter, tmp_path):
        result = exporter.export([], tmp_path, "prefix", (64, 64))
        assert result == []


# ─────────────────────────────────────────────────────────────────────────────
# SVG file content
# ─────────────────────────────────────────────────────────────────────────────

class TestSVGContent:
    def test_layer_svg_contains_path_data(self, exporter, tmp_path):
        lp = _lp(0, paths=["M 10,10 L 50,50 Z"])
        exporter.export([lp], tmp_path, "img000", (64, 64))
        content = (tmp_path / "img000_layer00.svg").read_text()
        assert "M 10,10" in content

    def test_layer_svg_contains_fill_colour(self, exporter, tmp_path):
        lp = LayerPaths(index=0, colour_rgb=(255, 0, 0), paths=["M 0,0 Z"])
        exporter.export([lp], tmp_path, "img000", (64, 64))
        content = (tmp_path / "img000_layer00.svg").read_text()
        assert "#ff0000" in content

    def test_combined_svg_contains_all_layer_groups(self, exporter, tmp_path):
        exporter.export([_lp(0), _lp(1), _lp(2)], tmp_path, "img000", (64, 64))
        content = (tmp_path / "img000_combined.svg").read_text()
        assert "layer_00" in content
        assert "layer_01" in content
        assert "layer_02" in content

    def test_layer_svg_is_valid_xml_start(self, exporter, tmp_path):
        """SVG files should begin with an XML/SVG declaration."""
        exporter.export([_lp(0)], tmp_path, "img000", (64, 64))
        content = (tmp_path / "img000_layer00.svg").read_text()
        assert "<svg" in content

    def test_layer_svg_group_id_correct(self, exporter, tmp_path):
        exporter.export([_lp(3)], tmp_path, "img000", (64, 64))
        content = (tmp_path / "img000_layer03.svg").read_text()
        assert "layer_03" in content

    def test_combined_svg_all_path_data_present(self, exporter, tmp_path):
        lp0 = _lp(0, paths=["M 1,1 Z"])
        lp1 = _lp(1, paths=["M 2,2 Z"])
        exporter.export([lp0, lp1], tmp_path, "img000", (64, 64))
        content = (tmp_path / "img000_combined.svg").read_text()
        assert "M 1,1" in content
        assert "M 2,2" in content

    def test_layer_svg_with_no_paths_still_written(self, exporter, tmp_path):
        """A layer with zero traced paths should still produce a valid SVG."""
        lp = LayerPaths(index=0, colour_rgb=(0, 0, 0), paths=[])
        result = exporter.export([lp], tmp_path, "img000", (64, 64))
        assert (tmp_path / "img000_layer00.svg").exists()

    def test_viewbox_set(self, exporter, tmp_path):
        exporter.export([_lp(0)], tmp_path, "img000", (128, 96))
        content = (tmp_path / "img000_layer00.svg").read_text()
        assert "viewBox" in content or "viewbox" in content.lower()


# ─────────────────────────────────────────────────────────────────────────────
# SVGExporter configuration
# ─────────────────────────────────────────────────────────────────────────────

class TestSVGExporterConfig:
    def test_stores_vector_config(self, vcfg, scfg):
        e = SVGExporter(vcfg, scfg)
        assert e.vcfg is vcfg

    def test_stores_stencil_config(self, vcfg, scfg):
        e = SVGExporter(vcfg, scfg)
        assert e.scfg is scfg
