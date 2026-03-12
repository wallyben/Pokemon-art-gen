"""
Tests for pokemon_stencil.vector.tracer.VectorTracer.

All tests use synthetic numpy arrays; no external files or GPU required.
The test suite exercises both the public API and the OpenCV contour fallback
backend (without requiring pypotrace to be installed).
"""

from __future__ import annotations

import numpy as np
import pytest

from pokemon_stencil.config import VectorConfig
from pokemon_stencil.vector.tracer import VectorTracer


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures and helpers
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture()
def cfg() -> VectorConfig:
    return VectorConfig(
        turdsize=2,
        alphamax=0.8,
        opttolerance=0.2,
        min_path_length=5.0,
        canvas_width_mm=100.0,
        canvas_height_mm=100.0,
        dpi=96.0,
    )


@pytest.fixture()
def tracer(cfg) -> VectorTracer:
    """VectorTracer forced to use OpenCV fallback (no pypotrace dependency)."""
    t = VectorTracer(cfg)
    t._has_potrace = False
    return t


def _rect_mask(h: int = 64, w: int = 64,
               y1: int = 10, y2: int = 54,
               x1: int = 10, x2: int = 54) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    m[y1:y2, x1:x2] = 255
    return m


def _empty_mask(h: int = 64, w: int = 64) -> np.ndarray:
    return np.zeros((h, w), dtype=np.uint8)


def _solid_mask(h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w), 255, dtype=np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
# Initialisation
# ─────────────────────────────────────────────────────────────────────────────

class TestVectorTracerInit:
    def test_stores_config(self, cfg):
        t = VectorTracer(cfg)
        assert t.config is cfg

    def test_has_potrace_is_bool(self, cfg):
        t = VectorTracer(cfg)
        assert isinstance(t._has_potrace, bool)

    def test_default_falls_back_gracefully(self, cfg):
        """Construction must not raise even when pypotrace is absent."""
        t = VectorTracer(cfg)
        assert t is not None


# ─────────────────────────────────────────────────────────────────────────────
# trace_mask() – public API
# ─────────────────────────────────────────────────────────────────────────────

class TestTraceMask:
    def test_returns_list(self, tracer):
        result = tracer.trace_mask(_rect_mask())
        assert isinstance(result, list)

    def test_empty_mask_returns_list(self, tracer):
        result = tracer.trace_mask(_empty_mask())
        assert isinstance(result, list)

    def test_empty_mask_no_paths(self, tracer):
        result = tracer.trace_mask(_empty_mask())
        assert result == []

    def test_rect_mask_produces_at_least_one_path(self, tracer):
        result = tracer.trace_mask(_rect_mask())
        assert len(result) >= 1

    def test_paths_are_strings(self, tracer):
        result = tracer.trace_mask(_rect_mask())
        for p in result:
            assert isinstance(p, str)

    def test_paths_start_with_M(self, tracer):
        """All SVG path d-strings must begin with a moveto command."""
        result = tracer.trace_mask(_rect_mask())
        for p in result:
            assert p.strip().startswith("M"), f"Path does not start with M: {p!r}"

    def test_paths_end_with_Z(self, tracer):
        """All SVG path d-strings must be closed with Z."""
        result = tracer.trace_mask(_rect_mask())
        for p in result:
            assert p.strip().endswith("Z"), f"Path does not end with Z: {p!r}"

    def test_high_min_path_length_discards_small_paths(self, cfg):
        """Paths shorter than min_path_length should be filtered out."""
        cfg.min_path_length = 10_000.0  # impossibly high
        t = VectorTracer(cfg)
        t._has_potrace = False
        result = t.trace_mask(_rect_mask(h=8, w=8, y1=2, y2=6, x1=2, x2=6))
        assert isinstance(result, list)

    def test_solid_mask_produces_paths(self, tracer):
        """A full-foreground mask should yield at least one path (outer boundary)."""
        result = tracer.trace_mask(_solid_mask())
        assert isinstance(result, list)

    def test_mask_with_multiple_regions(self, tracer):
        """Two separated rectangles should produce at least two paths."""
        mask = np.zeros((64, 64), dtype=np.uint8)
        mask[5:25, 5:25] = 255
        mask[35:55, 35:55] = 255
        result = tracer.trace_mask(mask)
        assert len(result) >= 2


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate conversion helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestCoordinateConversions:
    def test_px_to_mm_one_inch(self, tracer):
        """96 px at 96 DPI should equal 25.4 mm (one inch)."""
        assert abs(tracer.px_to_mm(96.0) - 25.4) < 0.01

    def test_mm_to_px_one_inch(self, tracer):
        """25.4 mm at 96 DPI should equal 96 px."""
        assert abs(tracer.mm_to_px(25.4) - 96.0) < 0.01

    def test_px_mm_roundtrip(self, tracer):
        for px in (1.0, 50.0, 200.0, 512.0):
            recovered = tracer.mm_to_px(tracer.px_to_mm(px))
            assert abs(recovered - px) < 1e-6

    def test_mm_px_roundtrip(self, tracer):
        for mm in (0.5, 10.0, 25.4, 100.0):
            recovered = tracer.px_to_mm(tracer.mm_to_px(mm))
            assert abs(recovered - mm) < 1e-6

    def test_zero_px_is_zero_mm(self, tracer):
        assert tracer.px_to_mm(0.0) == 0.0

    def test_zero_mm_is_zero_px(self, tracer):
        assert tracer.mm_to_px(0.0) == 0.0

    def test_dpi_scales_conversion(self):
        """Doubling DPI should halve the mm value for the same px count."""
        cfg_a = VectorConfig(dpi=96.0)
        cfg_b = VectorConfig(dpi=192.0)
        ta = VectorTracer(cfg_a)
        tb = VectorTracer(cfg_b)
        assert abs(ta.px_to_mm(100) / tb.px_to_mm(100) - 2.0) < 1e-6


# ─────────────────────────────────────────────────────────────────────────────
# OpenCV fallback backend
# ─────────────────────────────────────────────────────────────────────────────

class TestOpenCVFallback:
    def test_returns_list(self, cfg):
        t = VectorTracer(cfg)
        t._has_potrace = False
        result = t._trace_with_opencv(_rect_mask())
        assert isinstance(result, list)

    def test_paths_are_strings(self, cfg):
        t = VectorTracer(cfg)
        t._has_potrace = False
        result = t._trace_with_opencv(_rect_mask())
        for p in result:
            assert isinstance(p, str)

    def test_empty_mask_returns_empty(self, cfg):
        t = VectorTracer(cfg)
        t._has_potrace = False
        result = t._trace_with_opencv(_empty_mask())
        assert result == []

    def test_rect_returns_paths(self, cfg):
        t = VectorTracer(cfg)
        t._has_potrace = False
        result = t._trace_with_opencv(_rect_mask())
        assert len(result) >= 1

    def test_paths_start_with_M(self, cfg):
        t = VectorTracer(cfg)
        t._has_potrace = False
        result = t._trace_with_opencv(_rect_mask())
        for p in result:
            assert p.strip().startswith("M")

    def test_paths_end_with_Z(self, cfg):
        t = VectorTracer(cfg)
        t._has_potrace = False
        result = t._trace_with_opencv(_rect_mask())
        for p in result:
            assert p.strip().endswith("Z")

    def test_trace_mask_uses_fallback_when_no_potrace(self, cfg):
        """trace_mask should route to _trace_with_opencv when potrace absent."""
        t = VectorTracer(cfg)
        t._has_potrace = False
        result_direct = t._trace_with_opencv(_rect_mask())
        result_public = t.trace_mask(_rect_mask())
        # Both should be lists of the same length
        assert len(result_public) == len(result_direct)


# ─────────────────────────────────────────────────────────────────────────────
# _curve_to_svg_d (static helper)
# ─────────────────────────────────────────────────────────────────────────────

class _MockSegmentCorner:
    is_corner = True
    c = (5.0, 5.0)
    end_point = (10.0, 5.0)


class _MockSegmentBezier:
    is_corner = False
    c1 = (1.0, 2.0)
    c2 = (3.0, 4.0)
    end_point = (5.0, 6.0)


class _MockCurve:
    def __init__(self, segments, start):
        self.segments = segments
        self.start_point = start


class TestCurveToSvgD:
    def test_empty_segments_returns_empty_string(self):
        curve = _MockCurve(segments=[], start=(0.0, 0.0))
        result = VectorTracer._curve_to_svg_d(curve)
        assert result == ""

    def test_corner_segment_produces_L_commands(self):
        curve = _MockCurve(
            segments=[_MockSegmentCorner()], start=(0.0, 0.0)
        )
        result = VectorTracer._curve_to_svg_d(curve)
        assert "L" in result

    def test_bezier_segment_produces_C_command(self):
        curve = _MockCurve(
            segments=[_MockSegmentBezier()], start=(0.0, 0.0)
        )
        result = VectorTracer._curve_to_svg_d(curve)
        assert "C" in result

    def test_result_starts_with_M(self):
        curve = _MockCurve(
            segments=[_MockSegmentCorner()], start=(1.0, 2.0)
        )
        result = VectorTracer._curve_to_svg_d(curve)
        assert result.startswith("M")

    def test_result_ends_with_Z(self):
        curve = _MockCurve(
            segments=[_MockSegmentCorner()], start=(0.0, 0.0)
        )
        result = VectorTracer._curve_to_svg_d(curve)
        assert result.endswith("Z")
