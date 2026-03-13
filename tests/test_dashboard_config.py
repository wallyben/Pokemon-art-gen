"""
Tests for the Streamlit dashboard module.

These tests verify that the dashboard imports correctly, that configuration
loads as expected, and that the pipeline is callable from the UI layer.
No Streamlit server is started; no SD model is loaded.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

# Pre-load all C-extension-heavy modules that the dashboard app pulls in transitively.
# patch.dict(sys.modules) saves/restores the full dict; any module first imported
# *inside* the block gets evicted on exit, and C extensions cannot be reloaded.
# Importing these here ensures they are already in sys.modules before the first
# patch.dict block runs, so they are preserved across all test methods.
import cv2  # noqa: F401
import numpy  # noqa: F401
import pokemon_stencil.factory.factory_runner  # noqa: F401
import pokemon_stencil.image_proc.segmenter  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _mock_streamlit():
    """
    Return a lightweight MagicMock that stands in for the ``streamlit`` module.

    dashboard/app.py calls ``st.set_page_config()`` at import time (module
    level), so we must patch ``streamlit`` before importing the dashboard.
    """
    st_mock = MagicMock()
    # set_page_config must be a no-op callable at module scope.
    st_mock.set_page_config.return_value = None
    st_mock.session_state = {}
    return st_mock


# ─────────────────────────────────────────────────────────────────────────────
# Import correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestDashboardImports:
    """The dashboard module must import cleanly with streamlit mocked."""

    def test_dashboard_app_module_importable(self):
        """pokemon_stencil.dashboard.app should be importable."""
        st_mock = _mock_streamlit()
        with patch.dict(sys.modules, {"streamlit": st_mock}):
            # Remove any cached version so we re-import under the mock.
            sys.modules.pop("pokemon_stencil.dashboard.app", None)
            import pokemon_stencil.dashboard.app as app_module  # noqa: F401
        assert app_module is not None

    def test_dashboard_package_importable(self):
        """pokemon_stencil.dashboard package __init__ should import cleanly."""
        import pokemon_stencil.dashboard  # noqa: F401

    def test_app_exposes_main_function(self):
        """dashboard.app must expose a callable ``main`` function."""
        st_mock = _mock_streamlit()
        with patch.dict(sys.modules, {"streamlit": st_mock}):
            sys.modules.pop("pokemon_stencil.dashboard.app", None)
            import pokemon_stencil.dashboard.app as app_module
        assert callable(app_module.main)

    def test_app_exposes_progress_runner(self):
        """_ProgressFactoryRunner must be defined in the dashboard module."""
        st_mock = _mock_streamlit()
        with patch.dict(sys.modules, {"streamlit": st_mock}):
            sys.modules.pop("pokemon_stencil.dashboard.app", None)
            import pokemon_stencil.dashboard.app as app_module
        assert hasattr(app_module, "_ProgressFactoryRunner")

    def test_stage_labels_defined(self):
        """_STAGE_LABELS constant must be a non-empty list of strings."""
        st_mock = _mock_streamlit()
        with patch.dict(sys.modules, {"streamlit": st_mock}):
            sys.modules.pop("pokemon_stencil.dashboard.app", None)
            import pokemon_stencil.dashboard.app as app_module
        labels = app_module._STAGE_LABELS
        assert isinstance(labels, list)
        assert len(labels) > 0
        assert all(isinstance(s, str) for s in labels)


# ─────────────────────────────────────────────────────────────────────────────
# Configuration loading
# ─────────────────────────────────────────────────────────────────────────────

class TestDashboardConfiguration:
    """Verify that _build_config produces a valid, usable PipelineConfig."""

    @pytest.fixture(autouse=True)
    def _import_app(self):
        st_mock = _mock_streamlit()
        with patch.dict(sys.modules, {"streamlit": st_mock}):
            sys.modules.pop("pokemon_stencil.dashboard.app", None)
            import pokemon_stencil.dashboard.app as app_module
        self.app = app_module

    def test_build_config_returns_pipeline_config(self, tmp_path):
        from pokemon_stencil.config import PipelineConfig
        cfg = self.app._build_config(tmp_path, candidate_count=5, top_k=2)
        assert isinstance(cfg, PipelineConfig)

    def test_build_config_sets_candidate_count(self, tmp_path):
        cfg = self.app._build_config(tmp_path, candidate_count=7, top_k=3)
        assert cfg.factory.count == 7

    def test_build_config_sets_top_k(self, tmp_path):
        cfg = self.app._build_config(tmp_path, candidate_count=10, top_k=4)
        assert cfg.factory.top_k == 4

    def test_build_config_uses_cpu(self, tmp_path):
        cfg = self.app._build_config(tmp_path, candidate_count=3, top_k=1)
        assert cfg.generation.device == "cpu"

    def test_build_config_output_dir(self, tmp_path):
        cfg = self.app._build_config(tmp_path, candidate_count=3, top_k=1)
        assert cfg.output.base_dir == tmp_path

    def test_build_config_sequential_workers(self, tmp_path):
        """Dashboard always uses workers=1 for Streamlit compatibility."""
        cfg = self.app._build_config(tmp_path, candidate_count=5, top_k=2)
        assert cfg.factory.workers == 1

    def test_build_config_float32_dtype(self, tmp_path):
        cfg = self.app._build_config(tmp_path, candidate_count=5, top_k=2)
        assert cfg.generation.torch_dtype == "float32"


# ─────────────────────────────────────────────────────────────────────────────
# _ProgressFactoryRunner
# ─────────────────────────────────────────────────────────────────────────────

class TestProgressFactoryRunner:
    """Verify the progress-tracking subclass behaves correctly."""

    @pytest.fixture(autouse=True)
    def _import_app(self):
        st_mock = _mock_streamlit()
        with patch.dict(sys.modules, {"streamlit": st_mock}):
            sys.modules.pop("pokemon_stencil.dashboard.app", None)
            import pokemon_stencil.dashboard.app as app_module
        self.app = app_module

    def test_progress_runner_is_factory_runner_subclass(self, tmp_path):
        from pokemon_stencil.config import PipelineConfig
        from pokemon_stencil.factory.factory_runner import FactoryRunner
        cfg = PipelineConfig()
        cfg.output.base_dir = tmp_path
        runner = self.app._ProgressFactoryRunner(cfg)
        assert isinstance(runner, FactoryRunner)

    def test_on_candidate_done_callback_called(self, tmp_path):
        """Callback fires once per candidate in skip_generation mode."""
        from pokemon_stencil.config import PipelineConfig

        config = PipelineConfig(skip_generation=True)
        config.factory.count = 2
        config.factory.top_k = 1
        config.output.base_dir = tmp_path

        # Create reference images so the runner doesn't error.
        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        for i in range(2):
            Image.new("RGB", (64, 64), (i * 100, 150, 200)).save(
                ref_dir / f"ref_{i}.png"
            )

        calls: list[tuple[int, int]] = []

        runner = self.app._ProgressFactoryRunner(
            config,
            on_candidate_done=lambda done, total: calls.append((done, total)),
        )

        try:
            runner.run("Pikachu", reference_dir=ref_dir)
        except Exception:
            pass  # SVG/tracing failures are fine; we just check callbacks.

        # Should have fired at least once (one per candidate).
        assert len(calls) >= 1
        # Total argument should always equal factory.count.
        assert all(total == 2 for _, total in calls)

    def test_callbacks_default_to_none(self, tmp_path):
        """Runner works fine when no callbacks are supplied."""
        from pokemon_stencil.config import PipelineConfig

        config = PipelineConfig(skip_generation=True)
        config.factory.count = 1
        config.factory.top_k = 1
        config.output.base_dir = tmp_path

        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        Image.new("RGB", (64, 64), (200, 150, 100)).save(ref_dir / "ref.png")

        runner = self.app._ProgressFactoryRunner(config)  # no callbacks
        assert runner._on_candidate_done is None
        assert runner._on_stencil_done is None

    def test_progress_runner_accepts_stencil_callback(self, tmp_path):
        """on_stencil_done keyword is accepted without error."""
        from pokemon_stencil.config import PipelineConfig

        config = PipelineConfig()
        config.output.base_dir = tmp_path
        stencil_calls: list = []
        runner = self.app._ProgressFactoryRunner(
            config,
            on_stencil_done=lambda d, t: stencil_calls.append((d, t)),
        )
        assert runner._on_stencil_done is not None


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

class TestDashboardHelpers:
    """Unit tests for the module-level helper functions."""

    @pytest.fixture(autouse=True)
    def _import_app(self):
        st_mock = _mock_streamlit()
        with patch.dict(sys.modules, {"streamlit": st_mock}):
            sys.modules.pop("pokemon_stencil.dashboard.app", None)
            import pokemon_stencil.dashboard.app as app_module
        self.app = app_module

    def test_find_preview_images_returns_empty_when_dir_absent(self, tmp_path):
        result = self.app._find_preview_images(tmp_path, "nonexistent_run")
        assert result == []

    def test_find_preview_images_returns_pngs(self, tmp_path):
        run_name = "pikachu_00"
        img_dir = tmp_path / run_name / "images"
        img_dir.mkdir(parents=True)
        (img_dir / "source_000.png").write_bytes(b"")
        (img_dir / "source_001.png").write_bytes(b"")

        result = self.app._find_preview_images(tmp_path, run_name)
        assert len(result) == 2
        assert all(p.suffix == ".png" for p in result)

    def test_find_svg_files_returns_empty_when_absent(self, tmp_path):
        result = self.app._find_svg_files(tmp_path, "no_run")
        assert result == []

    def test_find_svg_files_returns_svgs(self, tmp_path):
        run_name = "pikachu_00"
        svg_dir = tmp_path / run_name / "svg"
        svg_dir.mkdir(parents=True)
        (svg_dir / "layer_00.svg").write_text("<svg/>")
        (svg_dir / "layer_01.svg").write_text("<svg/>")

        result = self.app._find_svg_files(tmp_path, run_name)
        assert len(result) == 2
        assert all(p.suffix == ".svg" for p in result)

    def test_refs_dir_for_lowercases_name(self):
        result = self.app._refs_dir_for("Pikachu")
        assert "pikachu" in str(result).lower()

    def test_refs_dir_for_replaces_spaces_with_hyphens(self):
        result = self.app._refs_dir_for("Mr Mime")
        assert "mr-mime" in str(result)


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline callable from dashboard layer
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineCallableFromDashboard:
    """Verify the dashboard can construct and invoke the factory pipeline."""

    def test_factory_runner_instantiates_from_dashboard_config(self, tmp_path):
        from pokemon_stencil.config import PipelineConfig
        from pokemon_stencil.factory.factory_runner import FactoryRunner

        st_mock = _mock_streamlit()
        with patch.dict(sys.modules, {"streamlit": st_mock}):
            sys.modules.pop("pokemon_stencil.dashboard.app", None)
            import pokemon_stencil.dashboard.app as app_module

        cfg = app_module._build_config(tmp_path, candidate_count=3, top_k=1)
        runner = FactoryRunner(cfg)
        assert runner is not None

    def test_skip_generation_run_completes(self, tmp_path):
        """End-to-end run with skip_generation=True should not raise."""
        from pokemon_stencil.config import PipelineConfig

        st_mock = _mock_streamlit()
        with patch.dict(sys.modules, {"streamlit": st_mock}):
            sys.modules.pop("pokemon_stencil.dashboard.app", None)
            import pokemon_stencil.dashboard.app as app_module

        config = PipelineConfig(skip_generation=True)
        config.factory.count = 1
        config.factory.top_k = 1
        config.factory.workers = 1
        config.output.base_dir = tmp_path

        ref_dir = tmp_path / "refs"
        ref_dir.mkdir()
        Image.new("RGB", (64, 64), (200, 150, 100)).save(ref_dir / "ref.png")

        runner = app_module._ProgressFactoryRunner(config)

        # Should not raise — stencil export failures are captured internally.
        result = runner.run("Pikachu", reference_dir=ref_dir)
        # Either succeeded or recorded errors — but no unhandled exception.
        assert hasattr(result, "succeeded")
        assert hasattr(result, "errors")
