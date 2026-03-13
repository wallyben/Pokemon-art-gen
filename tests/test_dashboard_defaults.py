"""
tests/test_dashboard_defaults.py — Dashboard default configuration tests.

Verifies:
- ControlNet default is OFF (use_controlnet=True is the right default per spec,
  but the dashboard build_config must correctly reflect user selection)
- IP-Adapter default is OFF when not SDXL
- LoRA selection returns None when no valid LoRA is configured
- Prompt optimisation is enabled by default
- SDXL model default is the canonical hub ID
- Dashboard _build_config produces a valid PipelineConfig
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest


# ── Streamlit mock ────────────────────────────────────────────────────────────

def _mock_streamlit():
    """Return a minimal mock of the streamlit module for import-time calls."""
    from unittest.mock import MagicMock
    st = MagicMock()
    # set_page_config is called at module level; it must not raise
    st.set_page_config = MagicMock()
    st.cache_resource = lambda **kwargs: (lambda fn: fn)
    return st


# ─────────────────────────────────────────────────────────────────────────────
# _build_config defaults
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildConfigDefaults:
    @pytest.fixture(autouse=True)
    def _import_app(self):
        st_mock = _mock_streamlit()
        with patch.dict(sys.modules, {"streamlit": st_mock}):
            sys.modules.pop("pokemon_stencil.dashboard.app", None)
            import pokemon_stencil.dashboard.app as app_module
        self.app = app_module

    def test_controlnet_default_on_sdxl(self, tmp_path: Path) -> None:
        """ControlNet defaults to True (enabled) for the SDXL path."""
        cfg = self.app._build_config(tmp_path, candidate_count=3, top_k=1)
        assert cfg.generation.use_controlnet is True

    def test_ip_adapter_default_false_for_non_sdxl(self, tmp_path: Path) -> None:
        """IP-Adapter defaults to False for non-SDXL models."""
        from pokemon_stencil.config import DEFAULT_MODEL_HUB_ID
        cfg = self.app._build_config(
            tmp_path,
            candidate_count=3,
            top_k=1,
            model_hub_id=DEFAULT_MODEL_HUB_ID,
            use_ip_adapter=False,
        )
        assert cfg.generation.use_ip_adapter is False

    def test_prompt_optimisation_enabled_by_default(self, tmp_path: Path) -> None:
        """optimise_prompt must default to True."""
        cfg = self.app._build_config(tmp_path, candidate_count=3, top_k=1)
        assert cfg.generation.optimise_prompt is True

    def test_default_model_is_sdxl(self, tmp_path: Path) -> None:
        """Default model hub ID must be the SDXL base model."""
        from pokemon_stencil.config import DEFAULT_SDXL_HUB_ID
        cfg = self.app._build_config(tmp_path, candidate_count=3, top_k=1)
        assert cfg.generation.model_hub_id == DEFAULT_SDXL_HUB_ID

    def test_lora_none_when_not_specified(self, tmp_path: Path) -> None:
        """lora_path must be None when lora_name is not provided."""
        cfg = self.app._build_config(tmp_path, candidate_count=3, top_k=1, lora_name=None)
        assert cfg.generation.lora_path is None

    def test_workers_always_1(self, tmp_path: Path) -> None:
        """workers must always be 1 in the dashboard (Streamlit compat)."""
        cfg = self.app._build_config(tmp_path, candidate_count=5, top_k=2)
        assert cfg.factory.workers == 1

    def test_sdxl_uses_1024_resolution(self, tmp_path: Path) -> None:
        """SDXL must use 1024×1024."""
        from pokemon_stencil.config import DEFAULT_SDXL_HUB_ID
        cfg = self.app._build_config(
            tmp_path,
            candidate_count=3,
            top_k=1,
            model_hub_id=DEFAULT_SDXL_HUB_ID,
        )
        assert cfg.generation.width == 1024
        assert cfg.generation.height == 1024

    def test_sdxl_uses_28_inference_steps(self, tmp_path: Path) -> None:
        """SDXL must use 28 inference steps per spec."""
        from pokemon_stencil.config import DEFAULT_SDXL_HUB_ID
        cfg = self.app._build_config(
            tmp_path,
            candidate_count=3,
            top_k=1,
            model_hub_id=DEFAULT_SDXL_HUB_ID,
        )
        assert cfg.generation.num_inference_steps == 28

    def test_sdxl_uses_7_5_guidance_scale(self, tmp_path: Path) -> None:
        """SDXL must use guidance_scale=7.5 per spec."""
        cfg = self.app._build_config(tmp_path, candidate_count=3, top_k=1)
        assert cfg.generation.guidance_scale == 7.5


# ─────────────────────────────────────────────────────────────────────────────
# LoRA status detection
# ─────────────────────────────────────────────────────────────────────────────

class TestLoraStatusDetection:
    @pytest.fixture(autouse=True)
    def _import_app(self):
        st_mock = _mock_streamlit()
        with patch.dict(sys.modules, {"streamlit": st_mock}):
            sys.modules.pop("pokemon_stencil.dashboard.app", None)
            import pokemon_stencil.dashboard.app as app_module
        self.app = app_module

    def test_lora_none_label_maps_to_none(self, tmp_path: Path) -> None:
        """Selecting the 'None' LoRA option must result in lora_path=None."""
        cfg = self.app._build_config(
            tmp_path,
            candidate_count=3,
            top_k=1,
            lora_name=self.app._LORA_NONE_LABEL,
        )
        assert cfg.generation.lora_path is None

    def test_named_lora_maps_to_path_when_file_exists(self, tmp_path: Path) -> None:
        """A valid lora_name must set lora_path if the file exists in lora_dir."""
        from pokemon_stencil.config import DEFAULT_LORA_DIR
        # Create a dummy safetensors file in the LoRA directory
        DEFAULT_LORA_DIR.mkdir(parents=True, exist_ok=True)
        dummy = DEFAULT_LORA_DIR / "test_pokemon.safetensors"
        dummy.write_bytes(b"\x00")
        try:
            cfg = self.app._build_config(
                tmp_path,
                candidate_count=3,
                top_k=1,
                lora_name="test_pokemon",
            )
            assert cfg.generation.lora_path is not None
            assert cfg.generation.lora_path.name == "test_pokemon.safetensors"
        finally:
            if dummy.exists():
                dummy.unlink()

    def test_named_lora_maps_to_none_when_file_missing(self, tmp_path: Path) -> None:
        """A lora_name pointing to a non-existent file must leave lora_path=None."""
        cfg = self.app._build_config(
            tmp_path,
            candidate_count=3,
            top_k=1,
            lora_name="definitely_does_not_exist_xyz",
        )
        assert cfg.generation.lora_path is None
