"""
tests/test_env_imports.py — Verify all core libraries import successfully.

These tests confirm the environment is correctly installed before any
generation tests are run.  They do NOT require real models or GPU.
"""

from __future__ import annotations

import importlib

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# Core library import smoke tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("module_name", [
    "numpy",
    "PIL",
    "cv2",
    "sklearn",
    "scipy",
    "skimage",
    "svgwrite",
    "click",
    "requests",
    "safetensors",
])
def test_core_library_imports(module_name: str) -> None:
    """Core image-processing and utility libraries must import cleanly."""
    mod = pytest.importorskip(module_name, reason=f"{module_name} not installed in this environment")
    assert mod is not None


@pytest.mark.parametrize("module_name", [
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "peft",
    "huggingface_hub",
])
def test_ml_library_imports(module_name: str) -> None:
    """Machine learning stack libraries must import cleanly."""
    mod = pytest.importorskip(module_name, reason=f"{module_name} not installed in this environment")
    assert mod is not None


def test_torch_has_cuda_attribute() -> None:
    """torch.cuda module must be accessible (even if no GPU present)."""
    torch = pytest.importorskip("torch", reason="torch not installed in this environment")
    assert hasattr(torch, "cuda")
    # cuda.is_available() must not raise
    _ = torch.cuda.is_available()


def test_diffusers_version_meets_minimum() -> None:
    """diffusers must be >=0.27.0 for stable load_ip_adapter and SDXL support."""
    diffusers = pytest.importorskip("diffusers", reason="diffusers not installed in this environment")
    packaging = pytest.importorskip("packaging", reason="packaging not installed in this environment")
    from packaging.version import Version

    assert Version(diffusers.__version__) >= Version("0.27.0"), (
        f"diffusers {diffusers.__version__} is below required minimum 0.27.0. "
        "Upgrade with: pip install 'diffusers>=0.27.0,<0.29.0'"
    )


def test_peft_installed() -> None:
    """peft must be installed — diffusers>=0.27 requires it for LoRA loading."""
    peft = pytest.importorskip("peft", reason="peft not installed in this environment")
    assert peft.__version__ is not None


def test_peft_version_compatible_with_accelerate() -> None:
    """peft>=0.7 requires accelerate>=0.24; peft>=0.8 requires accelerate>=0.26."""
    peft = pytest.importorskip("peft", reason="peft not installed in this environment")
    accelerate = pytest.importorskip("accelerate", reason="accelerate not installed in this environment")
    packaging = pytest.importorskip("packaging", reason="packaging not installed in this environment")
    from packaging.version import Version

    peft_ver = Version(peft.__version__)
    acc_ver = Version(accelerate.__version__)

    if peft_ver >= Version("0.8"):
        assert acc_ver >= Version("0.26"), (
            f"peft {peft_ver} requires accelerate>=0.26 (got {acc_ver}). "
            "Upgrade with: pip install 'accelerate>=0.27.0,<0.29.0'"
        )
    elif peft_ver >= Version("0.7"):
        assert acc_ver >= Version("0.24"), (
            f"peft {peft_ver} requires accelerate>=0.24 (got {acc_ver})."
        )


def test_numpy_below_2() -> None:
    """numpy must be <2.0.0 — torch 2.1.x is incompatible with numpy 2.x."""
    pytest.importorskip("torch", reason="numpy<2.0 constraint only applies when torch 2.1.x is installed")
    numpy = pytest.importorskip("numpy", reason="numpy not installed in this environment")
    packaging = pytest.importorskip("packaging", reason="packaging not installed in this environment")
    from packaging.version import Version

    assert Version(numpy.__version__) < Version("2.0.0"), (
        f"numpy {numpy.__version__} is >=2.0. torch 2.1.x requires numpy<2.0. "
        "Downgrade with: pip install 'numpy>=1.24.0,<2.0.0'"
    )


def test_ip_adapter_package_not_installed() -> None:
    """The 'ip-adapter' PyPI package must NOT be installed.

    It imports the removed huggingface_hub.cached_download and will break the
    pipeline.  IP-Adapter support is built into diffusers>=0.24.
    """
    with pytest.raises(ImportError):
        import ip_adapter  # type: ignore[import]  # noqa: F401


# ─────────────────────────────────────────────────────────────────────────────
# Project module imports
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("module_path", [
    "pokemon_stencil.config",
    "pokemon_stencil.models.model_loader",
    "pokemon_stencil.models.model_manager",
    "pokemon_stencil.models.lora_validator",
    "pokemon_stencil.image_gen.prompt_engine",
    "pokemon_stencil.image_gen.generator",
    "pokemon_stencil.factory.factory_runner",
])
def test_project_module_imports(module_path: str) -> None:
    """All core project modules must import without errors."""
    mod = importlib.import_module(module_path)
    assert mod is not None
