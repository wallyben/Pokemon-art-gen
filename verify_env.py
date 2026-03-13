"""
verify_env.py — Pokemon Stencil Art Factory environment verification.

Imports and prints versions for all critical dependencies, then confirms the
full stack is valid.  Run after install_env.ps1 to verify the environment
before generating images.

Usage::

    python verify_env.py

Exit codes:
    0 — all checks pass
    1 — one or more checks failed
"""

from __future__ import annotations

import sys

_PASS = "  ✓"
_FAIL = "  ✗"
_errors: list[str] = []


def _check(label: str, import_expr: str, version_attr: str = "__version__") -> None:
    """Import a module and print its version, recording failures."""
    try:
        mod = __import__(import_expr.split(".")[0])
        for part in import_expr.split(".")[1:]:
            mod = getattr(mod, part)
        ver = getattr(mod, version_attr, "unknown")
        print(f"{_PASS}  {label:25s} {ver}")
    except Exception as exc:
        print(f"{_FAIL}  {label:25s} FAILED — {exc}")
        _errors.append(f"{label}: {exc}")


def _check_torch_cuda() -> None:
    """Check whether CUDA is available for torch."""
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            print(f"{_PASS}  {'torch CUDA':25s} available ({device_name})")
        else:
            print(f"  ℹ  {'torch CUDA':25s} not available (CPU mode)")
    except Exception as exc:
        print(f"  ℹ  {'torch CUDA':25s} check failed: {exc}")


def _check_peft_accelerate_compat() -> None:
    """Verify PEFT and accelerate are mutually compatible."""
    try:
        import peft  # noqa: F401
        import accelerate  # noqa: F401
        # peft 0.7+ requires accelerate>=0.24; peft 0.8+ requires accelerate>=0.26
        from packaging.version import Version
        peft_ver = Version(peft.__version__)
        acc_ver = Version(accelerate.__version__)
        if peft_ver >= Version("0.8") and acc_ver < Version("0.26"):
            _errors.append(
                f"peft {peft_ver} requires accelerate>=0.26, got {acc_ver}"
            )
            print(f"{_FAIL}  {'peft/accelerate compat':25s} INCOMPATIBLE — "
                  f"peft {peft_ver} needs accelerate>=0.26 (got {acc_ver})")
        else:
            print(f"{_PASS}  {'peft/accelerate compat':25s} OK "
                  f"(peft={peft_ver}, accel={acc_ver})")
    except ImportError:
        # packaging may not be installed; skip compat check
        print(f"  ℹ  {'peft/accelerate compat':25s} skipped (packaging not installed)")
    except Exception as exc:
        print(f"  ℹ  {'peft/accelerate compat':25s} check failed: {exc}")


def _check_no_ip_adapter_package() -> None:
    """Verify the broken ip-adapter PyPI package is NOT installed."""
    try:
        import ip_adapter  # noqa: F401
        # If we get here, the package is installed — this is a problem
        _errors.append(
            "The 'ip-adapter' PyPI package is installed. "
            "It imports the removed huggingface_hub.cached_download and will "
            "break the pipeline. Uninstall with: pip uninstall ip-adapter"
        )
        print(
            f"{_FAIL}  {'ip-adapter pkg absent':25s} INSTALLED (must uninstall: "
            "pip uninstall ip-adapter)"
        )
    except ImportError:
        print(f"{_PASS}  {'ip-adapter pkg absent':25s} not installed (correct)")


def _check_no_pypotrace() -> None:
    """Verify the Windows-incompatible pypotrace is NOT a hard requirement."""
    try:
        import potrace  # noqa: F401
        print(f"  ℹ  {'pypotrace':25s} installed (optional, not required for SVG export)")
    except ImportError:
        print(f"{_PASS}  {'pypotrace absent':25s} not installed (correct — OpenCV fallback active)")


def _check_fal_provider() -> None:
    """Check whether the fal.ai cloud provider is configured."""
    import os
    try:
        import fal_client  # noqa: F401
        fal_installed = True
    except ImportError:
        fal_installed = False

    fal_key = bool(os.environ.get("FAL_KEY"))

    if fal_installed and fal_key:
        print(f"{_PASS}  {'fal.ai provider':25s} ready (fal-client installed + FAL_KEY set)")
    elif fal_installed and not fal_key:
        print(f"  ℹ  {'fal.ai provider':25s} fal-client installed but FAL_KEY not set")
        print(f"       Set with: $env:FAL_KEY = 'your-key'  (get at https://fal.ai/dashboard/keys)")
    elif not fal_installed:
        print(f"  ℹ  {'fal.ai provider':25s} not installed (pip install fal-client)")


def _check_project_imports() -> None:
    """Verify core project modules import cleanly."""
    modules = [
        ("config", "pokemon_stencil.config"),
        ("model_loader", "pokemon_stencil.models.model_loader"),
        ("model_manager", "pokemon_stencil.models.model_manager"),
        ("lora_validator", "pokemon_stencil.models.lora_validator"),
        ("prompt_engine", "pokemon_stencil.image_gen.prompt_engine"),
        ("generator", "pokemon_stencil.image_gen.generator"),
        ("generation.base", "pokemon_stencil.generation.base"),
        ("generation.prompt_builder", "pokemon_stencil.generation.prompt_builder"),
        ("generation.candidate_runner", "pokemon_stencil.generation.candidate_runner"),
    ]
    for label, mod_path in modules:
        try:
            parts = mod_path.split(".")
            mod = __import__(mod_path)
            for part in parts[1:]:
                mod = getattr(mod, part)
            print(f"{_PASS}  {label:25s} OK")
        except Exception as exc:
            print(f"{_FAIL}  {label:25s} FAILED — {exc}")
            _errors.append(f"project import {label}: {exc}")


def main() -> int:
    print()
    print("=" * 60)
    print("  Pokemon Stencil Art Factory — Environment Verification")
    print("=" * 60)
    print(f"\nPython {sys.version}\n")

    print("── Core dependencies ──────────────────────────────────────")
    _check("torch", "torch")
    _check("torchvision", "torchvision")
    _check_torch_cuda()
    _check("diffusers", "diffusers")
    _check("transformers", "transformers")
    _check("accelerate", "accelerate")
    _check("peft", "peft")
    _check("safetensors", "safetensors")
    _check("huggingface_hub", "huggingface_hub")

    print("\n── Image processing ───────────────────────────────────────")
    _check("numpy", "numpy")
    _check("Pillow", "PIL", "PILLOW_VERSION" if hasattr(__import__("PIL", fromlist=["PIL"]), "PILLOW_VERSION") else "__version__")
    _check("opencv", "cv2", "__version__")
    _check("scikit-image", "skimage", "__version__")
    _check("scikit-learn", "sklearn", "__version__")
    _check("scipy", "scipy")

    print("\n── ControlNet & misc ──────────────────────────────────────")
    _check("controlnet-aux", "controlnet_aux")
    _check("streamlit", "streamlit")
    _check("click", "click")
    _check("svgwrite", "svgwrite")
    _check("requests", "requests")

    print("\n── Cloud generation provider ──────────────────────────────")
    _check_fal_provider()

    print("\n── Compatibility checks ───────────────────────────────────")
    _check_peft_accelerate_compat()
    _check_no_ip_adapter_package()
    _check_no_pypotrace()

    print("\n── Project imports ────────────────────────────────────────")
    _check_project_imports()

    print()
    if _errors:
        print("=" * 60)
        print(f"  ✗ {len(_errors)} issue(s) found:")
        for e in _errors:
            print(f"    • {e}")
        print("=" * 60)
        print()
        return 1
    else:
        print("=" * 60)
        print("  ✓ All checks passed — environment is valid.")
        print("=" * 60)
        print()
        return 0


if __name__ == "__main__":
    sys.exit(main())
