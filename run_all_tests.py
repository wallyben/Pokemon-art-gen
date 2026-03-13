"""
run_all_tests.py — Pokemon Stencil Art Factory full test suite runner.

Runs the complete test suite and prints a clear summary.
Prints "SDXL + LoRA generator environment validated" only when all tests pass.

Usage::

    python run_all_tests.py              # run all tests
    python run_all_tests.py -v           # verbose output
    python run_all_tests.py --fast       # skip slow integration tests
    python run_all_tests.py --lora-only  # run LoRA validation tests only

Exit codes:
    0 — all tests passed
    1 — one or more tests failed or errored
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path


_REPO_ROOT = Path(__file__).parent
_TEST_DIR = _REPO_ROOT / "tests"

# ── Test suites in priority order ─────────────────────────────────────────────
_ALL_SUITES = [
    {
        "name": "Environment imports",
        "path": "tests/test_env_imports.py",
        "description": "Verifies all core libraries import successfully",
    },
    {
        "name": "LoRA validation",
        "path": "tests/test_lora_validation.py",
        "description": "SDXL LoRA accepted, SD1.5 LoRA rejected, missing file handled",
    },
    {
        "name": "SDXL pipeline load",
        "path": "tests/test_sdxl_pipeline_load.py",
        "description": "Pipeline loads with correct arguments and caching",
    },
    {
        "name": "SDXL + LoRA pipeline load",
        "path": "tests/test_sdxl_lora_pipeline_load.py",
        "description": "Pipeline loads with valid SDXL LoRA; SD1.5 LoRA rejected",
    },
    {
        "name": "Generation smoke",
        "path": "tests/test_generation_smoke.py",
        "description": "Single image generated, PIL output returned, no tensor errors",
    },
    {
        "name": "Dashboard defaults",
        "path": "tests/test_dashboard_defaults.py",
        "description": "ControlNet off by default, IP-Adapter default, LoRA config",
    },
    {
        "name": "Prompt optimizer",
        "path": "tests/test_prompt_optimizer.py",
        "description": "Prompt reduction preserves subject, fits CLIP limit",
    },
    {
        "name": "Full generator integration",
        "path": "tests/test_full_generator_integration.py",
        "description": "2–3 candidate mini end-to-end run, no runtime exceptions",
    },
    # ── Legacy suites (maintained from v1) ────────────────────────────────────
    {
        "name": "Config",
        "path": "tests/test_config.py",
        "description": "GenerationConfig, ProcessingConfig, PipelineConfig defaults",
    },
    {
        "name": "Model manager",
        "path": "tests/test_model_manager.py",
        "description": "ModelManager downloads, integrity checks, LoRA listing",
    },
    {
        "name": "Model loader",
        "path": "tests/test_model_loader.py",
        "description": "Pipeline resolution, caching, and cache helpers",
    },
    {
        "name": "Prompt engine",
        "path": "tests/test_prompt_engine.py",
        "description": "PromptEngine CLIP token limit, build(), compress()",
    },
    {
        "name": "Image generator",
        "path": "tests/test_image_generator.py",
        "description": "PokemonImageGenerator generate(), prompt, white-bg",
    },
    {
        "name": "Factory runner",
        "path": "tests/test_factory_runner.py",
        "description": "FactoryRunner candidate generation and scoring",
    },
    {
        "name": "Dashboard config",
        "path": "tests/test_dashboard_config.py",
        "description": "_build_config, _ProgressFactoryRunner",
    },
]

_LORA_ONLY_SUITES = [
    "tests/test_lora_validation.py",
    "tests/test_sdxl_lora_pipeline_load.py",
]

_FAST_SKIP = [
    "tests/test_full_generator_integration.py",
]


def _run_suite(suite: dict, verbose: bool) -> tuple[bool, str]:
    """Run a single test suite. Return (passed, summary_line)."""
    path = _REPO_ROOT / suite["path"]
    if not path.exists():
        return True, f"  ⚪  {suite['name']:35s} SKIPPED (file not found)"

    cmd = [sys.executable, "-m", "pytest", str(path), "-q", "--tb=short"]
    if verbose:
        cmd.append("-v")

    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(_REPO_ROOT),
            capture_output=True,
            text=True,
        )
        elapsed = time.monotonic() - start
        passed = result.returncode == 0
        icon = "  ✅" if passed else "  ❌"
        status = "PASSED" if passed else "FAILED"
        line = f"{icon}  {suite['name']:35s} {status}  ({elapsed:.1f}s)"
        return passed, line, result.stdout + result.stderr
    except Exception as exc:
        return False, f"  ❌  {suite['name']:35s} ERROR: {exc}", ""


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    verbose = "-v" in args or "--verbose" in args
    fast = "--fast" in args
    lora_only = "--lora-only" in args

    if lora_only:
        suites = [s for s in _ALL_SUITES if s["path"] in _LORA_ONLY_SUITES]
    elif fast:
        suites = [s for s in _ALL_SUITES if s["path"] not in _FAST_SKIP]
    else:
        suites = _ALL_SUITES

    print()
    print("=" * 65)
    print("  Pokemon Stencil Art Factory — Full Test Suite")
    print("=" * 65)
    print(f"  Running {len(suites)} suite(s) …\n")

    all_passed = True
    failures: list[tuple[str, str]] = []
    output_blocks: list[tuple[str, str]] = []

    for suite in suites:
        passed, line, output = _run_suite(suite, verbose)
        print(line)
        if not passed:
            all_passed = False
            failures.append((suite["name"], output))
        if verbose and output.strip():
            output_blocks.append((suite["name"], output))

    # Print failure details
    if failures:
        print()
        print("─" * 65)
        print("  FAILURE DETAILS")
        print("─" * 65)
        for name, output in failures:
            print(f"\n  [{name}]")
            for ln in output.strip().splitlines()[-20:]:  # last 20 lines
                print(f"    {ln}")

    if verbose:
        for name, output in output_blocks:
            print(f"\n  [{name} — full output]")
            for ln in output.strip().splitlines():
                print(f"    {ln}")

    print()
    print("=" * 65)
    if all_passed:
        print("  ✅ All tests passed.")
        print()
        print("  SDXL + LoRA generator environment validated")
        print("=" * 65)
        print()
        return 0
    else:
        failed_names = [name for name, _ in failures]
        print(f"  ❌ {len(failures)} suite(s) failed: {', '.join(failed_names)}")
        print("=" * 65)
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
