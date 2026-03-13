"""
scan_loras.py — LoRA directory scanner for the Pokémon Stencil Art Factory.

Scans ``models/lora/`` and classifies every file found:

- ✅  SDXL-compatible ``.safetensors`` LoRA → safe to use
- ❌  SD1.5 / incompatible ``.safetensors`` LoRA → will be blocked by the pipeline
- ⚠️   Non-``.safetensors`` files at top level (unexpected)
- 🗑   Cache residue: ``.cache/``, ``.lock``, ``.meta`` files (safe to delete)

Usage::

    python scan_loras.py
    python scan_loras.py --lora-dir path/to/lora/dir

Exit codes:
    0 — at least one SDXL-compatible LoRA found (or no files at all)
    1 — only incompatible / broken LoRAs found, or validation errors
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ── ANSI colour helpers ────────────────────────────────────────────────────────

def _green(s: str) -> str:
    return f"\033[32m{s}\033[0m"

def _red(s: str) -> str:
    return f"\033[31m{s}\033[0m"

def _yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m"

def _dim(s: str) -> str:
    return f"\033[2m{s}\033[0m"


# ── Scanner ────────────────────────────────────────────────────────────────────

def scan(lora_dir: Path) -> int:
    """
    Scan *lora_dir* and print a classification report.

    Returns:
        0 if at least one SDXL-compatible LoRA exists.
        1 if only incompatible LoRAs or no valid LoRAs found.
    """
    print()
    print("=" * 70)
    print("  Pokémon Stencil Art Factory — LoRA Directory Scanner")
    print("=" * 70)
    print(f"\n  LoRA directory: {lora_dir.resolve()}\n")

    if not lora_dir.exists():
        print(_yellow(f"  ⚠️  Directory does not exist: {lora_dir}"))
        print(_dim("     Create it with: mkdir -p models/lora"))
        print()
        return 0  # Not an error — just no LoRAs yet

    # ── Check for safetensors package ─────────────────────────────────────────
    try:
        from pokemon_stencil.models.lora_validator import validate_lora_sdxl_compatible
        validator_available = True
    except ImportError:
        print(_yellow(
            "  ⚠️  lora_validator not available. "
            "Install project package to enable validation."
        ))
        validator_available = False

    # ── Classify all items ────────────────────────────────────────────────────
    sdxl_loras: list[Path] = []
    sd15_loras: list[Path] = []
    unknown_loras: list[Path] = []
    non_safetensors: list[Path] = []
    cache_residue: list[Path] = []

    # Top-level files
    for item in sorted(lora_dir.iterdir()):
        if item.is_dir():
            # Subdirectories are cache residue (e.g. .cache/)
            cache_files = list(item.rglob("*"))
            if cache_files:
                cache_residue.append(item)
            continue

        if item.suffix.lower() == ".safetensors":
            if validator_available:
                ok, msg = validate_lora_sdxl_compatible(item)
                if ok:
                    sdxl_loras.append((item, msg))
                else:
                    sd15_loras.append((item, msg))
            else:
                unknown_loras.append(item)
        elif item.suffix.lower() in (".lock", ".meta", ".incomplete"):
            cache_residue.append(item)
        else:
            non_safetensors.append(item)

    # ── SDXL-compatible ───────────────────────────────────────────────────────
    print("  ── SDXL-compatible LoRAs (safe to use) " + "─" * 28)
    if sdxl_loras:
        for path, msg in sdxl_loras:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(_green(f"  ✅  {path.name}") + _dim(f"  ({size_mb:.1f} MB)"))
            print(_dim(f"       {msg}"))
    else:
        print(_yellow("  (none found)"))
        print(_dim(
            "  To add an SDXL LoRA, place a .safetensors file in models/lora/.\n"
            "  Recommended: civitai.com → filter 'LoRA' + 'SDXL 1.0' → download."
        ))

    # ── Incompatible ─────────────────────────────────────────────────────────
    print()
    print("  ── Incompatible / SD1.5 LoRAs (blocked by pipeline) " + "─" * 14)
    if sd15_loras:
        for path, msg in sd15_loras:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(_red(f"  ❌  {path.name}") + _dim(f"  ({size_mb:.1f} MB)"))
            print(_dim(f"       {msg}"))
        print()
        print(_yellow(
            "  ACTION: Remove or replace these files with SDXL-compatible LoRAs.\n"
            "  They will be blocked by the pipeline validator and cannot be used."
        ))
    else:
        print(_dim("  (none)"))

    # ── Unknown (no validator) ────────────────────────────────────────────────
    if unknown_loras:
        print()
        print("  ── Unknown (validator unavailable) " + "─" * 32)
        for path in unknown_loras:
            size_mb = path.stat().st_size / (1024 * 1024)
            print(_yellow(f"  ⚠️   {path.name}") + _dim(f"  ({size_mb:.1f} MB)"))

    # ── Unexpected non-safetensors ────────────────────────────────────────────
    if non_safetensors:
        print()
        print("  ── Unexpected files (non-.safetensors) " + "─" * 27)
        for path in non_safetensors:
            print(_yellow(f"  ⚠️   {path.name}"))
        print(_dim("  These files will be ignored by the pipeline."))

    # ── Cache residue ─────────────────────────────────────────────────────────
    print()
    print("  ── Cache residue (safe to delete) " + "─" * 32)
    if cache_residue:
        for item in cache_residue:
            if item.is_dir():
                file_count = sum(1 for _ in item.rglob("*") if _.is_file())
                print(_dim(f"  🗑   {item.name}/  ({file_count} file(s))"))
            else:
                print(_dim(f"  🗑   {item.name}"))
        print()
        print(_yellow(
            "  ACTION: Delete cache residue to clean up disk space.\n"
            "  These are incomplete downloads — they do NOT affect the pipeline."
        ))
        _print_cleanup_commands(lora_dir, cache_residue)
    else:
        print(_dim("  (none — directory is clean)"))

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  Summary")
    print("=" * 70)
    print(f"  SDXL-compatible LoRAs :  {_green(str(len(sdxl_loras)))}")
    print(f"  Incompatible LoRAs    :  {_red(str(len(sd15_loras))) if sd15_loras else '0'}")
    print(f"  Cache residue items   :  {len(cache_residue)}")

    if sdxl_loras:
        print()
        print(_green("  ✅  System ready — SDXL generation will use the LoRA(s) above."))
    elif not sd15_loras and not unknown_loras:
        print()
        print(_yellow("  ⚠️  No LoRAs installed — generation will run in pure SDXL mode."))
        print(_dim("     This works, but a Pokémon-specific SDXL LoRA gives better results."))
    else:
        print()
        print(_red("  ❌  No SDXL-compatible LoRAs found."))
        if sd15_loras:
            print(_yellow("     Remove the SD1.5 LoRA(s) and replace with SDXL-compatible ones."))
    print()

    return 0 if (sdxl_loras or (not sd15_loras and not unknown_loras)) else 1


def _print_cleanup_commands(lora_dir: Path, residue: list[Path]) -> None:
    """Print cleanup commands for cache residue."""
    print()
    print(_dim("  Cleanup commands (Windows PowerShell):"))
    for item in residue:
        full = (lora_dir / item.name).resolve()
        if item.is_dir():
            print(_dim(f"    Remove-Item -Recurse -Force \"{full}\""))
        else:
            print(_dim(f"    Remove-Item -Force \"{full}\""))

    print()
    print(_dim("  Cleanup commands (Unix / WSL):"))
    for item in residue:
        full = (lora_dir / item.name).resolve()
        if item.is_dir():
            print(_dim(f"    rm -rf \"{full}\""))
        else:
            print(_dim(f"    rm \"{full}\""))


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan models/lora/ and classify LoRA files for SDXL compatibility."
    )
    parser.add_argument(
        "--lora-dir",
        default="models/lora",
        help="Path to the LoRA directory (default: models/lora)",
    )
    args = parser.parse_args()

    lora_dir = Path(args.lora_dir)
    sys.exit(scan(lora_dir))


if __name__ == "__main__":
    main()
