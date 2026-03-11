"""
CLI entry point for the Pokemon Stencil Art Factory.

Installed as the ``pokemon-stencil`` console script via ``setup.py``.

Commands
--------
generate   Generate artwork and stencil SVGs for a single Pokemon.
process    Convert an existing image directly into stencil SVGs.
batch      Run factory generation over multiple Pokemon from a JSON config.
inspect    Visualise colour segmentation for debugging.

Global options (available on every command)
-------------------------------------------
--output-dir PATH   Root directory for all outputs  [default: outputs]
--n-colors  INT     Number of colour layers          [default: 6]
--steps     INT     SD inference steps               [default: 25]
--seed      INT     Global RNG seed
--workers   INT     Parallel worker processes for batch  [default: 1]
--verbose           Enable DEBUG-level logging

Usage examples
--------------
# Generate 3 Pikachu stencils from scratch using Stable Diffusion
pokemon-stencil generate Pikachu --refs refs/pikachu/ --count 3

# Convert an existing image directly to SVG stencils
pokemon-stencil process art/charizard.png --name charizard --n-colors 5

# Batch-generate from a JSON config, 3 designs per Pokemon, 2 workers
pokemon-stencil batch configs/pokemon_list.json --count 3 --workers 2

# Debug: inspect segmentation layers in a simplified image
pokemon-stencil inspect outputs/pikachu_00/simplified/simplified_000.png
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from pokemon_stencil.config import (
    BatchConfig,
    GenerationConfig,
    OutputConfig,
    PipelineConfig,
    ProcessingConfig,
)
from pokemon_stencil.logging_config import setup_logging

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared option decorators
# ─────────────────────────────────────────────────────────────────────────────

def _common_options(func):
    """Decorator that attaches global options to any command."""
    options = [
        click.option(
            "--output-dir",
            default="outputs",
            show_default=True,
            type=click.Path(file_okay=False, writable=True),
            help="Root directory for all generated outputs.",
        ),
        click.option(
            "--n-colors",
            default=6,
            show_default=True,
            type=click.IntRange(2, 16),
            help="Number of colour layers (K-Means clusters).",
        ),
        click.option(
            "--steps",
            default=25,
            show_default=True,
            type=click.IntRange(1, 150),
            help="Stable Diffusion inference steps (more = better quality, slower).",
        ),
        click.option(
            "--seed",
            default=None,
            type=int,
            help="Fixed RNG seed for reproducible results.",
        ),
        click.option(
            "--verbose",
            is_flag=True,
            default=False,
            help="Enable DEBUG-level logging.",
        ),
    ]
    for option in reversed(options):
        func = option(func)
    return func


# ─────────────────────────────────────────────────────────────────────────────
# CLI group
# ─────────────────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="1.0.0", prog_name="pokemon-stencil")
def main() -> None:
    """Pokemon Stencil Art Factory – generate Cricut-ready stencil SVGs."""


# ─────────────────────────────────────────────────────────────────────────────
# generate command
# ─────────────────────────────────────────────────────────────────────────────

@main.command("generate")
@click.argument("pokemon_name", metavar="POKEMON_NAME")
@click.option(
    "--refs",
    default=None,
    type=click.Path(exists=True, file_okay=False),
    help="Directory of reference images for this Pokemon.",
)
@click.option(
    "--count",
    default=1,
    show_default=True,
    type=click.IntRange(1, 50),
    help="Number of designs to generate.",
)
@click.option(
    "--skip-generation",
    is_flag=True,
    default=False,
    help="Use reference images directly; skip Stable Diffusion generation.",
)
@click.option(
    "--prompt-extra",
    default="",
    help="Additional text appended to the generation prompt.",
)
@_common_options
def cmd_generate(
    pokemon_name: str,
    refs: Optional[str],
    count: int,
    skip_generation: bool,
    prompt_extra: str,
    output_dir: str,
    n_colors: int,
    steps: int,
    seed: Optional[int],
    verbose: bool,
) -> None:
    """Generate artwork and stencil SVGs for a single POKEMON_NAME.

    When --refs is provided the reference images inform prompt construction.
    When --skip-generation is set, reference images are used as-is (no SD).

    \b
    Examples:
      pokemon-stencil generate Pikachu --refs refs/pikachu/ --count 3
      pokemon-stencil generate Gengar --skip-generation --refs refs/gengar/
    """
    setup_logging(verbose=verbose)
    logger.info("Command: generate | pokemon=%s | count=%d", pokemon_name, count)

    if skip_generation and refs is None:
        raise click.UsageError(
            "--skip-generation requires --refs to be provided."
        )

    config = _build_pipeline_config(
        output_dir=output_dir,
        n_colors=n_colors,
        steps=steps,
        seed=seed,
        skip_generation=skip_generation,
    )

    from pokemon_stencil.pipeline import Pipeline

    pipeline = Pipeline(config)
    refs_path = Path(refs) if refs else None

    for design_index in range(count):
        run_name = _make_run_name(pokemon_name, design_index)
        logger.info(
            "Design %d/%d → run_name=%s", design_index + 1, count, run_name
        )
        try:
            result = pipeline.run(
                pokemon_name=pokemon_name,
                reference_dir=refs_path,
                run_name=run_name,
                prompt_extra=prompt_extra,
            )
            click.echo(f"  ✓ {run_name} → {result.svg_dir}")
        except Exception as exc:
            logger.error("Design %s failed: %s", run_name, exc, exc_info=verbose)
            click.echo(f"  ✗ {run_name} failed: {exc}", err=True)

    click.echo("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# process command
# ─────────────────────────────────────────────────────────────────────────────

@main.command("process")
@click.argument("image_path", metavar="IMAGE_PATH", type=click.Path(exists=True))
@click.option(
    "--name",
    default=None,
    help="Design name used in output filenames. Defaults to the image stem.",
)
@_common_options
def cmd_process(
    image_path: str,
    name: Optional[str],
    output_dir: str,
    n_colors: int,
    steps: int,
    seed: Optional[int],
    verbose: bool,
) -> None:
    """Convert an existing IMAGE_PATH directly into stencil SVGs.

    Skips Stable Diffusion generation entirely. Useful for converting
    manually created or downloaded artwork.

    \b
    Examples:
      pokemon-stencil process art/charizard.png --name charizard
      pokemon-stencil process refs/bulbasaur/front.png --n-colors 4
    """
    setup_logging(verbose=verbose)
    img = Path(image_path)
    design_name = name or img.stem
    logger.info("Command: process | image=%s | name=%s", img, design_name)

    config = _build_pipeline_config(
        output_dir=output_dir,
        n_colors=n_colors,
        steps=steps,
        seed=seed,
        skip_generation=True,
    )

    from pokemon_stencil.pipeline import Pipeline

    pipeline = Pipeline(config)
    run_name = _make_run_name(design_name, 0)

    try:
        result = pipeline.run_from_image(
            image_path=img,
            design_name=design_name,
            run_name=run_name,
        )
        click.echo(f"✓ {run_name} → {result.svg_dir}")
    except Exception as exc:
        logger.error("Process failed: %s", exc, exc_info=verbose)
        click.echo(f"✗ Failed: {exc}", err=True)
        sys.exit(1)

    click.echo("Done.")


# ─────────────────────────────────────────────────────────────────────────────
# batch command
# ─────────────────────────────────────────────────────────────────────────────

@main.command("batch")
@click.argument(
    "config_file",
    metavar="CONFIG_FILE",
    type=click.Path(exists=True, dir_okay=False),
)
@click.option(
    "--count",
    default=3,
    show_default=True,
    type=click.IntRange(1, 50),
    help="Number of designs to generate per Pokemon entry.",
)
@click.option(
    "--workers",
    default=1,
    show_default=True,
    type=click.IntRange(1, 32),
    help="Number of parallel worker processes.",
)
@click.option(
    "--skip-generation",
    is_flag=True,
    default=False,
    help="Use reference images directly; skip Stable Diffusion generation.",
)
@_common_options
def cmd_batch(
    config_file: str,
    count: int,
    workers: int,
    skip_generation: bool,
    output_dir: str,
    n_colors: int,
    steps: int,
    seed: Optional[int],
    verbose: bool,
) -> None:
    """Run factory generation over multiple Pokemon from CONFIG_FILE.

    CONFIG_FILE must be a JSON array where each element has:

    \b
      { "name": "Pikachu", "refs": "refs/pikachu/" }

    The "refs" field is optional when --skip-generation is not set.

    \b
    Examples:
      pokemon-stencil batch configs/pokemon_list.json --count 3 --workers 2
      pokemon-stencil batch configs/pokemon_list.json --skip-generation
    """
    setup_logging(verbose=verbose)
    cfg_path = Path(config_file)
    logger.info(
        "Command: batch | config=%s | count=%d | workers=%d",
        cfg_path,
        count,
        workers,
    )

    try:
        entries = _load_batch_config(cfg_path)
    except (json.JSONDecodeError, ValueError) as exc:
        click.echo(f"✗ Invalid config file: {exc}", err=True)
        sys.exit(1)

    click.echo(
        f"Batch: {len(entries)} Pokemon × {count} designs "
        f"({workers} worker{'s' if workers > 1 else ''})"
    )

    pipeline_config = _build_pipeline_config(
        output_dir=output_dir,
        n_colors=n_colors,
        steps=steps,
        seed=seed,
        skip_generation=skip_generation,
    )
    pipeline_config.batch.count = count
    pipeline_config.batch.workers = workers

    from pokemon_stencil.batch.batch_runner import BatchRunner

    runner = BatchRunner(pipeline_config)
    summary = runner.run(entries)

    click.echo(
        f"\nBatch complete: {summary.succeeded} succeeded, "
        f"{summary.failed} failed."
    )
    if summary.failed:
        sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# inspect command
# ─────────────────────────────────────────────────────────────────────────────

@main.command("inspect")
@click.argument("image_path", metavar="IMAGE_PATH", type=click.Path(exists=True))
@click.option(
    "--n-colors",
    default=6,
    show_default=True,
    type=click.IntRange(2, 16),
    help="Number of colour clusters to segment.",
)
@click.option(
    "--save",
    default=None,
    type=click.Path(dir_okay=False, writable=True),
    help="Save the visualisation to this path instead of displaying it.",
)
@click.option(
    "--verbose",
    is_flag=True,
    default=False,
    help="Enable DEBUG-level logging.",
)
def cmd_inspect(
    image_path: str,
    n_colors: int,
    save: Optional[str],
    verbose: bool,
) -> None:
    """Visualise colour segmentation layers of IMAGE_PATH for debugging.

    Displays (or saves) a colour-coded image where each pixel is painted
    in the representative colour of its assigned K-Means cluster.

    \b
    Examples:
      pokemon-stencil inspect outputs/pikachu_00/simplified/simplified_000.png
      pokemon-stencil inspect art/eevee.png --n-colors 4 --save /tmp/seg.png
    """
    setup_logging(verbose=verbose)
    img_path = Path(image_path)
    logger.info("Command: inspect | image=%s | n_colors=%d", img_path, n_colors)

    from PIL import Image

    from pokemon_stencil.config import ProcessingConfig
    from pokemon_stencil.image_proc.segmenter import ColourSegmenter
    from pokemon_stencil.image_proc.simplifier import ImageSimplifier

    proc_cfg = ProcessingConfig(n_colors=n_colors)
    simplifier = ImageSimplifier(proc_cfg)
    segmenter = ColourSegmenter(proc_cfg)

    img = Image.open(img_path).convert("RGB")
    simplified = simplifier.simplify(img)
    layers = segmenter.segment(simplified)
    vis = segmenter.visualise(layers, simplified.size)

    click.echo(f"Segmented into {len(layers)} layer(s):")
    for layer in layers:
        hex_colour = "#{:02X}{:02X}{:02X}".format(*layer.colour_rgb)
        click.echo(
            f"  Layer {layer.index:02d}  {hex_colour}  "
            f"({layer.coverage_fraction * 100:.1f}% coverage)"
        )

    if save:
        vis.save(save)
        click.echo(f"Visualisation saved → {save}")
    else:
        vis.show()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_pipeline_config(
    *,
    output_dir: str,
    n_colors: int,
    steps: int,
    seed: Optional[int],
    skip_generation: bool,
) -> PipelineConfig:
    """Construct a PipelineConfig from flat CLI option values."""
    config = PipelineConfig(
        generation=GenerationConfig(
            num_inference_steps=steps,
            seed=seed,
        ),
        processing=ProcessingConfig(n_colors=n_colors),
        output=OutputConfig(base_dir=Path(output_dir)),
        skip_generation=skip_generation,
    )
    return config


def _make_run_name(pokemon_name: str, index: int) -> str:
    """Generate a filesystem-safe run name, e.g. ``pikachu_00``."""
    safe = pokemon_name.lower().replace(" ", "_")
    return f"{safe}_{index:02d}"


def _load_batch_config(path: Path) -> list[dict]:
    """
    Parse a JSON batch config file.

    Expected format::

        [
          { "name": "Pikachu",   "refs": "refs/pikachu/" },
          { "name": "Charizard", "refs": "refs/charizard/" }
        ]

    Args:
        path: Path to the JSON file.

    Returns:
        List of dicts, each with at least a ``"name"`` key.

    Raises:
        json.JSONDecodeError: If the file is not valid JSON.
        ValueError: If the top-level element is not a list, or any entry
                    is missing the required ``"name"`` key.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("Batch config must be a JSON array at the top level.")
    for i, entry in enumerate(raw):
        if "name" not in entry:
            raise ValueError(
                f"Entry {i} in batch config is missing the required 'name' key."
            )
    return raw
