"""
Batch factory runner for multi-Pokemon, multi-design generation.

``BatchRunner`` consumes a list of ``BatchEntry`` dicts (loaded from the
JSON config file by the CLI) and drives ``Pipeline.run()`` for every
(Pokemon, design-index) combination.

Concurrency model
-----------------
- ``workers=1`` (default): sequential execution in the calling process.
  Recommended on 16 GB RAM laptops because a single SD pipeline already
  occupies ~4–6 GB.
- ``workers>1``: each worker is a separate *process* (not a thread)
  spawned via ``multiprocessing.Pool``.  Each worker loads its own copy
  of the SD pipeline, so RAM usage scales linearly with worker count.
  Only use >1 workers when ``skip_generation=True`` or with abundant RAM.

Error handling
--------------
Failures are caught per-design; a failed design is logged and counted but
does not stop the batch.  The caller (CLI) exits with a non-zero code if
``summary.failed > 0``.
"""

from __future__ import annotations

import logging
import multiprocessing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from pokemon_stencil.config import PipelineConfig
from pokemon_stencil.pipeline import Pipeline, PipelineResult

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BatchEntry:
    """One row in a batch config file."""

    name: str
    """Pokemon name, e.g. ``"Pikachu"``."""

    refs: Optional[Path] = None
    """Optional reference image directory."""

    prompt_extra: str = ""
    """Additional generation prompt text for this Pokemon."""

    @classmethod
    def from_dict(cls, d: Dict) -> "BatchEntry":
        """Construct from the raw JSON dict loaded by the CLI."""
        return cls(
            name=d["name"],
            refs=Path(d["refs"]) if "refs" in d else None,
            prompt_extra=d.get("prompt_extra", ""),
        )


@dataclass
class BatchSummary:
    """Outcome summary returned by :meth:`BatchRunner.run`."""

    total: int = 0
    succeeded: int = 0
    failed: int = 0
    results: List[PipelineResult] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Worker helper (module-level so it is picklable for multiprocessing)
# ─────────────────────────────────────────────────────────────────────────────

def _run_single_design(args: tuple) -> Optional[PipelineResult]:
    """
    Top-level worker function executed in a subprocess.

    Must be module-level (not a closure or method) so that Python's
    ``multiprocessing`` can pickle it.

    Args:
        args: Tuple of (serialised_config_dict, entry_dict, run_name).
              Config is passed as plain Python dicts to avoid pickling
              dataclass issues across Python versions.

    Returns:
        ``PipelineResult`` on success, ``None`` on failure (the worker
        logs the exception before returning).
    """
    import logging as _logging

    config_dict, entry_dict, run_name = args

    # Re-initialise logging in the child process (handlers are not inherited).
    _logging.basicConfig(
        level=_logging.INFO,
        format="%(asctime)s [%(levelname)-8s] %(name)s – %(message)s",
    )
    _log = _logging.getLogger(__name__)

    entry = BatchEntry.from_dict(entry_dict)
    config = _rebuild_config(config_dict)

    pipeline = Pipeline(config)
    try:
        result = pipeline.run(
            pokemon_name=entry.name,
            run_name=run_name,
            reference_dir=entry.refs,
            prompt_extra=entry.prompt_extra,
        )
        _log.info("Worker completed: %s", run_name)
        return result
    except Exception as exc:
        _log.error("Worker failed for %s: %s", run_name, exc, exc_info=True)
        return None


def _rebuild_config(d: dict) -> PipelineConfig:
    """
    Reconstruct a ``PipelineConfig`` from a plain-dict representation.

    This is used by the worker function because ``PipelineConfig`` dataclasses
    may not pickle cleanly across all platforms; shipping raw dicts is safer.
    """
    from pokemon_stencil.config import (
        BatchConfig,
        GenerationConfig,
        OutputConfig,
        ProcessingConfig,
        StencilConfig,
        VectorConfig,
    )

    cfg = PipelineConfig(
        generation=GenerationConfig(**d.get("generation", {})),
        processing=ProcessingConfig(**d.get("processing", {})),
        vector=VectorConfig(**d.get("vector", {})),
        stencil=StencilConfig(**d.get("stencil", {})),
        output=OutputConfig(base_dir=Path(d.get("output_base_dir", "outputs"))),
        batch=BatchConfig(**d.get("batch", {})),
        skip_generation=d.get("skip_generation", False),
    )
    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# BatchRunner
# ─────────────────────────────────────────────────────────────────────────────

class BatchRunner:
    """
    Drives multi-Pokemon, multi-design stencil generation.

    Args:
        config: ``PipelineConfig`` shared across all batch entries.
                ``config.batch.count`` and ``config.batch.workers`` control
                the batch size and parallelism.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, entries: List[Dict]) -> BatchSummary:
        """
        Execute the full batch.

        Args:
            entries: List of raw JSON dicts from the batch config file.
                     Each must have a ``"name"`` key; ``"refs"`` is optional.

        Returns:
            ``BatchSummary`` with success/failure counts and result paths.
        """
        batch_entries = [BatchEntry.from_dict(e) for e in entries]
        count = self.config.batch.count
        workers = self.config.batch.workers

        # Build the flat list of (entry, run_name) jobs.
        jobs = []
        for entry in batch_entries:
            for design_idx in range(count):
                run_name = self._make_run_name(entry.name, design_idx)
                jobs.append((entry, run_name))

        total = len(jobs)
        logger.info(
            "Batch starting: %d job(s) | workers=%d | skip_generation=%s",
            total,
            workers,
            self.config.skip_generation,
        )

        summary = BatchSummary(total=total)

        if workers == 1:
            self._run_sequential(jobs, summary)
        else:
            self._run_parallel(jobs, summary, workers)

        logger.info(
            "Batch done: %d succeeded, %d failed.",
            summary.succeeded,
            summary.failed,
        )
        return summary

    # ── Execution strategies ──────────────────────────────────────────────────

    def _run_sequential(
        self,
        jobs: List[tuple],
        summary: BatchSummary,
    ) -> None:
        """Execute all jobs in the current process, one at a time."""
        pipeline = Pipeline(self.config)
        for job_idx, (entry, run_name) in enumerate(jobs, start=1):
            logger.info(
                "Job %d/%d: %s → %s",
                job_idx,
                summary.total,
                entry.name,
                run_name,
            )
            try:
                result = pipeline.run(
                    pokemon_name=entry.name,
                    run_name=run_name,
                    reference_dir=entry.refs,
                    prompt_extra=entry.prompt_extra,
                )
                summary.results.append(result)
                summary.succeeded += 1
            except Exception as exc:
                msg = f"{run_name}: {exc}"
                logger.error("Job failed – %s", msg, exc_info=True)
                summary.errors.append(msg)
                summary.failed += 1

    def _run_parallel(
        self,
        jobs: List[tuple],
        summary: BatchSummary,
        workers: int,
    ) -> None:
        """Execute jobs across *workers* sub-processes using a Pool."""
        config_dict = self._serialise_config()

        # Build picklable argument tuples for the worker function.
        worker_args = [
            (config_dict, self._serialise_entry(entry), run_name)
            for entry, run_name in jobs
        ]

        ctx = multiprocessing.get_context("spawn")
        effective_workers = min(workers, len(jobs))
        logger.info("Spawning %d worker process(es).", effective_workers)

        with ctx.Pool(processes=effective_workers) as pool:
            results = pool.map(_run_single_design, worker_args)

        for (entry, run_name), result in zip(jobs, results):
            if result is not None:
                summary.results.append(result)
                summary.succeeded += 1
            else:
                msg = f"{run_name}: worker returned None (see worker log)"
                summary.errors.append(msg)
                summary.failed += 1

    # ── Serialisation helpers ─────────────────────────────────────────────────

    def _serialise_config(self) -> dict:
        """Convert PipelineConfig to a plain dict for cross-process passing."""
        cfg = self.config
        return {
            "generation": {
                "num_inference_steps": cfg.generation.num_inference_steps,
                "guidance_scale": cfg.generation.guidance_scale,
                "width": cfg.generation.width,
                "height": cfg.generation.height,
                "seed": cfg.generation.seed,
                "torch_dtype": cfg.generation.torch_dtype,
                "device": cfg.generation.device,
                "style_suffix": cfg.generation.style_suffix,
                "negative_prompt": cfg.generation.negative_prompt,
            },
            "processing": {
                "bilateral_d": cfg.processing.bilateral_d,
                "bilateral_sigma_color": cfg.processing.bilateral_sigma_color,
                "bilateral_sigma_space": cfg.processing.bilateral_sigma_space,
                "bilateral_passes": cfg.processing.bilateral_passes,
                "n_colors": cfg.processing.n_colors,
                "min_region_area": cfg.processing.min_region_area,
                "morph_kernel_size": cfg.processing.morph_kernel_size,
                "canny_low": cfg.processing.canny_low,
                "canny_high": cfg.processing.canny_high,
                "output_size": cfg.processing.output_size,
            },
            "vector": {
                "turdsize": cfg.vector.turdsize,
                "alphamax": cfg.vector.alphamax,
                "opttolerance": cfg.vector.opttolerance,
                "min_path_length": cfg.vector.min_path_length,
                "canvas_width_mm": cfg.vector.canvas_width_mm,
                "canvas_height_mm": cfg.vector.canvas_height_mm,
                "dpi": cfg.vector.dpi,
            },
            "stencil": {
                "bridge_width": cfg.stencil.bridge_width,
                "outline_stroke_width": cfg.stencil.outline_stroke_width,
                "min_cut_width_mm": cfg.stencil.min_cut_width_mm,
                "add_registration_marks": cfg.stencil.add_registration_marks,
                "reg_mark_radius_mm": cfg.stencil.reg_mark_radius_mm,
                "padding_mm": cfg.stencil.padding_mm,
            },
            "batch": {
                "count": cfg.batch.count,
                "workers": cfg.batch.workers,
            },
            "output_base_dir": str(cfg.output.base_dir),
            "skip_generation": cfg.skip_generation,
        }

    @staticmethod
    def _serialise_entry(entry: BatchEntry) -> dict:
        """Convert a BatchEntry to a plain dict for cross-process passing."""
        return {
            "name": entry.name,
            "refs": str(entry.refs) if entry.refs else None,
            "prompt_extra": entry.prompt_extra,
        }

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _make_run_name(pokemon_name: str, design_index: int) -> str:
        """Generate a filesystem-safe run name, e.g. ``pikachu_00``."""
        safe = pokemon_name.lower().replace(" ", "_")
        return f"{safe}_{design_index:02d}"
