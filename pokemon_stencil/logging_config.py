"""
Logging configuration for the Pokemon Stencil Art Factory.

Call ``setup_logging()`` once at process start (the CLI does this
automatically).  All internal modules obtain their logger via::

    import logging
    logger = logging.getLogger(__name__)

Log levels
----------
- ``INFO``  – high-level stage progress; default for normal use.
- ``DEBUG`` – per-image and per-layer detail; enabled with ``--verbose``.
- ``WARNING`` / ``ERROR`` – always shown.

A rotating file handler is added automatically so full debug logs are
always available at ``outputs/pokemon_stencil.log`` regardless of the
console verbosity level.
"""

from __future__ import annotations

import logging
import logging.handlers
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────────────────────

_LOG_FORMAT_VERBOSE = (
    "%(asctime)s [%(levelname)-8s] %(name)s:%(lineno)d – %(message)s"
)
_LOG_FORMAT_NORMAL = "%(asctime)s [%(levelname)-8s] %(message)s"
_DATE_FORMAT = "%H:%M:%S"

_DEFAULT_LOG_FILE = Path("outputs") / "pokemon_stencil.log"
_MAX_LOG_BYTES = 5 * 1024 * 1024   # 5 MB per file
_BACKUP_COUNT = 3


# ── Public API ────────────────────────────────────────────────────────────────

def setup_logging(
    verbose: bool = False,
    log_file: Path | None = None,
) -> None:
    """
    Configure the root logger for the application.

    Args:
        verbose: When ``True`` the console handler emits ``DEBUG`` messages
                 and uses a more detailed format with module name and line
                 number.  When ``False`` only ``INFO`` and above are shown
                 on the console.
        log_file: Path to the rotating log file.  Defaults to
                  ``outputs/pokemon_stencil.log``.  Pass ``None`` to
                  disable file logging.
    """
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)  # capture everything; handlers filter

    # Remove any handlers already attached (e.g. from a previous call or
    # from a test that imported logging before us).
    root.handlers.clear()

    # ── Console handler ───────────────────────────────────────────────────────
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    fmt = _LOG_FORMAT_VERBOSE if verbose else _LOG_FORMAT_NORMAL
    console.setFormatter(logging.Formatter(fmt, datefmt=_DATE_FORMAT))
    root.addHandler(console)

    # ── Rotating file handler ─────────────────────────────────────────────────
    effective_log_file = log_file if log_file is not None else _DEFAULT_LOG_FILE
    try:
        effective_log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            effective_log_file,
            maxBytes=_MAX_LOG_BYTES,
            backupCount=_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(_LOG_FORMAT_VERBOSE, datefmt=_DATE_FORMAT)
        )
        root.addHandler(file_handler)
    except OSError as exc:
        logging.getLogger(__name__).warning(
            "Could not open log file %s: %s", effective_log_file, exc
        )

    # Quieten overly chatty third-party loggers that pollute output.
    for noisy in ("PIL", "urllib3", "filelock", "diffusers", "transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    logging.getLogger(__name__).debug(
        "Logging initialised (verbose=%s, file=%s)", verbose, effective_log_file
    )
