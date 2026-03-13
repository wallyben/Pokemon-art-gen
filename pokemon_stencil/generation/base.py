"""
pokemon_stencil.generation.base — abstract generation provider contract.

Every generation backend (fal.ai, local SDXL, etc.) implements
``AbstractGenerationProvider``.  Callers work only against this interface,
making it trivial to swap or fall back between providers.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


# ── Request / Result data containers ──────────────────────────────────────────

@dataclass
class GenerationRequest:
    """All parameters needed to generate one batch of candidate images.

    ``reference_images`` are passed to providers that support reference/
    image-to-image conditioning (e.g. fal.ai img2img, IP-Adapter).  Providers
    that cannot use references fall back to text-only generation and log a
    warning.
    """

    prompt: str
    negative_prompt: str = ""

    #: Reference images for character conditioning (order matters — best first).
    reference_images: List[Path] = field(default_factory=list)

    width: int = 1024
    height: int = 1024
    num_images: int = 1

    #: Seed for reproducibility.  None = random.
    seed: Optional[int] = None

    guidance_scale: float = 3.5
    num_steps: int = 28

    #: How strongly to follow reference vs. prompt (0=prompt only, 1=reference only).
    #: Used by img2img / IP-Adapter providers.  Ignored by text-only providers.
    reference_strength: float = 0.80

    #: Arbitrary extra parameters forwarded verbatim to the underlying provider.
    extra: dict = field(default_factory=dict)


@dataclass
class GenerationResult:
    """Output from a single generation call."""

    #: Local paths of generated PNG files (len == request.num_images).
    images: List[Path]

    #: Short identifier of the provider that produced these images.
    provider: str

    #: Provider-specific metadata (model IDs, timing, cost, etc.).
    metadata: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return bool(self.images)


# ── Abstract provider ──────────────────────────────────────────────────────────

class AbstractGenerationProvider(ABC):
    """Base class for all generation backends.

    Implementors must:
    1. Override ``name`` — short unique identifier.
    2. Override ``is_available()`` — return True when the backend can be used.
    3. Override ``generate()`` — produce images and write them to ``output_dir``.

    ``generate()`` must NEVER raise — wrap all errors, log them, and return a
    ``GenerationResult`` with an empty ``images`` list on failure.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier, e.g. 'fal', 'local_sdxl'."""

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this provider can be used right now."""

    @abstractmethod
    def generate(
        self,
        request: GenerationRequest,
        output_dir: Path,
    ) -> GenerationResult:
        """Generate images and save them to *output_dir*.

        Args:
            request: Generation parameters.
            output_dir: Directory where output PNGs must be saved.

        Returns:
            GenerationResult with paths to saved images, or empty list on failure.

        This method must NEVER raise.
        """

    # ── Shared helpers ─────────────────────────────────────────────────────────

    def _ensure_output_dir(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

    def _image_path(self, output_dir: Path, index: int) -> Path:
        return output_dir / f"gen_{index:04d}.png"
