"""
Stable Diffusion 1.5 image generator optimised for CPU execution.

Generates stylised Pokemon artwork suitable for stencil conversion.
The generator is intentionally configured for flat, bold, minimal-detail
output via prompt engineering and post-processing.

Model loading is delegated to :mod:`pokemon_stencil.models.model_loader`
so that all pipeline consumers share a single cached instance.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from pokemon_stencil.config import GenerationConfig
from pokemon_stencil.models.model_loader import load_sd_pipeline

logger = logging.getLogger(__name__)

#: Prompt template that produces stencil-friendly Pokemon artwork.
_PROMPT_TEMPLATE = (
    "A dynamic illustration of {pokemon_name}, bold outlines, flat colors, "
    "stencil-friendly art style, poster style, strong silhouette"
)

#: Default negative prompt; can be overridden per-config.
_NEGATIVE_PROMPT = (
    "photorealistic, blurry, noisy, watercolor, oil painting, detailed texture"
)


class PokemonImageGenerator:
    """
    Wraps a Stable Diffusion 1.5 pipeline for CPU-based Pokemon art generation.

    The SD pipeline is resolved and cached by
    :func:`~pokemon_stencil.models.model_loader.load_sd_pipeline` on first
    call to :meth:`generate`, so startup cost is incurred only when generation
    is actually needed (``skip_generation=False``).

    Args:
        config: :class:`~pokemon_stencil.config.GenerationConfig` controlling
                model source, inference steps, guidance scale, and seed.
    """

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config

    # ── Public API ────────────────────────────────────────────────────────────

    def generate(
        self,
        pokemon_name: str,
        prompt_extras: str = "",
        num_images: int = 1,
        seed: Optional[int] = None,
    ) -> List[Image.Image]:
        """
        Generate one or more stylised images of a named Pokemon.

        Args:
            pokemon_name: Canonical Pokemon name (e.g. ``"Pikachu"``).
            prompt_extras: Additional prompt fragments appended to the template.
            num_images: Number of images to generate.
            seed: Optional RNG seed; overrides ``config.seed`` when provided.

        Returns:
            List of PIL Images in RGB mode with white backgrounds.
        """
        pipe = load_sd_pipeline(self.config)
        prompt = self._build_prompt(pokemon_name, prompt_extras)
        negative = self.config.negative_prompt or _NEGATIVE_PROMPT
        effective_seed = seed if seed is not None else self.config.seed

        logger.info(
            "Generating %d image(s) for '%s' | steps=%d | seed=%s",
            num_images,
            pokemon_name,
            self.config.num_inference_steps,
            effective_seed,
        )

        images: List[Image.Image] = []
        for i in range(num_images):
            img_seed = (
                (effective_seed + i) if effective_seed is not None else None
            )
            generator = self._make_generator(img_seed)
            result = pipe(
                prompt=prompt,
                negative_prompt=negative,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                width=self.config.width,
                height=self.config.height,
                generator=generator,
            )
            img = result.images[0]
            img = self._ensure_white_background(img)
            images.append(img)
            logger.debug("Generated image %d/%d", i + 1, num_images)

        return images

    def save_images(
        self,
        images: List[Image.Image],
        output_dir: Path,
        prefix: str = "gen",
    ) -> List[Path]:
        """
        Save a list of PIL Images to *output_dir* as PNG files.

        Args:
            images: Images to save.
            output_dir: Directory to write files into (created if absent).
            prefix: Filename prefix (default ``"gen"``).

        Returns:
            List of :class:`~pathlib.Path` objects for the saved files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []
        for idx, img in enumerate(images):
            path = output_dir / f"{prefix}_{idx:03d}.png"
            img.save(path)
            logger.info("Saved generated image: %s", path)
            saved.append(path)
        return saved

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _build_prompt(self, pokemon_name: str, extras: str) -> str:
        """
        Construct the full positive prompt for *pokemon_name*.

        Uses the module-level :data:`_PROMPT_TEMPLATE` and appends the
        config's ``style_suffix`` and any caller-supplied *extras*.
        """
        base = _PROMPT_TEMPLATE.format(pokemon_name=pokemon_name)
        parts = [base]
        if self.config.style_suffix:
            parts.append(self.config.style_suffix)
        if extras:
            parts.append(extras)
        return ", ".join(parts)

    def _make_generator(self, seed: Optional[int]):
        """Create a ``torch.Generator`` seeded with *seed* (``None`` → random)."""
        try:
            import torch
        except ImportError:
            return None
        if seed is None:
            return None
        gen = torch.Generator(device=self.config.device)
        gen.manual_seed(seed)
        return gen

    @staticmethod
    def _ensure_white_background(img: Image.Image) -> Image.Image:
        """Composite any alpha channel onto a white RGB background."""
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            return background
        return img.convert("RGB")
