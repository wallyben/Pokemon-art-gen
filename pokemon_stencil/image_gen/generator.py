"""
Stable Diffusion 1.5 image generator optimised for CPU execution.

Generates stylised Pokemon artwork suitable for stencil conversion.
The generator is intentionally configured for flat, bold, minimal-detail
output via prompt engineering and post-processing.
"""

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

from pokemon_stencil.config import GenerationConfig

logger = logging.getLogger(__name__)


class PokemonImageGenerator:
    """
    Wraps a Stable Diffusion 1.5 pipeline for CPU-based Pokemon art generation.

    The pipeline is loaded lazily on first use to avoid startup cost when the
    caller only wants to use reference images (skip_generation=True).
    """

    def __init__(self, config: GenerationConfig) -> None:
        """
        Initialise the generator with the provided configuration.

        Args:
            config: GenerationConfig controlling model, steps, guidance, etc.
        """
        self.config = config
        self._pipe = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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
            pokemon_name: Canonical Pokemon name (e.g. "Pikachu").
            prompt_extras: Additional prompt fragments from the caller.
            num_images: How many images to generate in this call.
            seed: Optional RNG seed for reproducibility.

        Returns:
            List of PIL Images with white backgrounds.
        """
        pipe = self._load_pipeline()
        prompt = self._build_prompt(pokemon_name, prompt_extras)
        negative = self.config.negative_prompt
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
            prefix: Filename prefix.

        Returns:
            List of Path objects for the saved files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []
        for idx, img in enumerate(images):
            path = output_dir / f"{prefix}_{idx:03d}.png"
            img.save(path)
            logger.info("Saved generated image: %s", path)
            saved.append(path)
        return saved

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_pipeline(self):
        """Lazy-load and cache the diffusers StableDiffusionPipeline."""
        if self._pipe is not None:
            return self._pipe

        try:
            import torch
            from diffusers import StableDiffusionPipeline
        except ImportError as exc:
            raise ImportError(
                "torch and diffusers are required for image generation. "
                "Install them with: pip install torch diffusers transformers accelerate"
            ) from exc

        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(self.config.torch_dtype, torch.float32)

        logger.info(
            "Loading Stable Diffusion pipeline '%s' on %s (dtype=%s) …",
            self.config.model_id,
            self.config.device,
            self.config.torch_dtype,
        )

        pipe = StableDiffusionPipeline.from_pretrained(
            self.config.model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,
            requires_safety_checker=False,
        )
        pipe = pipe.to(self.config.device)

        # CPU memory optimisations
        pipe.enable_attention_slicing()

        self._pipe = pipe
        logger.info("Pipeline loaded.")
        return self._pipe

    def _build_prompt(self, pokemon_name: str, extras: str) -> str:
        """Construct the full positive prompt for a given Pokemon."""
        parts = [
            f"{pokemon_name} pokemon",
            self.config.style_suffix,
        ]
        if extras:
            parts.append(extras)
        return ", ".join(parts)

    def _make_generator(self, seed: Optional[int]):
        """Create a torch Generator seeded with *seed* (or None for random)."""
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
        """Composite any alpha channel onto white."""
        if img.mode == "RGBA":
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            return background
        return img.convert("RGB")
