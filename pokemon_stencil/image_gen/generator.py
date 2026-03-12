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
from pokemon_stencil.models.reference_encoder import ReferenceEncoder

logger = logging.getLogger(__name__)


def _resolve_dtype(dtype_str: str) -> "torch.dtype":  # noqa: F821
    """
    Convert a string dtype name to a ``torch.dtype``.

    Args:
        dtype_str: One of ``"float32"``, ``"float16"``, or ``"bfloat16"``.

    Returns:
        Corresponding ``torch.dtype``.  Defaults to ``torch.float32`` for
        unknown values.
    """
    try:
        import torch
        return {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }.get(dtype_str, torch.float32)
    except ImportError:  # pragma: no cover
        return None  # type: ignore[return-value]


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
        reference_images: Optional[List[Image.Image]] = None,
    ) -> List[Image.Image]:
        """
        Generate one or more stylised images of a named Pokemon.

        When *reference_images* are supplied and
        ``config.use_reference_conditioning`` is ``True``, the
        :class:`~pokemon_stencil.models.reference_encoder.ReferenceEncoder`
        is used to produce a blended reference image which is then fed into
        an image-to-image pipeline.  This improves character identity
        consistency across generated candidates.

        Falls back to normal text-to-image generation when references are
        absent or when the img2img pipeline cannot be loaded.

        Args:
            pokemon_name: Canonical Pokemon name (e.g. ``"Pikachu"``).
            prompt_extras: Additional prompt fragments appended to the template.
            num_images: Number of images to generate.
            seed: Optional RNG seed; overrides ``config.seed`` when provided.
            reference_images: Optional list of reference PIL Images used to
                condition generation.  Ignored when
                ``config.use_reference_conditioning`` is ``False``.

        Returns:
            List of PIL Images in RGB mode with white backgrounds.
        """
        use_refs = (
            bool(reference_images)
            and self.config.use_reference_conditioning
        )

        if use_refs:
            return self._generate_with_references(
                pokemon_name, prompt_extras, num_images, seed,
                reference_images,  # type: ignore[arg-type]
            )
        return self._generate_text_to_image(
            pokemon_name, prompt_extras, num_images, seed
        )

    # ── Reference-conditioned generation ─────────────────────────────────────

    def _generate_with_references(
        self,
        pokemon_name: str,
        prompt_extras: str,
        num_images: int,
        seed: Optional[int],
        reference_images: List[Image.Image],
    ) -> List[Image.Image]:
        """
        Run image-to-image generation conditioned on *reference_images*.

        Blends the references into a single conditioning image via
        :class:`ReferenceEncoder`, then attempts to load an img2img SD
        pipeline.  Falls back to text-to-image if the img2img pipeline
        cannot be instantiated.

        Args:
            pokemon_name: Canonical Pokemon name.
            prompt_extras: Extra prompt text.
            num_images: Number of images to produce.
            seed: Optional RNG seed.
            reference_images: Non-empty list of reference PIL Images.

        Returns:
            List of RGB PIL Images with white backgrounds.
        """
        target_size = (self.config.width, self.config.height)
        encoder = ReferenceEncoder(target_size=target_size)
        conditioning_image = encoder.blend(reference_images)
        logger.info(
            "Reference conditioning: %d image(s) blended for '%s'.",
            len(reference_images),
            pokemon_name,
        )

        try:
            return self._generate_img2img(
                pokemon_name, prompt_extras, num_images, seed,
                conditioning_image,
            )
        except Exception as exc:
            logger.warning(
                "img2img pipeline unavailable (%s); falling back to text2img.",
                exc,
            )
            return self._generate_text_to_image(
                pokemon_name, prompt_extras, num_images, seed
            )

    def _generate_img2img(
        self,
        pokemon_name: str,
        prompt_extras: str,
        num_images: int,
        seed: Optional[int],
        conditioning_image: Image.Image,
    ) -> List[Image.Image]:
        """
        Run a Stable Diffusion img2img pass.

        Loads ``StableDiffusionImg2ImgPipeline`` from the same model source
        as the base pipeline.  The call is CPU-compatible.

        Args:
            pokemon_name: Canonical Pokemon name.
            prompt_extras: Extra prompt text.
            num_images: Number of images to produce.
            seed: Optional RNG seed.
            conditioning_image: Blended reference image used as the img2img
                input.

        Returns:
            List of RGB PIL Images.
        """
        from diffusers import StableDiffusionImg2ImgPipeline

        prompt = self._build_prompt(pokemon_name, prompt_extras)
        negative = self.config.negative_prompt or _NEGATIVE_PROMPT
        effective_seed = seed if seed is not None else self.config.seed
        strength = self.config.reference_strength

        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            str(self.config.model_id),
            torch_dtype=_resolve_dtype(self.config.torch_dtype),
            safety_checker=None,
        )
        pipe = pipe.to(self.config.device)

        logger.info(
            "img2img | pokemon=%s | steps=%d | strength=%.2f | seed=%s",
            pokemon_name,
            self.config.num_inference_steps,
            strength,
            effective_seed,
        )

        images: List[Image.Image] = []
        for i in range(num_images):
            img_seed = (
                (effective_seed + i) if effective_seed is not None else None
            )
            gen = self._make_generator(img_seed)
            result = pipe(
                prompt=prompt,
                negative_prompt=negative,
                image=conditioning_image,
                strength=strength,
                num_inference_steps=self.config.num_inference_steps,
                guidance_scale=self.config.guidance_scale,
                generator=gen,
            )
            img = self._ensure_white_background(result.images[0])
            images.append(img)
            logger.debug("img2img image %d/%d done.", i + 1, num_images)

        return images

    # ── Text-to-image generation ──────────────────────────────────────────────

    def _generate_text_to_image(
        self,
        pokemon_name: str,
        prompt_extras: str,
        num_images: int,
        seed: Optional[int],
    ) -> List[Image.Image]:
        """
        Run standard text-to-image generation.

        Args:
            pokemon_name: Canonical Pokemon name.
            prompt_extras: Extra prompt text.
            num_images: Number of images to produce.
            seed: Optional RNG seed.

        Returns:
            List of RGB PIL Images with white backgrounds.
        """
        pipe = load_sd_pipeline(self.config)
        prompt = self._build_prompt(pokemon_name, prompt_extras)
        negative = self.config.negative_prompt or _NEGATIVE_PROMPT
        effective_seed = seed if seed is not None else self.config.seed

        logger.info(
            "text2img | pokemon=%s | %d image(s) | steps=%d | seed=%s",
            pokemon_name,
            num_images,
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
            img = self._ensure_white_background(result.images[0])
            images.append(img)
            logger.debug("text2img image %d/%d done.", i + 1, num_images)

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
