"""
Pokemon image generator – upgraded to DreamShaper + ControlNet pipeline.

Generation stack
----------------
Text Prompt → PromptEngine → StableDiffusionControlNetPipeline
    ↓ ControlNet OpenPose (pose guidance)
    ↓ ControlNet Canny    (edge/structure guidance)
    ↓ DreamShaper base    (high-quality character generation)
    ↓ optional LoRA       (character-specific fine-tuning)
    ↓ optional img2img    (reference-encoding character preservation)

When ``config.use_controlnet`` is ``False`` the legacy
``StableDiffusionPipeline`` is used instead (SD 1.5 / DreamShaper text-only).

Model loading is delegated to
:mod:`~pokemon_stencil.models.model_loader` so that pipeline instances
are cached and never loaded twice in the same process.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from pokemon_stencil.config import GenerationConfig
from pokemon_stencil.image_gen.prompt_engine import PromptEngine
from pokemon_stencil.models.model_loader import (
    load_controlnet_pipeline,
    load_sd_pipeline,
)

logger = logging.getLogger(__name__)

#: Fallback prompt template used when the PromptEngine is disabled.
_PROMPT_TEMPLATE = (
    "A {pokemon_name} character, clean cartoon illustration, bold outlines, "
    "vector style, high contrast lighting, stencil art style, white background"
)

#: Default negative prompt.
_NEGATIVE_PROMPT = (
    "photorealistic, gradient, complex texture, noise, blurry, "
    "watermark, signature, multiple characters, extra limbs, deformed, low quality"
)


class PokemonImageGenerator:
    """
    Generates stylised Pokémon artwork using the DreamShaper + ControlNet stack.

    Args:
        config: :class:`~pokemon_stencil.config.GenerationConfig` controlling
                model selection, inference parameters, LoRA, and conditioning.
    """

    def __init__(self, config: GenerationConfig) -> None:
        self.config = config
        self._prompt_engine = PromptEngine(
            style_tokens=config.style_suffix,
            negative_prompt=config.negative_prompt or _NEGATIVE_PROMPT,
        )

    # ── Public API ─────────────────────────────────────────────────────────────

    def generate(
        self,
        pokemon_name: str,
        prompt_extras: str = "",
        num_images: int = 1,
        seed: Optional[int] = None,
        composition_map: Optional[np.ndarray] = None,
        reference_image: Optional[Image.Image] = None,
        camera_angle: Optional[str] = None,
        lighting: Optional[str] = None,
    ) -> List[Image.Image]:
        """
        Generate one or more stylised images of a named Pokémon.

        When ``config.use_controlnet`` is ``True``:
        - OpenPose and Canny conditioning images are derived from
          *reference_image* or *composition_map*.
        - ``StableDiffusionControlNetPipeline`` is used.

        When ``config.use_reference_encoding`` is ``True`` and a
        *reference_image* is provided, generation uses img2img mode at
        ``config.reference_strength`` denoising strength.

        When ``config.optimise_prompt`` is ``True``, the prompt is routed
        through :class:`~pokemon_stencil.image_gen.prompt_engine.PromptEngine`
        to enforce the 77-token CLIP limit.

        Args:
            pokemon_name: Canonical Pokémon name (e.g. ``"Pikachu"``).
            prompt_extras: Additional prompt text (pose/scene description).
            num_images: Number of images to generate.
            seed: Optional RNG seed; overrides ``config.seed`` when provided.
            composition_map: Grayscale structural guide (legacy path).
            reference_image: Pre-encoded reference conditioning image.
            camera_angle: Optional camera framing token injected by the factory.
            lighting: Optional lighting token injected by the factory.

        Returns:
            List of RGB PIL Images with white backgrounds.
        """
        effective_seed = seed if seed is not None else self.config.seed
        prompt = self._build_prompt(pokemon_name, prompt_extras, camera_angle, lighting)
        negative = self.config.negative_prompt or _NEGATIVE_PROMPT

        logger.info(
            "Generating %d image(s) for '%s' | controlnet=%s | steps=%d | seed=%s",
            num_images,
            pokemon_name,
            self.config.use_controlnet,
            self.config.num_inference_steps,
            effective_seed,
        )

        images: List[Image.Image] = []
        for i in range(num_images):
            img_seed = (effective_seed + i) if effective_seed is not None else None
            generator = self._make_generator(img_seed)

            img = self._generate_one(
                prompt=prompt,
                negative=negative,
                generator=generator,
                composition_map=composition_map,
                reference_image=reference_image,
            )
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
            List of saved file paths.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        saved: List[Path] = []
        for idx, img in enumerate(images):
            path = output_dir / f"{prefix}_{idx:03d}.png"
            img.save(path)
            logger.info("Saved generated image: %s", path)
            saved.append(path)
        return saved

    # ── Internal generation dispatch ───────────────────────────────────────────

    def _generate_one(
        self,
        prompt: str,
        negative: str,
        generator,
        composition_map: Optional[np.ndarray],
        reference_image: Optional[Image.Image],
    ) -> Image.Image:
        """Dispatch to ControlNet or legacy SD pipeline for one image."""
        if self.config.use_controlnet:
            return self._generate_controlnet(
                prompt, negative, generator, composition_map, reference_image
            )
        return self._generate_sd(
            prompt, negative, generator, composition_map, reference_image
        )

    def _generate_controlnet(
        self,
        prompt: str,
        negative: str,
        generator,
        composition_map: Optional[np.ndarray],
        reference_image: Optional[Image.Image],
    ) -> Image.Image:
        """Generate using StableDiffusionControlNetPipeline."""
        pipe = load_controlnet_pipeline(self.config)

        # Build ControlNet conditioning images.
        ctrl_size = (self.config.width, self.config.height)
        conditioning_source = reference_image or (
            self._composition_map_to_image(composition_map, ctrl_size)
            if composition_map is not None
            else self._blank_image(ctrl_size)
        )

        openpose_img = self._extract_openpose(conditioning_source)
        canny_img = self._extract_canny(conditioning_source)

        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative,
            image=[openpose_img, canny_img],
            controlnet_conditioning_scale=[
                self.config.controlnet_openpose_scale,
                self.config.controlnet_canny_scale,
            ],
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            width=self.config.width,
            height=self.config.height,
            generator=generator,
        )

        # img2img mode when reference encoding is enabled.
        if (
            self.config.use_reference_encoding
            and reference_image is not None
        ):
            pipe_kwargs["image"] = reference_image.resize(ctrl_size, Image.LANCZOS)
            pipe_kwargs["strength"] = self.config.reference_strength

        result = pipe(**pipe_kwargs)
        return result.images[0]

    def _generate_sd(
        self,
        prompt: str,
        negative: str,
        generator,
        composition_map: Optional[np.ndarray],
        reference_image: Optional[Image.Image],
    ) -> Image.Image:
        """Generate using legacy StableDiffusionPipeline (no ControlNet)."""
        pipe = load_sd_pipeline(self.config)

        use_img2img = (
            (
                composition_map is not None
                and self.config.use_composition_guidance
            )
            or (
                reference_image is not None
                and self.config.use_reference_encoding
            )
        )

        pipe_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=self.config.num_inference_steps,
            guidance_scale=self.config.guidance_scale,
            width=self.config.width,
            height=self.config.height,
            generator=generator,
        )

        if use_img2img:
            ctrl_size = (self.config.width, self.config.height)
            if reference_image is not None:
                cond_img = reference_image.resize(ctrl_size, Image.LANCZOS)
                strength = self.config.reference_strength
            else:
                cond_img = self._composition_map_to_image(composition_map, ctrl_size)
                strength = self.config.composition_strength
            pipe_kwargs["image"] = cond_img
            pipe_kwargs["strength"] = strength

        result = pipe(**pipe_kwargs)
        return result.images[0]

    # ── Prompt helpers ─────────────────────────────────────────────────────────

    def _build_prompt(
        self,
        pokemon_name: str,
        extras: str,
        camera_angle: Optional[str],
        lighting: Optional[str],
    ) -> str:
        if self.config.optimise_prompt:
            return self._prompt_engine.build(
                pokemon_name,
                extras=extras,
                camera_angle=camera_angle,
                lighting=lighting,
            )
        base = _PROMPT_TEMPLATE.format(pokemon_name=pokemon_name)
        parts = [base]
        if extras:
            parts.append(extras)
        if camera_angle:
            parts.append(camera_angle)
        if lighting:
            parts.append(lighting)
        return ", ".join(parts)

    # ── ControlNet conditioning helpers ────────────────────────────────────────

    def _extract_openpose(self, image: Image.Image) -> Image.Image:
        """
        Extract an OpenPose conditioning map from *image*.

        Uses ``controlnet_aux`` OpenposeDetector when available; falls
        back to returning the Canny edge map which gives structural guidance
        even without pose detection.
        """
        size = (self.config.width, self.config.height)
        img = image.resize(size, Image.LANCZOS)
        try:
            from controlnet_aux import OpenposeDetector
            detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            return detector(img)
        except Exception:
            logger.debug(
                "OpenposeDetector unavailable; using Canny map for OpenPose conditioning."
            )
            return self._extract_canny(img)

    @staticmethod
    def _extract_canny(image: Image.Image) -> Image.Image:
        """Extract a Canny edge map from *image* for ControlNet conditioning."""
        import cv2
        import numpy as np
        arr = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, threshold1=100, threshold2=200)
        edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(edges_rgb)

    # ── General helpers ────────────────────────────────────────────────────────

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

    @staticmethod
    def _composition_map_to_image(
        composition_map: np.ndarray,
        size: Tuple[int, int],
    ) -> Image.Image:
        """Convert a composition map array to a PIL RGB Image of *size*."""
        if composition_map.ndim == 2:
            comp_rgb = np.stack([composition_map] * 3, axis=-1)
        else:
            comp_rgb = composition_map
        pil_img = Image.fromarray(comp_rgb.astype(np.uint8))
        return pil_img.resize(size, Image.LANCZOS)

    @staticmethod
    def _blank_image(size: Tuple[int, int]) -> Image.Image:
        """Return a blank white image of *size* for use when no reference exists."""
        return Image.new("RGB", size, (255, 255, 255))
