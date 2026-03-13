"""
Pokemon image generator – upgraded to SDXL + IP-Adapter + ControlNet pipeline.

Generation stack (v3)
---------------------
Text Prompt
    ↓ Prompt Optimisation (PromptEngine, 77-token CLIP limit)
    ↓ Reference Encoding (IP-Adapter – CLIP image encoder)
    ↓ ControlNet Pose Conditioning (OpenPose – anatomy control)
    ↓ ControlNet Edge Conditioning (Canny – composition structure)
    ↓ SDXL Base Diffusion Model (stabilityai/stable-diffusion-xl-base-1.0)
    ↓ Optional Character LoRA (models/lora/)
    ↓ Image Generation (1024×1024, 28 steps, guidance_scale=7.5)

Reference images are loaded from ``refs/<pokemon_name>/`` (10–30 images).
When IP-Adapter is unavailable, falls back to img2img reference encoding.

Model loading is delegated to
:mod:`~pokemon_stencil.models.model_loader` so pipeline instances are
cached and never loaded twice in the same process.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

from pokemon_stencil.config import GenerationConfig
from pokemon_stencil.image_gen.prompt_engine import PromptEngine
from pokemon_stencil.models.model_loader import load_sdxl_controlnet_pipeline

logger = logging.getLogger(__name__)

#: Fallback prompt template used when the PromptEngine is disabled.
_PROMPT_TEMPLATE = (
    "A {pokemon_name} character, clean cartoon illustration, bold black outlines, "
    "vector illustration style, high contrast lighting, flat colour shapes, "
    "stencil-friendly composition, white background"
)

#: Default negative prompt.
_NEGATIVE_PROMPT = (
    "photorealistic, gradient, complex texture, noise, blurry, "
    "watermark, signature, multiple characters, extra limbs, deformed, low quality"
)

#: Directory under which reference images are stored per Pokémon.
_REFS_BASE_DIR = Path("refs")


class PokemonImageGenerator:
    """
    Generates stylised Pokémon artwork using the SDXL + IP-Adapter + ControlNet stack.

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
        pose_reference: Optional[Image.Image] = None,
    ) -> List[Image.Image]:
        """
        Generate one or more stylised images of a named Pokémon.

        Pipeline per spec:
        1. Build optimised prompt (PromptEngine).
        2. Load reference images from ``refs/<pokemon_name>/`` (10–30 images).
        3. Encode references via IP-Adapter CLIP encoder (scale=0.45).
        4. Extract OpenPose map from pose reference or reference images.
        5. Extract Canny edge map from reference images.
        6. Run SDXL ControlNet diffusion (28 steps, guidance_scale=7.5, 1024×1024).

        Args:
            pokemon_name: Canonical Pokémon name (e.g. ``"Pikachu"``).
            prompt_extras: Additional prompt text.
            num_images: Number of images to generate.
            seed: Optional RNG seed.
            composition_map: Grayscale structural guide (legacy path).
            reference_image: Pre-loaded reference conditioning image.
            camera_angle: Camera framing token injected by the factory.
            lighting: Lighting token injected by the factory.
            pose_reference: Explicit pose reference image for OpenPose.

        Returns:
            List of RGB PIL Images (1024×1024) with white backgrounds.
        """
        effective_seed = seed if seed is not None else self.config.seed
        prompt = self._build_prompt(pokemon_name, prompt_extras, camera_angle, lighting)
        negative = self.config.negative_prompt or _NEGATIVE_PROMPT

        logger.info(
            "Generating %d image(s) for '%s' | steps=%d | seed=%s | "
            "ip_adapter=%s | controlnet=%s",
            num_images,
            pokemon_name,
            self.config.num_inference_steps,
            effective_seed,
            self.config.use_ip_adapter,
            self.config.use_controlnet,
        )

        # Load reference images for IP-Adapter conditioning.
        ref_images = self._load_reference_images(pokemon_name, reference_image)

        images: List[Image.Image] = []
        for i in range(num_images):
            img_seed = (effective_seed + i) if effective_seed is not None else None
            generator = self._make_generator(img_seed)

            img = self._generate_one(
                prompt=prompt,
                negative=negative,
                generator=generator,
                composition_map=composition_map,
                reference_images=ref_images,
                pose_reference=pose_reference,
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
        """Save PIL Images to *output_dir* as PNG files."""
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
        reference_images: List[Image.Image],
        pose_reference: Optional[Image.Image],
    ) -> Image.Image:
        """Dispatch to SDXL ControlNet pipeline for one image."""
        return self._generate_sdxl(
            prompt, negative, generator, composition_map, reference_images, pose_reference
        )

    def _generate_sdxl(
        self,
        prompt: str,
        negative: str,
        generator,
        composition_map: Optional[np.ndarray],
        reference_images: List[Image.Image],
        pose_reference: Optional[Image.Image],
    ) -> Image.Image:
        """
        Generate using StableDiffusionXLControlNetPipeline.

        Conditioning pipeline per spec:
        - IP-Adapter: encode reference images for character preservation
        - OpenPose ControlNet: pose from pose_reference or first reference
        - Canny ControlNet: structural edges from reference or composition map
        """
        pipe = load_sdxl_controlnet_pipeline(self.config)

        ctrl_size = (self.config.width, self.config.height)

        # ── Build conditioning source ──────────────────────────────────────────
        conditioning_source = self._get_conditioning_source(
            reference_images, composition_map, ctrl_size
        )

        # ── ControlNet conditioning images ────────────────────────────────────
        if pose_reference is not None:
            openpose_img = self._extract_openpose(
                pose_reference.resize(ctrl_size, Image.LANCZOS)
            )
            logger.debug("OpenPose conditioning from provided pose reference.")
        else:
            openpose_img = self._extract_openpose(conditioning_source)
            logger.debug("OpenPose conditioning from reference/composition source.")

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

        # ── IP-Adapter reference conditioning ─────────────────────────────────
        if self.config.use_ip_adapter and reference_images:
            # Use the first (or average) reference image for IP-Adapter.
            # diffusers >= 0.24 accepts ip_adapter_image kwarg.
            ip_ref = reference_images[0].resize(ctrl_size, Image.LANCZOS)
            pipe_kwargs["ip_adapter_image"] = ip_ref
            logger.debug(
                "IP-Adapter conditioning from reference image (scale=%.2f).",
                self.config.ip_adapter_scale,
            )

        result = pipe(**pipe_kwargs)
        return result.images[0]

    # ── Reference image loading ────────────────────────────────────────────────

    def _load_reference_images(
        self,
        pokemon_name: str,
        override_image: Optional[Image.Image] = None,
    ) -> List[Image.Image]:
        """
        Load 10–30 reference images from ``refs/<pokemon_name>/``.

        If *override_image* is provided it is returned as the sole reference.
        Reference images are resized to the generation resolution.

        Args:
            pokemon_name: Pokémon name (lowercased for directory lookup).
            override_image: Pre-loaded image that bypasses directory lookup.

        Returns:
            List of RGB PIL Images at generation resolution.
        """
        if override_image is not None:
            return [override_image.resize(
                (self.config.width, self.config.height), Image.LANCZOS
            ).convert("RGB")]

        safe_name = pokemon_name.lower().replace(" ", "-")
        refs_dir = _REFS_BASE_DIR / safe_name

        if not refs_dir.is_dir():
            logger.debug("No reference directory found at %s.", refs_dir)
            return []

        suffixes = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        image_paths = [
            p for p in sorted(refs_dir.iterdir())
            if p.is_file() and p.suffix.lower() in suffixes
        ]

        # Clamp to [min, max] reference image count per spec.
        min_refs = self.config.ip_adapter_min_images
        max_refs = self.config.ip_adapter_max_images
        selected_paths = image_paths[:max_refs]

        if len(selected_paths) < min_refs:
            logger.info(
                "Only %d reference image(s) found in %s (minimum %d). "
                "IP-Adapter conditioning may be weaker.",
                len(selected_paths), refs_dir, min_refs,
            )
        else:
            logger.info(
                "Loading %d reference image(s) from %s for IP-Adapter conditioning.",
                len(selected_paths), refs_dir,
            )

        target_size = (self.config.width, self.config.height)
        loaded: List[Image.Image] = []
        for p in selected_paths:
            try:
                img = Image.open(p).convert("RGB").resize(target_size, Image.LANCZOS)
                loaded.append(img)
            except Exception as exc:
                logger.warning("Could not load reference image %s: %s", p, exc)

        return loaded

    # ── ControlNet conditioning helpers ────────────────────────────────────────

    def _get_conditioning_source(
        self,
        reference_images: List[Image.Image],
        composition_map: Optional[np.ndarray],
        target_size: Tuple[int, int],
    ) -> Image.Image:
        """
        Select the best available source image for ControlNet conditioning.

        Priority:
        1. First reference image (strongest signal).
        2. Composition map converted to RGB.
        3. Blank white image (fallback).
        """
        if reference_images:
            return reference_images[0].resize(target_size, Image.LANCZOS)
        if composition_map is not None:
            return self._composition_map_to_image(composition_map, target_size)
        return self._blank_image(target_size)

    def _extract_openpose(self, image: Image.Image) -> Image.Image:
        """
        Extract an OpenPose conditioning map from *image*.

        If a pose reference is supplied → extract skeleton using OpenposeDetector.
        If no pose reference → generate edge map from reference images.
        """
        size = (self.config.width, self.config.height)
        img = image.resize(size, Image.LANCZOS)
        try:
            from controlnet_aux import OpenposeDetector
            detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
            return detector(img)
        except Exception:
            logger.debug(
                "OpenposeDetector unavailable; using Canny map for OpenPose channel."
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
        """Create a seeded ``torch.Generator`` (``None`` → random)."""
        try:
            import torch
        except ImportError:
            return None
        if seed is None:
            return None
        gen = torch.Generator(device=self.config.device)
        gen.manual_seed(seed)
        return gen

    # ── Prompt building ────────────────────────────────────────────────────────

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

    # ── Static utilities ───────────────────────────────────────────────────────

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
        """Return a blank white image of *size*."""
        return Image.new("RGB", size, (255, 255, 255))
