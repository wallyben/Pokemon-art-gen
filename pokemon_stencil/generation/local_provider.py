"""
pokemon_stencil.generation.local_provider — local SDXL generation backend.

Fallback backend using the hardened local SDXL + ControlNet + IP-Adapter
stack.  Wraps ``pokemon_stencil.models.model_loader`` which was rebuilt to
NEVER crash from an incompatible LoRA.

When to use:
    - No FAL_KEY configured (offline / no API cost mode).
    - User explicitly requests provider="local".

Performance note:
    CPU inference is very slow (~2–4 hours per image for SDXL on a modern
    CPU without GPU offload).  For local generation, a CUDA-capable GPU is
    strongly recommended.  Use fal.ai for CPU-only machines.

LoRA safety:
    All LoRA validation is handled by the model_loader layer.  SD1.5 LoRAs
    are silently blocked; the pipeline continues in pure SDXL mode.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from pokemon_stencil.generation.base import (
    AbstractGenerationProvider,
    GenerationRequest,
    GenerationResult,
)

logger = logging.getLogger(__name__)


class LocalSDXLProvider(AbstractGenerationProvider):
    """Fallback SDXL generation using the local model stack.

    Reference conditioning:
        When reference_images are provided, the first valid reference is used
        as the IP-Adapter conditioning image (character appearance guidance).

    LoRA:
        If ``lora_path`` is set in ``extra``, it is forwarded to the pipeline.
        Only SDXL-compatible LoRAs are applied; incompatible LoRAs are silently
        blocked at the model_loader level.
    """

    @property
    def name(self) -> str:
        return "local_sdxl"

    def is_available(self) -> bool:
        """Return True if torch and diffusers can be imported."""
        try:
            import torch  # noqa: F401
            import diffusers  # noqa: F401
            return True
        except ImportError:
            return False

    def generate(
        self,
        request: GenerationRequest,
        output_dir: Path,
    ) -> GenerationResult:
        """Run local SDXL pipeline and save images to *output_dir*."""
        self._ensure_output_dir(output_dir)

        try:
            from pokemon_stencil.config import GenerationConfig
            from pokemon_stencil.image_gen.generator import PokemonImageGenerator
        except ImportError as exc:
            logger.error("LocalSDXLProvider: missing dependency — %s", exc)
            return GenerationResult(images=[], provider=self.name)

        # ── Build GenerationConfig from request ───────────────────────────────
        extra = request.extra or {}
        lora_path = extra.get("lora_path")
        lora_scale = float(extra.get("lora_scale", 0.8))
        use_ip_adapter = extra.get("use_ip_adapter", True)
        torch_dtype = extra.get("torch_dtype", "float32")

        try:
            cfg = GenerationConfig(
                lora_path=Path(lora_path) if lora_path else None,
                lora_scale=lora_scale,
                use_ip_adapter=bool(use_ip_adapter),
                steps=request.num_steps,
                guidance_scale=request.guidance_scale,
                width=request.width,
                height=request.height,
                seed=request.seed,
                torch_dtype=torch_dtype,
            )
        except Exception as exc:
            logger.error("LocalSDXLProvider: bad GenerationConfig — %s", exc)
            return GenerationResult(images=[], provider=self.name)

        # ── Reference image ───────────────────────────────────────────────────
        valid_refs = [r for r in request.reference_images if r.exists()]
        ip_adapter_image = valid_refs[0] if valid_refs else None

        # ── Generate ──────────────────────────────────────────────────────────
        saved: List[Path] = []
        try:
            gen = PokemonImageGenerator(cfg)
        except Exception as exc:
            logger.error("LocalSDXLProvider: generator init failed — %s", exc)
            return GenerationResult(images=[], provider=self.name)

        for img_idx in range(request.num_images):
            seed = (request.seed + img_idx) if request.seed is not None else None
            try:
                images = gen.generate(
                    pokemon_name=extra.get("pokemon_name", "pokemon"),
                    prompt_extras=request.prompt,
                    num_images=1,
                    seed=seed,
                    ip_adapter_image=ip_adapter_image,
                )
                if images:
                    dest = self._image_path(output_dir, img_idx)
                    images[0].save(str(dest), "PNG")
                    saved.append(dest)
                    logger.info("[local_sdxl] Saved image to %s", dest)
            except Exception as exc:
                logger.error(
                    "[local_sdxl] Generation failed for index %d: %s", img_idx, exc
                )

        return GenerationResult(
            images=saved,
            provider=self.name,
            metadata={"dtype": torch_dtype, "steps": request.num_steps},
        )
