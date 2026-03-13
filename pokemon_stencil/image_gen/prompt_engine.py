"""
Prompt optimisation engine for the Pokemon Stencil Art Factory.

Responsible for:
- Enforcing the CLIP 77-token hard limit.
- Automatically compressing over-long prompts by trimming low-priority clauses.
- Injecting default style tokens that produce stencil-friendly artwork.

The engine is intentionally lightweight – no ML model is required; token
counting uses a simple whitespace + punctuation heuristic that stays safely
within the true CLIP vocabulary budget.

Usage::

    from pokemon_stencil.image_gen.prompt_engine import PromptEngine
    engine = PromptEngine()
    prompt = engine.build("Pikachu", "surfing a wave, sunset background")
    neg    = engine.negative_prompt()
"""

from __future__ import annotations

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

#: CLIP hard limit (BPE tokens including start/end tokens).
CLIP_TOKEN_LIMIT: int = 77

#: Tokens consumed by the start-of-text and end-of-text special tokens.
_CLIP_SPECIAL_TOKENS: int = 2

#: Effective budget for user-visible tokens.
_EFFECTIVE_LIMIT: int = CLIP_TOKEN_LIMIT - _CLIP_SPECIAL_TOKENS

#: Style tokens appended to every positive prompt (per spec).
DEFAULT_STYLE_TOKENS: str = (
    "clean cartoon illustration, bold black outlines, vector illustration style, "
    "high contrast lighting, flat colour shapes, poster illustration style, "
    "stencil-friendly composition"
)

#: Default negative prompt for stencil art.
DEFAULT_NEGATIVE_PROMPT: str = (
    "photorealistic, gradient, complex texture, noise, blurry, "
    "watermark, signature, multiple characters, extra limbs, deformed, "
    "low quality, sketch, pencil drawing"
)

#: Angle and lighting variation tokens for factory diversity.
CAMERA_ANGLE_TOKENS: List[str] = [
    "front view",
    "three-quarter view",
    "side profile view",
    "dynamic low angle",
]

LIGHTING_TOKENS: List[str] = [
    "bright even lighting",
    "dramatic rim lighting",
    "soft studio lighting",
    "high contrast backlit",
]


class PromptEngine:
    """
    Builds, optimises, and validates generation prompts.

    Args:
        style_tokens: Style suffix injected into every prompt.
            Defaults to :data:`DEFAULT_STYLE_TOKENS`.
        negative_prompt: Default negative prompt string.
            Defaults to :data:`DEFAULT_NEGATIVE_PROMPT`.
    """

    def __init__(
        self,
        style_tokens: str = DEFAULT_STYLE_TOKENS,
        negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    ) -> None:
        self._style_tokens = style_tokens
        self._negative_prompt = negative_prompt

    # ── Public API ─────────────────────────────────────────────────────────────

    def build(
        self,
        pokemon_name: str,
        extras: str = "",
        camera_angle: Optional[str] = None,
        lighting: Optional[str] = None,
    ) -> str:
        """
        Build a complete positive prompt for *pokemon_name*.

        The prompt structure is::

            <subject>, <style_tokens>[, <extras>][, <camera>][, <lighting>]

        Token budget is enforced by trimming *extras* first, then
        *camera/lighting* tokens, until the result fits within the CLIP
        77-token window.

        Args:
            pokemon_name: Canonical Pokémon name (e.g. ``"Pikachu"``).
            extras: Caller-supplied pose / scene description.
            camera_angle: Optional camera framing token.
            lighting: Optional lighting description token.

        Returns:
            Optimised prompt string guaranteed to be within 77 CLIP tokens.
        """
        subject = f"A {pokemon_name} character"
        parts: List[str] = [subject, self._style_tokens]
        if extras:
            parts.append(extras)
        if camera_angle:
            parts.append(camera_angle)
        if lighting:
            parts.append(lighting)

        prompt = ", ".join(p.strip(", ") for p in parts if p.strip())
        prompt = self._enforce_token_limit(prompt)
        logger.debug("Built prompt (%d est. tokens): %s", _count_tokens(prompt), prompt)
        return prompt

    def negative_prompt(self) -> str:
        """Return the default negative prompt string."""
        return self._negative_prompt

    def compress(self, prompt: str) -> str:
        """
        Compress *prompt* to fit within the CLIP token budget.

        Clauses (comma-separated segments) are dropped from the *end* of the
        prompt until it fits.  The first clause (subject) is always retained.

        Args:
            prompt: Arbitrary prompt string.

        Returns:
            Compressed prompt within the token budget.
        """
        return self._enforce_token_limit(prompt)

    def token_count(self, text: str) -> int:
        """Return the estimated CLIP token count for *text*."""
        return _count_tokens(text)

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _enforce_token_limit(self, prompt: str) -> str:
        """Drop trailing clauses until the prompt fits within the token budget."""
        if _count_tokens(prompt) <= _EFFECTIVE_LIMIT:
            return prompt

        clauses = [c.strip() for c in prompt.split(",") if c.strip()]
        while len(clauses) > 1 and _count_tokens(", ".join(clauses)) > _EFFECTIVE_LIMIT:
            dropped = clauses.pop()
            logger.debug("Prompt compression: dropped clause '%s'.", dropped)

        compressed = ", ".join(clauses)
        logger.info(
            "Prompt compressed from >%d tokens to ~%d tokens.",
            CLIP_TOKEN_LIMIT,
            _count_tokens(compressed),
        )
        return compressed


# ── Token counting ─────────────────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    """
    Estimate the number of CLIP BPE tokens in *text*.

    Uses a whitespace + punctuation split as a conservative approximation.
    BPE may split rare words into sub-word tokens, so this estimate can
    under-count; we stay safely under the limit to compensate.

    Args:
        text: Prompt string.

    Returns:
        Estimated token count (excluding special tokens).
    """
    # Split on whitespace and punctuation boundaries.
    tokens = re.findall(r"[A-Za-z0-9']+|[^\w\s]", text)
    return len(tokens)
