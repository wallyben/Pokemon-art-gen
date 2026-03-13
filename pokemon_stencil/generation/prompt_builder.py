"""
pokemon_stencil.generation.prompt_builder — Pokémon-specific prompt construction.

Builds character-accurate, stencil-friendly prompts for both:
    - FLUX.1 (detailed natural-language description — better without IP-Adapter)
    - SDXL with IP-Adapter (short style + pose guide — reference does the work)

Pokémon accuracy design principles:
    1. Describe the species explicitly: colours, body shape, distinctive marks.
    2. Specify pose, action, and camera angle to control composition.
    3. Always include stencil-friendly style modifiers.
    4. Use strong negative prompts to prevent photorealism and complexity.
    5. For FLUX: include visual character description even with a reference
       (the reference is the image, the prompt reinforces it).
    6. Token economy: SDXL hard-caps at 77 CLIP tokens; FLUX handles longer.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Built-in character data ─────────────────────────────────────────────────────

# Detailed visual descriptions for accurate Pokémon generation.
# Format: (colours, body_description, distinctive_features)
_POKEMON_DESCRIPTIONS: dict[str, tuple[str, str, str]] = {
    "pikachu": (
        "bright yellow with brown stripes on its back and red circular cheek patches",
        "small chubby mouse-like body with large pointy ears tipped with black",
        "distinctive lightning bolt-shaped yellow-and-brown tail, expressive black eyes",
    ),
    "charizard": (
        "orange body with cream-white underside and wings, teal-blue wing membranes",
        "large bipedal dragon with broad wings, muscular body, long neck",
        "flame burning at tail tip, two small horns, powerful claws",
    ),
    "bulbasaur": (
        "blue-green teal body with dark teal spots, red eyes",
        "small quadruped with a large green plant bulb on its back",
        "bulb with thick veins, stubby legs, short snout",
    ),
    "squirtle": (
        "light blue body with pale yellow front, brown shell with dark edges",
        "small bipedal turtle with a rounded shell on its back",
        "curly tail, large round eyes, small ears, striped shell",
    ),
    "gengar": (
        "deep purple body with lighter purple belly, red eyes",
        "round wide-bodied ghost-type with a wide grinning mouth",
        "spiky back, stubby arms, pointed ears, shadowy form",
    ),
    "eevee": (
        "light brown fur with cream-white fluffy ruff at neck and tail tip",
        "small fox-like quadruped with large ears and big brown eyes",
        "fluffy bushy tail, small paws, cute round face",
    ),
    "snorlax": (
        "teal-green body with cream-white belly, small beady eyes",
        "enormous round-bodied bear-like creature, very large belly",
        "tiny hands and feet relative to body size, perpetually sleepy expression",
    ),
    "mewtwo": (
        "pale lavender-grey body with purple details, large purple eyes",
        "tall bipedal humanoid with a large head and long tail",
        "tube connecting from neck to back of skull, defined muscle structure",
    ),
    "umbreon": (
        "sleek black body with glowing yellow rings on ears, tail, and body",
        "cat-like quadruped with long ears and a lithe muscular body",
        "circular yellow ring markings, red eyes, smooth sleek form",
    ),
    "lucario": (
        "blue and black body with yellow spike on chest, red eyes",
        "tall bipedal jackal-like with dreadlock-like appendages from head",
        "aura sensors (head appendages), spike on chest and back of hands",
    ),
}

_DEFAULT_STYLE_FLUX = (
    "clean flat vector illustration, bold black outlines, high-contrast colours, "
    "white background, stencil-friendly simplified design, poster art style"
)

_DEFAULT_STYLE_SDXL = (
    "clean cartoon illustration, bold black outlines, vector art style, "
    "high contrast lighting, flat colour shapes, stencil-friendly composition"
)

_NEGATIVE_PROMPT = (
    "photorealistic, photograph, 3D render, gradient, complex texture, noise, "
    "blurry, watermark, signature, multiple characters, extra limbs, deformed, "
    "low quality, sketch, pencil drawing, detailed background, cluttered, "
    "anime style, chibi, cute kawaii"
)

# Camera angle variants for diversity generation
_CAMERA_ANGLES = [
    "front view, facing camera",
    "three-quarter view",
    "dynamic side view",
    "low angle looking up",
    "action pose, dynamic angle",
]

# Lighting variants for diversity
_LIGHTING = [
    "bright even studio lighting",
    "dramatic rim lighting",
    "high contrast backlit silhouette",
    "soft diffused lighting",
]

# Action variants per character type
_ACTIONS = [
    "standing upright, ready pose",
    "attacking, aggressive combat pose",
    "leaping through the air",
    "using signature move, energy burst",
    "running full speed",
]


class PokemonPromptBuilder:
    """Construct character-accurate stencil-friendly prompts.

    Supports loading extended character data from a JSON preset file so
    that new Pokémon descriptions can be added without code changes.

    Usage::

        builder = PokemonPromptBuilder()
        prompt = builder.build_flux_prompt(
            "Pikachu",
            action="bursting through a stained glass window",
            lighting="dramatic backlit",
        )
    """

    def __init__(self, presets_path: Optional[Path] = None) -> None:
        self._presets: dict = {}
        self._load_presets(presets_path)

    def _load_presets(self, path: Optional[Path]) -> None:
        """Load character presets from JSON if available."""
        candidates = [
            path,
            Path("prompts/pokemon_presets.json"),
            Path(__file__).parent.parent.parent / "prompts" / "pokemon_presets.json",
        ]
        for p in candidates:
            if p and p.exists():
                try:
                    with open(p) as f:
                        self._presets = json.load(f)
                    logger.debug("Loaded %d pokemon presets from %s", len(self._presets), p)
                    return
                except Exception as exc:
                    logger.warning("Failed to load presets from %s: %s", p, exc)

    def _get_character_info(self, pokemon_name: str) -> dict:
        """Return character info from presets or built-in descriptions."""
        name_lower = pokemon_name.lower().strip()

        # Preset file takes priority
        if name_lower in self._presets:
            return self._presets[name_lower]

        # Built-in descriptions
        if name_lower in _POKEMON_DESCRIPTIONS:
            colours, body, features = _POKEMON_DESCRIPTIONS[name_lower]
            return {"colours": colours, "body": body, "features": features}

        # Generic fallback — use the name itself
        return {
            "colours": "distinctive Pokémon colours",
            "body": f"{pokemon_name} Pokémon",
            "features": "iconic Pokémon character features",
        }

    def build_flux_prompt(
        self,
        pokemon_name: str,
        pose: Optional[str] = None,
        action: Optional[str] = None,
        lighting: Optional[str] = None,
        background: Optional[str] = "plain white background",
        style_override: Optional[str] = None,
        extra: Optional[str] = None,
    ) -> str:
        """Build a detailed character description for FLUX.1 text-to-image.

        FLUX.1 has strong prompt understanding — use full natural-language
        character descriptions for best character accuracy.

        Args:
            pokemon_name: Species name (e.g. "Pikachu").
            pose: Camera angle / body pose.
            action: What the character is doing.
            lighting: Lighting condition.
            background: Scene/background description.
            style_override: Replace the default visual style modifiers.
            extra: Any additional descriptor appended at the end.

        Returns:
            Full prompt string, ~80–150 words, optimised for FLUX.1.
        """
        info = self._get_character_info(pokemon_name)
        style = style_override or _DEFAULT_STYLE_FLUX

        parts = [
            f"{pokemon_name.title()} Pokémon character,",
            info["body"] + ",",
            info["colours"] + ",",
            info["features"] + ",",
        ]

        if action:
            parts.append(action + ",")
        elif pose:
            parts.append(pose + ",")
        else:
            parts.append("dynamic pose,")

        if lighting:
            parts.append(lighting + ",")

        if background:
            parts.append(background + ",")

        parts.append(style)

        if extra:
            parts.append(extra)

        return " ".join(parts)

    def build_sdxl_prompt(
        self,
        pokemon_name: str,
        pose: Optional[str] = None,
        action: Optional[str] = None,
        lighting: Optional[str] = None,
        style_override: Optional[str] = None,
        extra: Optional[str] = None,
    ) -> str:
        """Build a style-focused prompt for SDXL + IP-Adapter.

        When IP-Adapter is active, the reference image handles character
        appearance.  This prompt drives style, pose, and composition.
        Kept short to fit within SDXL's 77-token CLIP limit.

        Returns:
            Short prompt string (~25-40 tokens).
        """
        style = style_override or _DEFAULT_STYLE_SDXL
        parts = [f"{pokemon_name.title()} Pokémon,"]

        if action:
            parts.append(action + ",")
        elif pose:
            parts.append(pose + ",")

        if lighting:
            parts.append(lighting + ",")

        parts.append(style)

        if extra:
            parts.append(extra)

        prompt = " ".join(parts)
        return self._trim_to_sdxl_budget(prompt)

    def negative_prompt(self) -> str:
        """Standard negative prompt for stencil-friendly outputs."""
        return _NEGATIVE_PROMPT

    def diversity_variants(
        self,
        pokemon_name: str,
        count: int,
        provider: str = "flux",
    ) -> list[str]:
        """Return *count* prompt variants for candidate diversity generation.

        Cycles through combinations of camera angles, lighting, and actions.
        """
        build = self.build_flux_prompt if provider == "flux" else self.build_sdxl_prompt
        variants = []
        for i in range(count):
            pose = _CAMERA_ANGLES[i % len(_CAMERA_ANGLES)]
            light = _LIGHTING[i % len(_LIGHTING)]
            action = _ACTIONS[i % len(_ACTIONS)]
            variants.append(build(pokemon_name, pose=pose, action=action, lighting=light))
        return variants

    @staticmethod
    def _trim_to_sdxl_budget(prompt: str, max_tokens: int = 70) -> str:
        """Very rough token trimmer — drops trailing comma-clauses to fit budget."""
        import re
        tokens = re.findall(r"[A-Za-z0-9']+|[^\w\s]", prompt)
        if len(tokens) <= max_tokens:
            return prompt
        # Drop last clause at a time
        clauses = [c.strip() for c in prompt.split(",") if c.strip()]
        while clauses:
            candidate = ", ".join(clauses)
            toks = re.findall(r"[A-Za-z0-9']+|[^\w\s]", candidate)
            if len(toks) <= max_tokens:
                return candidate
            clauses.pop()
        return prompt[:200]  # last resort hard cut
