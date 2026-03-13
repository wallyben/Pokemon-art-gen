"""
tests/test_prompt_builder.py — PokemonPromptBuilder tests.

No GPU, no API, no external dependencies beyond the project itself.
"""

from __future__ import annotations

import pytest

from pokemon_stencil.generation.prompt_builder import PokemonPromptBuilder


class TestFluxPrompt:
    def test_basic_prompt_contains_name(self):
        builder = PokemonPromptBuilder()
        prompt = builder.build_flux_prompt("Pikachu")
        assert "pikachu" in prompt.lower()

    def test_prompt_contains_action(self):
        builder = PokemonPromptBuilder()
        prompt = builder.build_flux_prompt("Pikachu", action="bursting through a window")
        assert "bursting through a window" in prompt

    def test_prompt_contains_lighting(self):
        builder = PokemonPromptBuilder()
        prompt = builder.build_flux_prompt("Pikachu", lighting="dramatic backlit")
        assert "dramatic backlit" in prompt

    def test_prompt_contains_style(self):
        builder = PokemonPromptBuilder()
        prompt = builder.build_flux_prompt("Pikachu")
        assert "stencil" in prompt.lower()

    def test_known_pokemon_uses_description(self):
        builder = PokemonPromptBuilder()
        prompt = builder.build_flux_prompt("Pikachu")
        # Should include character-specific description from built-in data
        assert "yellow" in prompt.lower() or "mouse" in prompt.lower()

    def test_unknown_pokemon_still_works(self):
        builder = PokemonPromptBuilder()
        prompt = builder.build_flux_prompt("Mythical_Unknown_Pokemon_XYZ")
        assert "Mythical_Unknown_Pokemon_XYZ" in prompt
        assert len(prompt) > 20

    def test_extra_text_appended(self):
        builder = PokemonPromptBuilder()
        prompt = builder.build_flux_prompt("Pikachu", extra="UNIQUE_MARKER_TEXT")
        assert "UNIQUE_MARKER_TEXT" in prompt

    def test_charizard_has_flame(self):
        builder = PokemonPromptBuilder()
        prompt = builder.build_flux_prompt("Charizard")
        assert "flame" in prompt.lower() or "dragon" in prompt.lower() or "fire" in prompt.lower()

    def test_gengar_has_purple(self):
        builder = PokemonPromptBuilder()
        prompt = builder.build_flux_prompt("Gengar")
        assert "purple" in prompt.lower()


class TestSDXLPrompt:
    def test_sdxl_prompt_is_shorter_than_flux(self):
        builder = PokemonPromptBuilder()
        flux_prompt = builder.build_flux_prompt("Pikachu")
        sdxl_prompt = builder.build_sdxl_prompt("Pikachu")
        # SDXL should be more concise (token budget)
        assert len(sdxl_prompt) < len(flux_prompt)

    def test_sdxl_contains_name(self):
        builder = PokemonPromptBuilder()
        prompt = builder.build_sdxl_prompt("Pikachu")
        assert "pikachu" in prompt.lower()

    def test_sdxl_contains_style(self):
        builder = PokemonPromptBuilder()
        prompt = builder.build_sdxl_prompt("Pikachu")
        assert "stencil" in prompt.lower() or "cartoon" in prompt.lower()

    def test_sdxl_stays_under_token_budget(self):
        """SDXL prompts must stay well under 77 CLIP tokens."""
        import re
        builder = PokemonPromptBuilder()
        for name in ["Pikachu", "Charizard", "Mewtwo", "Umbreon", "Lucario"]:
            prompt = builder.build_sdxl_prompt(name)
            tokens = re.findall(r"[A-Za-z0-9']+|[^\w\s]", prompt)
            assert len(tokens) <= 70, (
                f"{name} SDXL prompt has {len(tokens)} tokens (limit 70): {prompt}"
            )


class TestNegativePrompt:
    def test_negative_prompt_present(self):
        builder = PokemonPromptBuilder()
        neg = builder.negative_prompt()
        assert len(neg) > 10

    def test_negative_prompt_contains_photorealistic(self):
        builder = PokemonPromptBuilder()
        assert "photorealistic" in builder.negative_prompt()

    def test_negative_prompt_contains_watermark(self):
        builder = PokemonPromptBuilder()
        assert "watermark" in builder.negative_prompt()


class TestDiversityVariants:
    def test_returns_correct_count(self):
        builder = PokemonPromptBuilder()
        variants = builder.diversity_variants("Pikachu", count=5)
        assert len(variants) == 5

    def test_single_variant(self):
        builder = PokemonPromptBuilder()
        variants = builder.diversity_variants("Pikachu", count=1)
        assert len(variants) == 1

    def test_variants_are_different(self):
        builder = PokemonPromptBuilder()
        variants = builder.diversity_variants("Pikachu", count=5)
        # At least some variants should differ
        assert len(set(variants)) > 1

    def test_all_contain_pokemon_name(self):
        builder = PokemonPromptBuilder()
        for variant in builder.diversity_variants("Charizard", count=4):
            assert "charizard" in variant.lower()

    def test_flux_vs_sdxl_style(self):
        builder = PokemonPromptBuilder()
        flux_variants = builder.diversity_variants("Pikachu", count=3, provider="flux")
        sdxl_variants = builder.diversity_variants("Pikachu", count=3, provider="sdxl")
        # FLUX prompts should be longer due to character description
        avg_flux = sum(len(v) for v in flux_variants) / len(flux_variants)
        avg_sdxl = sum(len(v) for v in sdxl_variants) / len(sdxl_variants)
        assert avg_flux > avg_sdxl

    def test_ten_variants_no_crash(self):
        builder = PokemonPromptBuilder()
        variants = builder.diversity_variants("Pikachu", count=10)
        assert len(variants) == 10

    def test_large_count_no_crash(self):
        builder = PokemonPromptBuilder()
        variants = builder.diversity_variants("Pikachu", count=20)
        assert len(variants) == 20


class TestStainedGlassPikachu:
    """The exact use-case from the requirements: aggressive stained-glass Pikachu."""

    def test_stained_glass_prompt_complete(self):
        builder = PokemonPromptBuilder()
        prompt = builder.build_flux_prompt(
            "Pikachu",
            action="bursting aggressively through a stained glass window, "
                   "shattering coloured glass fragments flying outward",
            lighting="dramatic backlit silhouette, coloured light rays through broken glass",
            background="dark background with coloured light shafts",
            extra="intense aggressive expression, dynamic action pose",
        )
        assert "pikachu" in prompt.lower()
        assert "stained glass" in prompt.lower()
        assert "aggressive" in prompt.lower() or "burst" in prompt.lower()
        assert len(prompt) > 100

    def test_stained_glass_is_non_empty_negative(self):
        builder = PokemonPromptBuilder()
        neg = builder.negative_prompt()
        assert neg
        assert "photorealistic" in neg


class TestPresetLoading:
    def test_load_builtin_pikachu(self):
        builder = PokemonPromptBuilder()
        info = builder._get_character_info("pikachu")
        assert "colours" in info
        assert "body" in info
        assert "features" in info
        assert "yellow" in info["colours"]

    def test_case_insensitive_lookup(self):
        builder = PokemonPromptBuilder()
        info_lower = builder._get_character_info("pikachu")
        info_upper = builder._get_character_info("PIKACHU")
        assert info_lower == info_upper

    def test_unknown_returns_fallback(self):
        builder = PokemonPromptBuilder()
        info = builder._get_character_info("nonexistent_pokemon_xyz")
        assert "colours" in info
        assert "body" in info
        assert "features" in info
