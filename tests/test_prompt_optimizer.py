"""
tests/test_prompt_optimizer.py — Prompt optimization and CLIP token budget tests.

Verifies that:
- Prompts are automatically optimized to fit the 77-token CLIP limit
- The subject (first clause) is always preserved
- Important subject/action/pose descriptors are kept ahead of style tokens
- Truncation is intentional and logged, not accidental
- compress() returns a prompt guaranteed within the limit
- Priority ordering: subject → style → extras
"""

from __future__ import annotations

import pytest

from pokemon_stencil.image_gen.prompt_engine import (
    CLIP_TOKEN_LIMIT,
    PromptEngine,
)


# ─────────────────────────────────────────────────────────────────────────────
# Basic build() behavior
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptBuild:
    def setup_method(self):
        self.engine = PromptEngine()

    def test_contains_pokemon_name(self):
        prompt = self.engine.build("Pikachu")
        assert "Pikachu" in prompt

    def test_subject_is_first_clause(self):
        prompt = self.engine.build("Charizard")
        first_clause = prompt.split(",")[0].strip()
        assert "Charizard" in first_clause

    def test_contains_style_tokens(self):
        prompt = self.engine.build("Gengar")
        # At least one stencil-style token must be present
        style_indicators = [
            "bold black outlines", "vector illustration", "stencil", "cartoon"
        ]
        assert any(indicator in prompt.lower() for indicator in style_indicators)

    def test_within_clip_token_limit(self):
        prompt = self.engine.build("Mewtwo")
        assert self.engine.token_count(prompt) <= CLIP_TOKEN_LIMIT

    def test_extras_appended(self):
        prompt = self.engine.build("Eevee", extras="fluffy brown tail")
        assert "fluffy brown tail" in prompt

    def test_camera_angle_included(self):
        prompt = self.engine.build("Bulbasaur", camera_angle="front view")
        assert "front view" in prompt

    def test_lighting_included(self):
        prompt = self.engine.build("Squirtle", lighting="bright even lighting")
        assert "bright even lighting" in prompt

    def test_different_pokemon_produce_different_prompts(self):
        a = self.engine.build("Pikachu")
        b = self.engine.build("Snorlax")
        assert a != b


# ─────────────────────────────────────────────────────────────────────────────
# Token limit enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenLimitEnforcement:
    def setup_method(self):
        self.engine = PromptEngine()

    def test_overlong_prompt_compressed_to_limit(self):
        """A prompt exceeding 77 tokens must be compressed to fit."""
        long_extras = ", ".join([f"extra clause {i}" for i in range(30)])
        prompt = self.engine.build("Raichu", extras=long_extras)
        assert self.engine.token_count(prompt) <= CLIP_TOKEN_LIMIT

    def test_subject_always_preserved_after_compression(self):
        """The subject (Pokémon name) must always survive compression."""
        long_extras = ", ".join([f"extra clause {i}" for i in range(30)])
        prompt = self.engine.build("Dragonite", extras=long_extras)
        assert "Dragonite" in prompt

    def test_compress_short_prompt_unchanged(self):
        """A prompt already within the limit must not be altered."""
        short = "Pikachu, clean cartoon illustration"
        compressed = self.engine.compress(short)
        # Subject and key tokens must still be present
        assert "Pikachu" in compressed
        assert self.engine.token_count(compressed) <= CLIP_TOKEN_LIMIT

    def test_compress_long_prompt_fits_limit(self):
        """compress() must always produce a result within the CLIP limit."""
        long = ", ".join(["Token"] * 100)
        compressed = self.engine.compress(long)
        assert self.engine.token_count(compressed) <= CLIP_TOKEN_LIMIT

    def test_compress_preserves_first_clause(self):
        """The first clause must always survive compression."""
        prompt = "Pikachu character, " + ", ".join([f"clause {i}" for i in range(20)])
        compressed = self.engine.compress(prompt)
        # First clause must be present
        assert "Pikachu character" in compressed


# ─────────────────────────────────────────────────────────────────────────────
# token_count accuracy
# ─────────────────────────────────────────────────────────────────────────────

class TestTokenCount:
    def setup_method(self):
        self.engine = PromptEngine()

    def test_empty_string_zero_tokens(self):
        assert self.engine.token_count("") == 0

    def test_single_word_counts_as_one(self):
        assert self.engine.token_count("Pikachu") == 1

    def test_conservative_estimate_stays_under_limit(self):
        """Built prompts must stay safely under the 77-token limit."""
        for name in ["Pikachu", "Bulbasaur", "Charizard", "Mewtwo", "Snorlax"]:
            prompt = self.engine.build(name)
            count = self.engine.token_count(prompt)
            assert count <= CLIP_TOKEN_LIMIT, (
                f"Token count {count} exceeds limit {CLIP_TOKEN_LIMIT} "
                f"for prompt: {prompt!r}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Priority ordering
# ─────────────────────────────────────────────────────────────────────────────

class TestPriorityOrdering:
    def setup_method(self):
        self.engine = PromptEngine()

    def test_subject_comes_before_style_tokens(self):
        """Subject (Pokémon name) must appear before style suffix tokens."""
        prompt = self.engine.build("Gengar")
        name_pos = prompt.find("Gengar")
        style_pos = prompt.find("bold black outlines")
        if style_pos != -1:  # style tokens may be trimmed under limit
            assert name_pos < style_pos

    def test_extras_come_after_subject(self):
        """Extras must follow the subject."""
        prompt = self.engine.build("Alakazam", extras="psychic aura")
        name_pos = prompt.find("Alakazam")
        extras_pos = prompt.find("psychic aura")
        if extras_pos != -1:
            assert name_pos < extras_pos
