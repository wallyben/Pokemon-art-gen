"""
Tests for pokemon_stencil.image_gen.prompt_engine.
"""

from __future__ import annotations

import pytest

from pokemon_stencil.image_gen.prompt_engine import (
    CAMERA_ANGLE_TOKENS,
    CLIP_TOKEN_LIMIT,
    LIGHTING_TOKENS,
    PromptEngine,
    _count_tokens,
)


# ─────────────────────────────────────────────────────────────────────────────
# _count_tokens
# ─────────────────────────────────────────────────────────────────────────────

class TestCountTokens:
    def test_empty_string(self):
        assert _count_tokens("") == 0

    def test_single_word(self):
        assert _count_tokens("Pikachu") == 1

    def test_multiple_words(self):
        count = _count_tokens("bold outlines flat colors")
        assert count == 4

    def test_comma_separated(self):
        # Each word + each comma = tokens
        count = _count_tokens("a, b, c")
        assert count >= 3  # at least the words

    def test_returns_integer(self):
        assert isinstance(_count_tokens("test"), int)


# ─────────────────────────────────────────────────────────────────────────────
# PromptEngine.build
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptEngineBuild:
    def setup_method(self):
        self.engine = PromptEngine()

    def test_contains_pokemon_name(self):
        prompt = self.engine.build("Pikachu")
        assert "Pikachu" in prompt

    def test_contains_style_tokens(self):
        prompt = self.engine.build("Pikachu")
        assert "bold black outlines" in prompt

    def test_includes_extras(self):
        prompt = self.engine.build("Pikachu", extras="surfing a wave")
        assert "surfing a wave" in prompt

    def test_includes_camera_angle(self):
        prompt = self.engine.build("Pikachu", camera_angle="front view")
        assert "front view" in prompt

    def test_includes_lighting(self):
        prompt = self.engine.build("Pikachu", lighting="bright even lighting")
        assert "bright even lighting" in prompt

    def test_within_clip_limit(self):
        prompt = self.engine.build("Pikachu")
        assert _count_tokens(prompt) <= CLIP_TOKEN_LIMIT

    def test_very_long_extras_compressed(self):
        long_extras = ", ".join([f"extra clause {i}" for i in range(50)])
        prompt = self.engine.build("Pikachu", extras=long_extras)
        assert _count_tokens(prompt) <= CLIP_TOKEN_LIMIT

    def test_returns_string(self):
        assert isinstance(self.engine.build("Charizard"), str)

    def test_no_extras_no_camera_no_lighting(self):
        prompt = self.engine.build("Bulbasaur")
        assert "Bulbasaur" in prompt
        assert len(prompt) > 0


# ─────────────────────────────────────────────────────────────────────────────
# PromptEngine.compress
# ─────────────────────────────────────────────────────────────────────────────

class TestPromptEngineCompress:
    def setup_method(self):
        self.engine = PromptEngine()

    def test_short_prompt_unchanged(self):
        short = "Pikachu, bold outlines"
        assert self.engine.compress(short) == short

    def test_long_prompt_compressed_within_limit(self):
        long_prompt = ", ".join([f"clause {i}" for i in range(100)])
        compressed = self.engine.compress(long_prompt)
        assert _count_tokens(compressed) <= CLIP_TOKEN_LIMIT

    def test_first_clause_preserved(self):
        prompt = "Pikachu character, " + ", ".join([f"c{i}" for i in range(80)])
        compressed = self.engine.compress(prompt)
        assert compressed.startswith("Pikachu character")

    def test_returns_string(self):
        assert isinstance(self.engine.compress("test"), str)


# ─────────────────────────────────────────────────────────────────────────────
# PromptEngine.negative_prompt
# ─────────────────────────────────────────────────────────────────────────────

class TestNegativePrompt:
    def test_returns_string(self):
        engine = PromptEngine()
        assert isinstance(engine.negative_prompt(), str)

    def test_contains_photorealistic(self):
        engine = PromptEngine()
        assert "photorealistic" in engine.negative_prompt()

    def test_custom_negative_prompt(self):
        engine = PromptEngine(negative_prompt="bad stuff")
        assert engine.negative_prompt() == "bad stuff"


# ─────────────────────────────────────────────────────────────────────────────
# Diversity token lists
# ─────────────────────────────────────────────────────────────────────────────

class TestDiversityTokens:
    def test_camera_angle_tokens_is_list(self):
        assert isinstance(CAMERA_ANGLE_TOKENS, list)
        assert len(CAMERA_ANGLE_TOKENS) >= 2

    def test_lighting_tokens_is_list(self):
        assert isinstance(LIGHTING_TOKENS, list)
        assert len(LIGHTING_TOKENS) >= 2

    def test_camera_angle_tokens_are_strings(self):
        assert all(isinstance(t, str) for t in CAMERA_ANGLE_TOKENS)

    def test_lighting_tokens_are_strings(self):
        assert all(isinstance(t, str) for t in LIGHTING_TOKENS)
