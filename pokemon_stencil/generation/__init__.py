"""
pokemon_stencil.generation — provider-agnostic image generation layer.

Public API::

    from pokemon_stencil.generation import (
        GenerationRequest,
        GenerationResult,
        get_provider,
        CandidateRunner,
        PokemonPromptBuilder,
    )
"""

from pokemon_stencil.generation.base import GenerationRequest, GenerationResult
from pokemon_stencil.generation.provider_factory import get_provider
from pokemon_stencil.generation.candidate_runner import CandidateRunner
from pokemon_stencil.generation.prompt_builder import PokemonPromptBuilder

__all__ = [
    "GenerationRequest",
    "GenerationResult",
    "get_provider",
    "CandidateRunner",
    "PokemonPromptBuilder",
]
