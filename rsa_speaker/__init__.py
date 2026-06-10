"""Are LLMs good pragmatic speakers? Comparing LLM and RSA referential scores."""

from __future__ import annotations

__version__ = "0.1.0"

from rsa_speaker.corpus import (
    Furniture,
    People,
    parse_furniture,
    parse_people,
    parse_furniture_features,
    produce_prompt,
)

__all__ = [
    "Furniture",
    "People",
    "parse_furniture",
    "parse_people",
    "parse_furniture_features",
    "produce_prompt",
    "__version__",
]
