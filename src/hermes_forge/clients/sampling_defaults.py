"""Per-model sampling defaults for known local models.

Maps model name patterns to recommended sampling parameters.
Used by client adapters when recommended_sampling=True.
"""

from __future__ import annotations

from typing import Any

MODEL_SAMPLING_DEFAULTS: dict[str, dict[str, Any]] = {
    # Ministral / Mistral family
    "ministral": {
        "temperature": 0.3,
        "top_p": 0.9,
        "min_p": 0.05,
        "repeat_penalty": 1.1,
    },
    "mistral": {
        "temperature": 0.3,
        "top_p": 0.9,
        "min_p": 0.05,
    },
    # Llama family
    "llama": {
        "temperature": 0.6,
        "top_p": 0.9,
        "min_p": 0.05,
    },
    "llama-3": {
        "temperature": 0.6,
        "top_p": 0.9,
        "min_p": 0.05,
    },
    "llama-2": {
        "temperature": 0.5,
        "top_p": 0.9,
        "min_p": 0.05,
    },
    # Qwen family
    "qwen": {
        "temperature": 0.3,
        "top_p": 0.9,
        "min_p": 0.05,
        "repeat_penalty": 1.05,
    },
    "qwen2": {
        "temperature": 0.3,
        "top_p": 0.9,
        "min_p": 0.05,
    },
    "qwen3": {
        "temperature": 0.3,
        "top_p": 0.9,
        "min_p": 0.05,
    },
    # DeepSeek family
    "deepseek": {
        "temperature": 0.3,
        "top_p": 0.9,
        "min_p": 0.05,
    },
    # Gemma family
    "gemma": {
        "temperature": 0.5,
        "top_p": 0.9,
        "min_p": 0.05,
    },
    "gemma-2": {
        "temperature": 0.5,
        "top_p": 0.9,
        "min_p": 0.05,
    },
    # Phi family
    "phi": {
        "temperature": 0.3,
        "top_p": 0.9,
        "min_p": 0.05,
    },
    # Default fallback
    "default": {
        "temperature": 0.5,
        "top_p": 0.9,
        "min_p": 0.05,
        "repeat_penalty": 1.1,
    },
}


def apply_sampling_defaults(
    model: str,
    strict: bool = False,
) -> dict[str, Any]:
    """Apply recommended sampling defaults for a model name.

    Args:
        model: Model name string (e.g., "qwen3:30b", "ministral-3:8b").
        strict: If True, raise UnsupportedModelError if no defaults found.

    Returns a dict of sampling parameters.

    Raises:
        ValueError: If strict=True and no defaults match.
    """
    model_lower = model.lower()

    # Check for exact match first
    if model_lower in MODEL_SAMPLING_DEFAULTS:
        return dict(MODEL_SAMPLING_DEFAULTS[model_lower])

    # Check by prefix (longest prefix wins)
    matched = ""
    for key in MODEL_SAMPLING_DEFAULTS:
        if key == "default":
            continue
        if model_lower.startswith(key) and len(key) > len(matched):
            matched = key

    if matched:
        return dict(MODEL_SAMPLING_DEFAULTS[matched])

    if strict:
        raise ValueError(
            f"No recommended sampling defaults registered for model {model!r}. "
            f"Either add an entry to MODEL_SAMPLING_DEFAULTS "
            f"or drop recommended_sampling=True."
        )

    return dict(MODEL_SAMPLING_DEFAULTS["default"])
