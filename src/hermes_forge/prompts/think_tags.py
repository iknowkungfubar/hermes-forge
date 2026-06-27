"""Think tag extraction utilities for models that emit <｜end▁of▁thinking｜>.

Some models (especially local ones) wrap their reasoning/thinking in
special tags like  ...  or  ... . This module extracts
that content so it can be handled according to ReasoningReplay policy.
"""

from __future__ import annotations

import re

_THINK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("think", re.compile(r"<think>(.*?)</think>", re.DOTALL)),
    ("reason", re.compile(r"<reason>(.*?)</reason>", re.DOTALL)),
    ("reasoning", re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL)),
]


def extract_think_tags(text: str) -> tuple[str, str]:
    """Extract think/reason content from model output.

    Returns (cleaned_text, think_content) where think_content is the
    concatenated content from all think-like tags found.

    If no tags are found, returns (text, "").
    """
    think_parts: list[str] = []
    cleaned = text

    for tag_name, pattern in _THINK_PATTERNS:
        matches = pattern.findall(cleaned)
        if matches:
            think_parts.extend(m.strip() for m in matches if m.strip())
            cleaned = pattern.sub("", cleaned).strip()

    combined_think = "\n".join(think_parts) if think_parts else ""

    # Clean up extra whitespace from tag removal
    cleaned = re.sub(r" +", " ", cleaned)
    return cleaned.strip(), combined_think


def has_think_tags(text: str) -> bool:
    """Check if text contains any think-like tags."""
    for _, pattern in _THINK_PATTERNS:
        if pattern.search(text):
            return True
    return False
