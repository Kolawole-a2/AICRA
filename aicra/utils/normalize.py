"""Normalization utilities for malware family mapping."""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

# Pre-compile common regex patterns at module level
_PUNCTUATION_PATTERN = re.compile(r"[^\w\s-]")
_WHITESPACE_PATTERN = re.compile(r"\s+")
_HYPHEN_PATTERN = re.compile(r"-+")


@lru_cache(maxsize=1000)
def _normalize_cached(raw_family: str) -> str:
    """
    Internal cached normalization function.

    Args:
        raw_family: Raw family name from data

    Returns:
        Normalized family name
    """
    if not raw_family or not isinstance(raw_family, str):
        return "unknown"

    # Step 1: Unicode normalization (NFKC)
    normalized = unicodedata.normalize("NFKC", raw_family)

    # Step 2: Case folding
    normalized = normalized.casefold()

    # Step 3: Remove punctuation (keep alphanumeric, spaces, hyphens)
    normalized = _PUNCTUATION_PATTERN.sub("", normalized)

    # Step 4: Normalize whitespace
    normalized = _WHITESPACE_PATTERN.sub(" ", normalized)

    # Step 5: Normalize hyphens
    normalized = _HYPHEN_PATTERN.sub("-", normalized)

    # Step 6: Strip and collapse
    normalized = normalized.strip()

    # Step 7: Handle empty result
    if not normalized:
        return "unknown"

    return normalized


class FamilyNormalizer:
    """Normalizes malware family names for consistent mapping."""

    def __init__(self):
        # Pre-compile common regex patterns
        self._punctuation_pattern = _PUNCTUATION_PATTERN
        self._whitespace_pattern = _WHITESPACE_PATTERN
        self._hyphen_pattern = _HYPHEN_PATTERN

    def normalize(self, raw_family: str) -> str:
        """
        Normalize a raw malware family name.

        Args:
            raw_family: Raw family name from data

        Returns:
            Normalized family name
        """
        return _normalize_cached(raw_family)

    def normalize_batch(self, raw_families: list[str]) -> list[str]:
        """
        Normalize a batch of family names.

        Args:
            raw_families: List of raw family names

        Returns:
            List of normalized family names
        """
        return [self.normalize(family) for family in raw_families]

    def get_variants(self, canonical_family: str) -> set[str]:
        """
        Generate common variants of a canonical family name.

        Args:
            canonical_family: Canonical family name

        Returns:
            Set of possible variants
        """
        variants = {canonical_family}

        # Add normalized version
        variants.add(self.normalize(canonical_family))

        # Add common substitutions
        substitutions = {
            " ": ["-", "_", ""],
            "-": ["_", " ", ""],
            "_": ["-", " ", ""],
        }

        for char, replacements in substitutions.items():
            if char in canonical_family:
                for replacement in replacements:
                    variant = canonical_family.replace(char, replacement)
                    variants.add(variant)
                    variants.add(self.normalize(variant))

        return variants


def normalize_family_name(raw_family: str) -> str:
    """
    Convenience function for single family normalization.

    Args:
        raw_family: Raw family name

    Returns:
        Normalized family name
    """
    normalizer = FamilyNormalizer()
    return normalizer.normalize(raw_family)


def normalize_family_batch(raw_families: list[str]) -> list[str]:
    """
    Convenience function for batch family normalization.

    Args:
        raw_families: List of raw family names

    Returns:
        List of normalized family names
    """
    normalizer = FamilyNormalizer()
    return normalizer.normalize_batch(raw_families)


def create_alias_mapping(canonical_families: list[str]) -> dict[str, str]:
    """
    Create alias mapping from canonical families.

    Args:
        canonical_families: List of canonical family names

    Returns:
        Dictionary mapping aliases to canonical names
    """
    normalizer = FamilyNormalizer()
    alias_map = {}

    for canonical in canonical_families:
        variants = normalizer.get_variants(canonical)
        for variant in variants:
            alias_map[variant] = canonical

    return alias_map
