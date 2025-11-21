
from __future__ import annotations

import re
from typing import List, Set


# A very small English stopword list to avoid flooding the concordance
_STOPWORDS: Set[str] = {
    "the", "and", "or", "of", "to", "in", "on", "for", "with", "by", "at",
    "is", "it", "this", "that", "a", "an", "as", "from", "be", "are", "was",
    "were", "but", "not", "into", "about", "over", "under", "between",
    "within", "without", "through",
}


def _normalize_token(tok: str) -> str:
    """Normalize a raw token for concordance purposes.

    - Lowercase
    - Strip surrounding punctuation
    - Keep internal apostrophes / hyphens (e.g., yaweh-name, spirit’s)
    """
    tok = tok.strip()
    # Remove leading/trailing non-word chars but preserve inner - and '
    tok = re.sub(r"^[^\w']+|[^\w']+$", "", tok)
    return tok.lower()


def extract_candidate_terms(text: str, language: str | None = None) -> List[str]:
    """Extract candidate terms from a chunk of text.

    This is a deliberately simple, language-agnostic heuristic:
    - Split on whitespace
    - Normalize tokens
    - Filter out very short tokens and basic stopwords
    - Keep a unique set per chunk

    The goal is to provide a *baseline concordance layer* without
    introducing extra heavy dependencies. You can later upgrade this
    function to use spaCy, custom lexicons, or LLM-based extraction.
    """
    if not text:
        return []

    tokens = re.split(r"\s+", text)
    terms: Set[str] = set()

    for tok in tokens:
        norm = _normalize_token(tok)
        if not norm:
            continue
        # Skip short tokens
        if len(norm) < 4:
            continue
        # Skip basic English stopwords if language is unknown/en
        if (language in (None, "unknown", "en")) and norm in _STOPWORDS:
            continue
        terms.add(norm)

    return sorted(terms)
