
from __future__ import annotations

from typing import List, Dict, Any


def chunk_text(
    text: str,
    max_chars: int = 1200,
    overlap: int = 200,
) -> List[Dict[str, Any]]:
    """
    Simple character-based chunker with overlapping windows.

    This is deliberately straightforward:
    - It avoids giant wall-of-text chunks that are hard for the model,
    - But it keeps enough local context to preserve meaning.

    Returns a list of dicts with:
    - "text": the chunk string
    - "index": 0-based index of the chunk inside the document
    """
    chunks: List[Dict[str, Any]] = []
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    n = len(text)
    if n == 0:
        return chunks

    start = 0
    idx = 0

    while start < n:
        end = min(start + max_chars, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append({"text": chunk, "index": idx})
            idx += 1
        if end == n:
            break
        # Step forward, keeping some overlap to help continuity.
        start = max(0, end - overlap)

    return chunks
