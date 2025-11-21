
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List

CONCORDANCE_PATH = Path("data/concordance/concordance.json")


def load_concordance() -> Dict[str, Any]:
    """Load the concordance JSON structure, or return an empty skeleton.

    Schema:
        {
            "occurrences": [
                {
                    "term": str,
                    "doc_id": str,
                    "chunk_id": str,
                    "chunk_index": int,
                    "language": str | None,
                    "title": str | None,
                },
                ...
            ]
        }
    """
    if CONCORDANCE_PATH.exists():
        with CONCORDANCE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"occurrences": []}


def append_occurrences(new_occurrences: List[Dict[str, Any]]) -> None:
    """Append a batch of occurrences to the concordance file.

    This is called once per document during ingestion to avoid excessive
    file IO inside tight loops.
    """
    if not new_occurrences:
        return

    data = load_concordance()
    occ = data.setdefault("occurrences", [])
    occ.extend(new_occurrences)

    CONCORDANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONCORDANCE_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
