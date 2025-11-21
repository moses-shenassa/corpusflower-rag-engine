
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple

GRAPH_PATH = Path("data/graph/semantic_graph.json")


@dataclass
class GraphNode:
    id: str
    type: str  # e.g. "document", later maybe "concept", "symbol"
    title: str | None = None
    language: str | None = None


@dataclass
class GraphEdge:
    source: str
    target: str
    weight: float  # semantic similarity score between 0 and 1


def _load_raw() -> Dict[str, Any]:
    if GRAPH_PATH.exists():
        with GRAPH_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {"nodes": {}, "edges": []}


def _save_raw(data: Dict[str, Any]) -> None:
    GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GRAPH_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def upsert_document_node(doc_id: str, title: str, language: str | None = None) -> None:
    """Create or update a document node in the semantic graph."""
    data = _load_raw()
    nodes = data.setdefault("nodes", {})
    nodes[doc_id] = {
        "id": doc_id,
        "type": "document",
        "title": title,
        "language": language,
    }
    _save_raw(data)


def add_document_edges(doc_id: str, similar_docs: List[Tuple[str, float]]) -> None:
    """Add undirected edges between doc_id and each similar doc.

    similar_docs is a list of (other_doc_id, similarity_score) where
    similarity_score is between 0 and 1 (higher = more similar).
    """
    if not similar_docs:
        return

    data = _load_raw()
    edges: List[Dict[str, Any]] = data.setdefault("edges", [])

    # Build a set of normalized pairs we already have so we don't duplicate.
    existing = {
        (min(e["source"], e["target"]), max(e["source"], e["target"]))
        for e in edges
    }

    for other_id, sim in similar_docs:
        if other_id == doc_id:
            continue
        key = (min(doc_id, other_id), max(doc_id, other_id))
        if key in existing:
            continue
        edges.append(
            {
                "source": doc_id,
                "target": other_id,
                "weight": float(sim),
            }
        )
        existing.add(key)

    _save_raw(data)


def get_graph_snapshot() -> Dict[str, Any]:
    """Return the entire semantic graph as a plain dict.

    This is primarily for debugging / visualization.
    """
    return _load_raw()
