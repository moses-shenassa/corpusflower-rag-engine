"""
Graph-style RAG retrieval for CorpusFlower.

This module adds a simple "graph-ish" retrieval strategy on top of the
existing dual-collection layout:

  - docs collection: 1 row per document (summary-level)
  - chunks collection: multiple rows per document (fine-grained passages)

Given a question, we:
  1. Embed the question once.
  2. Query the docs collection to find the most relevant documents.
  3. For each of those documents, query the chunks collection with a
     metadata filter (where={"doc_id": ...}) to get the best passages
     inside that doc.
  4. Merge, de-duplicate and sort the passages globally by distance.

This is not a full graph database, but it behaves like a tiny semantic
graph: documents are "nodes", passages are "attached detail nodes", and
we perform a two-hop walk: question → docs → passages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from .retrieval import embed_text, get_client, _get_collections


@dataclass
class GraphRAGResult:
    """Optional helper for debugging / inspection.

    Not currently used by the HTTP API, but kept here so you can import it
    in notebooks or future tools.
    """

    question: str
    doc_summaries: List[Dict[str, Any]]
    passages: List[Dict[str, Any]]


def _normalize_query_embedding(question: str) -> List[float]:
    """Embed the question text and return a single embedding vector.

    We always embed a *single* question string, so we expect a 1-element
    list of embeddings from `embed_text`.
    """
    embeddings = embed_text([question])
    if not embeddings:
        raise RuntimeError("embed_text() returned no embeddings for question.")
    return embeddings[0]


def graph_rag_retrieve(
    question: str,
    n_doc_summaries: int = 6,
    n_passages: int = 18,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Hierarchical retrieval over docs + chunks collections.

    Parameters
    ----------
    question:
        Natural language question from the user.
    n_doc_summaries:
        How many top documents (by summary-level similarity) to consider.
    n_passages:
        Total number of passages to return across all selected documents.

    Returns
    -------
    doc_summaries:
        List of records of the form:
            {
                "id": <doc_id>,
                "text": <summary or representative text>,
                "metadata": { ... },
                "distance": <float>,
            }
    passages:
        List of records of the form:
            {
                "id": <chunk_id>,
                "text": <chunk text>,
                "metadata": { ... },
                "distance": <float>,
            }
    """
    q = question.strip()
    if not q:
        return [], []

    client = get_client()
    chunks_col, docs_col = _get_collections(client)

    query_emb = _normalize_query_embedding(q)

    # 1) Query the document-level collection to find candidate docs.
    #
    # IMPORTANT: Chroma 0.4+/0.5 no longer accepts "ids" in the `include=`
    # parameter. IDs are *always* returned in the response, even if not
    # explicitly requested. The allowed values are:
    #   ["documents", "embeddings", "metadatas", "distances", "uris", "data"]
    #
    # So we only request the fields we actually need.
    doc_res = docs_col.query(
        query_embeddings=[query_emb],
        n_results=n_doc_summaries,
        include=["documents", "metadatas", "distances"],
    )

    # Chroma returns lists-of-lists for each field, one outer element per
    # query. We only made ONE query, so we always index at [0].
    doc_ids = doc_res.get("ids", [[]])[0] if doc_res.get("ids") else []
    doc_docs = doc_res.get("documents", [[]])[0] if doc_res.get("documents") else []
    doc_metas = doc_res.get("metadatas", [[]])[0] if doc_res.get("metadatas") else []
    doc_dists = doc_res.get("distances", [[]])[0] if doc_res.get("distances") else []

    doc_summaries: List[Dict[str, Any]] = []
    for doc_id, doc_text, meta, dist in zip(doc_ids, doc_docs, doc_metas, doc_dists):
        doc_summaries.append(
            {
                "id": doc_id,
                "text": doc_text,
                "metadata": meta or {},
                "distance": float(dist) if dist is not None else 0.0,
            }
        )

    # 2) For each candidate document, query the chunks collection using a
    #    metadata filter so we only search within that document's chunks.
    #
    # We use a simple per-doc budget heuristic so that we don't let a single
    # huge grimoire monopolize all the retrieved passages.
    all_passages: List[Dict[str, Any]] = []
    if doc_ids:
        # Rough per-doc budget; we ensure at least 1 result per doc.
        per_doc = max(1, n_passages // max(1, len(doc_ids)))

        for doc_id in doc_ids:
            chunk_res = chunks_col.query(
                query_embeddings=[query_emb],
                n_results=per_doc,
                where={"doc_id": doc_id},
                include=["documents", "metadatas", "distances"],
            )

            chunk_ids = (
                chunk_res.get("ids", [[]])[0] if chunk_res.get("ids") else []
            )
            chunk_docs = (
                chunk_res.get("documents", [[]])[0]
                if chunk_res.get("documents")
                else []
            )
            chunk_metas = (
                chunk_res.get("metadatas", [[]])[0]
                if chunk_res.get("metadatas")
                else []
            )
            chunk_dists = (
                chunk_res.get("distances", [[]])[0]
                if chunk_res.get("distances")
                else []
            )

            for cid, ctext, cmeta, cdist in zip(
                chunk_ids, chunk_docs, chunk_metas, chunk_dists
            ):
                all_passages.append(
                    {
                        "id": cid,
                        "text": ctext,
                        "metadata": cmeta or {},
                        "distance": float(cdist) if cdist is not None else 0.0,
                    }
                )

    # 3) De-duplicate and globally sort the passages by distance.
    #
    # It's possible that some chunks appear multiple times if the underlying
    # store has overlapping results; we keep the best (lowest distance).
    dedup: Dict[str, Dict[str, Any]] = {}
    for p in all_passages:
        pid = str(p.get("id"))
        if pid not in dedup or p["distance"] < dedup[pid]["distance"]:
            dedup[pid] = p

    passages_sorted = sorted(dedup.values(), key=lambda x: x["distance"])
    passages_final = passages_sorted[:n_passages]

    return doc_summaries, passages_final


def graph_rag_retrieve_result(
    question: str,
    n_doc_summaries: int = 6,
    n_passages: int = 18,
) -> GraphRAGResult:
    """Convenience wrapper that returns a typed result object.

    This is mostly for REPL / notebook exploration; the HTTP API uses the
    bare `(doc_summaries, passages)` tuple from `graph_rag_retrieve`.
    """
    doc_summaries, passages = graph_rag_retrieve(
        question=question,
        n_doc_summaries=n_doc_summaries,
        n_passages=n_passages,
    )
    return GraphRAGResult(
        question=question,
        doc_summaries=doc_summaries,
        passages=passages,
    )
