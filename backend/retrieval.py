from __future__ import annotations

import logging
import textwrap
from typing import Any, Dict, Iterable, List, Tuple

import chromadb
from chromadb.api import ClientAPI
from openai import OpenAI

from .config import (
    OPENAI_MODEL,
    OPENAI_EMBEDDING_MODEL,
    INDEX_PATH,
    require_api_key,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _sanitize_text(text: str) -> str:
    """
    Remove surrogate code points and coerce non-string values to string.

    This prevents:
      - UnicodeEncodeError: 'utf-8' codec can't encode surrogates
      - JSON encoding failures in the OpenAI client
      - UTF-8 issues when Chroma stores documents
    """
    if not isinstance(text, str):
        text = str(text)

    # Strip surrogate codepoints (UTF-16 halves) that are illegal in UTF-8
    return "".join(ch for ch in text if not (0xD800 <= ord(ch) <= 0xDFFF))


def _sanitize_texts(texts: Iterable[str]) -> List[str]:
    return [_sanitize_text(t) for t in texts]


# ---------------------------------------------------------------------------
# OpenAI + Chroma client helpers
# ---------------------------------------------------------------------------

_openai_client: OpenAI | None = None
_chroma_client: ClientAPI | None = None

# Collection names (must match what ingest_pdfs.py uses)
CHUNK_COLLECTION_NAME = "corpusflower_chunks"
DOCS_COLLECTION_NAME = "corpusflower_docs"


def get_openai_client() -> OpenAI:
    """
    Lazily create and cache a single OpenAI client.

    Uses require_api_key() from backend.config to ensure an API key is set
    in the environment, then relies on the OpenAI library's default loading.
    """
    global _openai_client
    if _openai_client is not None:
        return _openai_client

    # This should assert that OPENAI_API_KEY is configured and/or set env vars.
    require_api_key()

    # Let the OpenAI client read from environment variables
    _openai_client = OpenAI()
    return _openai_client


def get_client() -> ClientAPI:
    """
    Return a shared persistent Chroma client using INDEX_PATH from config.
    This is what both the indexer and backend should use.
    """
    global _chroma_client
    if _chroma_client is not None:
        return _chroma_client

    _chroma_client = chromadb.PersistentClient(path=str(INDEX_PATH))
    return _chroma_client


def _get_collections(client: ClientAPI | None = None):
    """
    Helper that returns (chunks_collection, docs_collection).

    Used by:
      - Simple RAG retrieval helpers in this module
      - GraphRAG (backend.graphrag) via its imports
    """
    if client is None:
        client = get_client()

    chunks_col = client.get_or_create_collection(name=CHUNK_COLLECTION_NAME)
    docs_col = client.get_or_create_collection(name=DOCS_COLLECTION_NAME)
    return chunks_col, docs_col


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def embed_text(texts: List[str]) -> List[List[float]]:
    """
    Embed a batch of texts using the configured OpenAI embedding model.

    NOTE:
        - The name is singular (embed_text) for historical reasons; it takes
          a *list* of texts and returns a list of embeddings.
        - To avoid hitting the OpenAI `max_tokens_per_request` limit, we send
          the inputs in smaller sub-batches.
    """
    client = get_openai_client()
    clean_inputs = _sanitize_texts(texts)

    # Conservative batch size: 32 texts per request.
    # Even if each text is near the per-input token limit, 32 keeps us safely
    # under the 300k tokens-per-request limit enforced by the API.
    MAX_TEXTS_PER_BATCH = 32

    all_embeddings: List[List[float]] = []
    for i in range(0, len(clean_inputs), MAX_TEXTS_PER_BATCH):
        batch = clean_inputs[i : i + MAX_TEXTS_PER_BATCH]
        resp = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=batch,
        )
        all_embeddings.extend(item.embedding for item in resp.data)

    return all_embeddings


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Backward-compatible alias used by older / patched indexer code."""
    return embed_text(texts)



def embed_texts(texts: List[str]) -> List[List[float]]:
    """Backward-compatible alias used by older / patched indexer code."""
    return embed_text(texts)


# ---------------------------------------------------------------------------
# Semantic retrieval helpers
# ---------------------------------------------------------------------------


def retrieve_semantic_passages(
    question: str,
    *,
    top_k_passages: int = 24,
    top_k_doc_summaries: int = 8,
    client: ClientAPI | None = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Retrieve both document-level summaries and chunk-level passages for a query.

    Returns:
        (doc_summaries, passages)

        doc_summaries: [
          {
            "doc_id": str,
            "summary": str,
            "metadata": {...},
            "distance": float,
          },
          ...
        ]

        passages: [
          {
            "chunk_id": str,
            "text": str,
            "metadata": {...},
            "distance": float,
          },
          ...
        ]
    """
    if client is None:
        client = get_client()

    chunks_col, docs_col = _get_collections(client)

    # Embed the question once
    q_embedding = embed_text([question])[0]

    # Chunk-level retrieval
    chunk_res = chunks_col.query(
        query_embeddings=[q_embedding],
        n_results=top_k_passages,
    )

    passages: List[Dict[str, Any]] = []
    for doc_id, metadata, text, distance in zip(
        chunk_res.get("ids", [[]])[0],
        chunk_res.get("metadatas", [[]])[0],
        chunk_res.get("documents", [[]])[0],
        chunk_res.get("distances", [[]])[0],
    ):
        passages.append(
            {
                "chunk_id": doc_id,
                "text": text,
                "metadata": metadata or {},
                "distance": float(distance),
            }
        )

    # Doc-level retrieval (summaries)
    doc_res = docs_col.query(
        query_embeddings=[q_embedding],
        n_results=top_k_doc_summaries,
    )

    doc_summaries: List[Dict[str, Any]] = []
    for doc_id, metadata, text, distance in zip(
        doc_res.get("ids", [[]])[0],
        doc_res.get("metadatas", [[]])[0],
        doc_res.get("documents", [[]])[0],
        doc_res.get("distances", [[]])[0],
    ):
        doc_summaries.append(
            {
                "doc_id": doc_id,
                "summary": text,
                "metadata": metadata or {},
                "distance": float(distance),
            }
        )

    return doc_summaries, passages


# ---------------------------------------------------------------------------
# Context block construction
# ---------------------------------------------------------------------------


def build_context_blocks(
    doc_summaries: List[Dict[str, Any]],
    passages: List[Dict[str, Any]],
    max_blocks: int = 12,
) -> List[str]:
    """
    Turn doc_summaries + passages into a list of string context blocks
    which can be fed directly into an LLM prompt.

    This is what backend.main._run_graph_rag imports and uses.
    """

    blocks: List[str] = []

    # 1) Add document-level summaries first (coarse context)
    for doc in doc_summaries:
        meta = doc.get("metadata") or {}
        title = meta.get("title") or meta.get("file_name") or "Document"
        source = meta.get("source") or meta.get("file_path") or ""
        summary = doc.get("summary") or ""

        block = textwrap.dedent(
            f"""
            [DOCUMENT SUMMARY]
            Title: {title}
            Source: {source}
            Summary:
            {summary.strip()}
            """
        ).strip()
        blocks.append(block)

        if len(blocks) >= max_blocks:
            return blocks

    # 2) Then add the highest-scoring passages (fine-grained evidence)
    for p in passages:
        meta = p.get("metadata") or {}
        title = meta.get("title") or meta.get("file_name") or "Document"
        loc = meta.get("location") or meta.get("page_label") or ""
        text = p.get("text") or ""

        header_extra = f" — {loc}" if loc else ""
        block = textwrap.dedent(
            f"""
            [PASSAGE]
            From: {title}{header_extra}

            {text.strip()}
            """
        ).strip()
        blocks.append(block)

        if len(blocks) >= max_blocks:
            break

    return blocks
