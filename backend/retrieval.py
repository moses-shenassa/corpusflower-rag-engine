
from __future__ import annotations

import logging
from typing import List, Dict, Any, Tuple

import chromadb
from openai import OpenAI

from .config import (
    require_api_key,
    OPENAI_MODEL,
    OPENAI_EMBEDDING_MODEL,
    INDEX_PATH,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------
# OpenAI / Chroma clients
# --------------------------------------------------------------------------------------


def get_openai_client() -> OpenAI:
    """
    Lightweight helper to create an OpenAI client using the API key resolved
    via our config module.
    """
    api_key = require_api_key()
    return OpenAI(api_key=api_key)


def get_client() -> chromadb.PersistentClient:
    """
    Create (or reuse) a persistent Chroma client rooted at INDEX_PATH.

    We use a persistent client so that repeated runs of the app share the same
    vector store built by the ingestion script.
    """
    return chromadb.PersistentClient(path=INDEX_PATH)


# --------------------------------------------------------------------------------------
# Embeddings
# --------------------------------------------------------------------------------------


def embed_text(texts: List[str], batch_size: int = 64) -> List[List[float]]:
    """
    Embed a list of strings using the configured OpenAI embedding model.

    This function is used in two places:
    - During ingestion (to embed chunks and document summaries)
    - At query time (to embed the user's question for retrieval)

    It batches requests so that we never exceed the max-tokens-per-request
    limit on the embeddings endpoint.
    """
    if not texts:
        return []

    client = get_openai_client()
    all_embeddings: List[List[float]] = []

    # Simple batching by number of inputs. With our chunk sizes this is
    # comfortably under the 300k-token per-request limit.
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        resp = client.embeddings.create(
            model=OPENAI_EMBEDDING_MODEL,
            input=batch,
        )
        for item in resp.data:
            all_embeddings.append(item.embedding)

    return all_embeddings


# --------------------------------------------------------------------------------------
# Retrieval helpers
# --------------------------------------------------------------------------------------


def _get_collections(client: chromadb.PersistentClient):
    """
    Helper to get (or lazily create) the two collections used by CorpusFlower:

    - bob_chunks: page / logical chunks (fine-grained retrieval)
    - bob_docs: document-level summaries (coarser topical retrieval)
    """
    chunks = client.get_or_create_collection(name="bob_chunks")
    docs = client.get_or_create_collection(name="bob_docs")
    return chunks, docs


def retrieve_for_question(
    question: str,
    n_doc_summaries: int = 5,
    n_passages: int = 15,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Retrieve both coarse document summaries and fine-grained passages
    for a given question.

    Returns:
        doc_summaries: list of {id, text, metadata, distance}
        passages: list of {id, text, metadata, distance}
    """
    client = get_client()
    chunks_col, docs_col = _get_collections(client)

    # Embed the question once and use the same embedding in both queries.
    q_embedding = embed_text([question])[0]

    # Doc-level retrieval (coarse topical)
    docs_res = docs_col.query(
        query_embeddings=[q_embedding],
        n_results=n_doc_summaries,
        include=["documents", "metadatas", "distances"],
    )

    doc_summaries: List[Dict[str, Any]] = []
    if docs_res and docs_res.get("documents"):
        for doc_id, text, meta, dist in zip(
            docs_res["ids"][0],
            docs_res["documents"][0],
            docs_res["metadatas"][0],
            docs_res["distances"][0],
        ):
            doc_summaries.append(
                {"id": doc_id, "text": text, "metadata": meta, "distance": dist}
            )

    # Passage-level retrieval (fine local detail)
    chunks_res = chunks_col.query(
        query_embeddings=[q_embedding],
        n_results=n_passages,
        include=["documents", "metadatas", "distances"],
    )

    passages: List[Dict[str, Any]] = []
    if chunks_res and chunks_res.get("documents"):
        for cid, text, meta, dist in zip(
            chunks_res["ids"][0],
            chunks_res["documents"][0],
            chunks_res["metadatas"][0],
            chunks_res["distances"][0],
        ):
            passages.append(
                {"id": cid, "text": text, "metadata": meta, "distance": dist}
            )

    return doc_summaries, passages


# --------------------------------------------------------------------------------------
# Context block construction
# --------------------------------------------------------------------------------------


def build_context_blocks(
    doc_summaries: List[Dict[str, Any]],
    passages: List[Dict[str, Any]],
    max_blocks: int = 12,
) -> List[str]:
    """
    Turn retrieved summaries + passages into a list of context blocks
    for the LLM.

    We keep formatting intentionally lightweight and conversational so
    answers don't sound like a wiki dump, but each block is still
    clearly labeled and citable.
    """
    blocks: List[str] = []

    # First, document summaries
    for i, doc in enumerate(doc_summaries, start=1):
        meta = doc.get("metadata") or {}
        src = meta.get("source", "unknown source")
        lang = meta.get("language", "unknown")
        title = meta.get("title", src)
        dist = doc.get("distance")

        header = f"Source #{i}: {title} (language={lang}, distance={dist:.3f} from query)"
        block = f"[DOCUMENT SUMMARY]\n{header}\n\n{doc['text']}"
        blocks.append(block)

    # Then, detailed passages
    for j, p in enumerate(passages, start=1):
        meta = p.get("metadata") or {}
        src = meta.get("source", "unknown source")
        lang = meta.get("language", "unknown")
        page = meta.get("page_start")
        chunk_idx = meta.get("chunk_index")
        dist = p.get("distance")

        loc_bits = []
        if page is not None:
            loc_bits.append(f"page {page}")
        if chunk_idx is not None:
            loc_bits.append(f"chunk {chunk_idx}")
        loc_str = ", ".join(loc_bits) if loc_bits else "location unknown"

        header = (
            f"Passage #{j} from {src} ({loc_str}, language={lang}, "
            f"distance={dist:.3f} from query)"
        )
        block = f"[PASSAGE]\n{header}\n\n{p['text']}"
        blocks.append(block)

        if len(blocks) >= max_blocks:
            break

    return blocks
