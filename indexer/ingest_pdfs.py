
from __future__ import annotations

import logging
import time
import warnings
from pathlib import Path
from typing import List, Dict, Any

from rich.console import Console
from rich.logging import RichHandler
from tqdm import tqdm
from openai import OpenAI

from backend.config import PDF_PATH, OPENAI_MODEL, require_api_key
from backend.retrieval import get_client, embed_texts
from indexer.manifest import load_manifest, save_manifest, file_fingerprint
from indexer.chunking import chunk_text
from indexer.pdf_ocr import extract_text_from_pdf
from graph.semantic_graph import upsert_document_node, add_document_edges
from concordance.terms import extract_candidate_terms
from concordance.index import append_occurrences

LOG_FILE = Path("logs/ingest.log")
console = Console()



# ----------------------------------------------------------------------------------
# String sanitization for safe indexing (Chroma / UTF-8)
# ----------------------------------------------------------------------------------

def _clean_string(value: Any) -> str:
    """Return a UTF-8 safe string with surrogate codepoints removed.

    Some PDFs contain broken Unicode (lone surrogates) that will cause
    downstream libraries (like Chroma's Rust bindings) to raise
    UnicodeEncodeError when encoding to UTF-8. This helper normalizes
    *any* input to a plain Python str, strips surrogate codepoints,
    and re-encodes with errors ignored so we never pass invalid
    sequences into external systems.
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        try:
            value = str(value)
        except Exception:
            value = ""
    if not value:
        return ""

    # Remove surrogate codepoints explicitly.
    filtered = "".join(ch for ch in value if not (0xD800 <= ord(ch) <= 0xDFFF))
    # Round-trip through UTF-8 to drop any remaining invalid sequences.
    return filtered.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")


def _clean_documents(docs: List[str]) -> List[str]:
    return [_clean_string(d) for d in docs]


def _clean_metadatas(metas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    cleaned: List[Dict[str, Any]] = []
    for meta in metas:
        new_meta: Dict[str, Any] = {}
        for k, v in meta.items():
            if isinstance(v, str):
                new_meta[k] = _clean_string(v)
            else:
                new_meta[k] = v
        cleaned.append(new_meta)
    return cleaned


# ----------------------------------------------------------------------------------
# Logging setup
# ----------------------------------------------------------------------------------


def _setup_logging() -> logging.Logger:
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            RichHandler(console=console, rich_tracebacks=True, show_time=False),
            logging.FileHandler(LOG_FILE, encoding="utf-8"),
        ],
    )

    logger = logging.getLogger("corpusflower.ingest")

    # Suppress noisy low-level warnings on the console, but still log them.
    def _warn_to_log(message, category, filename, lineno, file=None, line=None):
        msg = str(message)
        if "unknown widths" in msg.lower():
            human = (
                "PDF font width table is malformed or incomplete; text extraction "
                "may be slightly degraded, but ingestion will continue. "
                "See ingest.log for raw details."
            )
            logger.warning(human)
        else:
            logger.warning(f"{category.__name__}: {msg} ({filename}:{lineno})")

    warnings.showwarning = _warn_to_log  # type: ignore[assignment]

    return logger


logger = _setup_logging()


# ----------------------------------------------------------------------------------
# Core helpers
# ----------------------------------------------------------------------------------


def _get_chat_client() -> OpenAI:
    return OpenAI(api_key=require_api_key())


def _summarize_document(text: str, language: str, title: str) -> str:
    """Build a short summary of the document using the chat model.

    We only pass a truncated slice of the document to stay well under
    context limits. The summary is used both for human understanding
    and for coarse-grained semantic relationships between books.
    """
    if not text.strip():
        return f"Empty or unreadable document: {title}."

    client = _get_chat_client()

    snippet = text[:8000]
    prompt = (
        "You are building an internal catalog description for an enterprise document repository.\n"
        f"Document title: {title}\n"
        f"Detected language: {language}\n\n"
        "Below is an excerpt from the document. Write a concise 1–2 paragraph summary "
        "of its contents, focusing on key topics, entities, and themes that would help a researcher "
        "quickly understand what the document is about. Do not invent content that is not supported by the text.\n\n"
        "--- DOCUMENT EXCERPT ---\n"
        f"{snippet}\n"
        "--- END EXCERPT ---\n"
    )

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert research librarian and cataloger."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def _update_semantic_graph_for_doc(
    docs_col,
    doc_id: str,
    title: str,
    language: str,
    summary_embedding: List[float],
    max_neighbors: int = 6,
) -> None:
    """Update the semantic graph with relationships for a single document.

    Strategy:
    - Ensure the document has a node in the graph.
    - Query the docs collection using the summary embedding for this document.
    - Treat the nearest neighbors as semantically related works and add undirected
      edges between this doc and those neighbors, weighted by similarity.
    """
    # 1. Upsert the node itself.
    upsert_document_node(doc_id=doc_id, title=title, language=language)

    # 2. Query for similar document summaries.
    res = docs_col.query(
        query_embeddings=[summary_embedding],
        n_results=max_neighbors,
        include=["distances"],  # 'ids' is always returned; cannot be in include list
    )
    if not res or not res.get("ids"):
        return

    ids = res["ids"][0]
    dists = res["distances"][0]

    similar: List[tuple[str, float]] = []
    for other_id, dist in zip(ids, dists):
        if other_id == doc_id:
            continue
        # Convert Chroma distance into a similarity between 0 and 1.
        similarity = max(0.0, min(1.0, 1.0 - float(dist)))
        similar.append((other_id, similarity))

    if similar:
        add_document_edges(doc_id, similar)


# ----------------------------------------------------------------------------------
# Main ingest
# ----------------------------------------------------------------------------------


def ingest_pdfs() -> None:
    """Full ingestion pipeline with semantic-graph and concordance updates.

    - Load manifest and detect which PDFs are new or changed.
    - For each PDF to ingest:
        * Extract text (PyPDF2 + OCR fallback),
        * Chunk into overlapping segments,
        * Extract candidate terms per chunk (baseline concordance),
        * Summarize the document,
        * Embed all chunks + the summary in batches,
        * Upsert into Chroma collections,
        * Update the semantic graph based on doc-level similarity,
        * Append concordance occurrences for all chunk terms.
    - Save updated manifest.
    """
    pdf_root = Path(PDF_PATH)
    pdf_root.mkdir(parents=True, exist_ok=True)

    all_pdfs = sorted(
        [p for p in pdf_root.glob("*.pdf") if p.is_file()],
        key=lambda p: p.name.lower(),
    )
    if not all_pdfs:
        logger.info("No PDFs found to ingest. Put files into data/pdfs and run again.")
        return

    manifest = load_manifest()
    updated_manifest: Dict[str, Any] = {}

    client = get_client()
    chunks_col = client.get_or_create_collection(name="corpusflower_chunks")
    docs_col = client.get_or_create_collection(name="corpusflower_docs")

    # Determine which files actually need work
    to_ingest: List[Path] = []
    for path in all_pdfs:
        fp = file_fingerprint(path)
        updated_manifest[path.name] = fp
        if manifest.get(path.name) != fp:
            to_ingest.append(path)

    if not to_ingest:
        logger.info("Manifest up to date; no PDFs changed. Nothing to ingest.")
        save_manifest(updated_manifest)
        return

    logger.info(f"{len(to_ingest)} / {len(all_pdfs)} PDFs require ingestion.")

    start_time = time.time()

    for idx, path in enumerate(tqdm(to_ingest, desc="Ingesting PDFs", unit="pdf")):
        t0 = time.time()
        logger.info(f"Processing {path.name} ...")

        full_text, meta = extract_text_from_pdf(path)
        language = meta.get("language", "unknown")
        page_count = meta.get("page_count", None)

        if not full_text.strip():
            logger.warning(f"{path.name}: no text could be extracted (even with OCR). Skipping.")
            continue

        chunks = chunk_text(full_text)
        if not chunks:
            logger.warning(f"{path.name}: chunker returned no chunks. Skipping.")
            continue

        doc_id = path.name
        title = path.stem

        # Extract candidate terms per chunk for concordance + metadata
        chunk_texts: List[str] = []
        chunk_metas: List[Dict[str, Any]] = []
        concordance_batch: List[Dict[str, Any]] = []

        for c in chunks:
            c_text = c["text"]
            c_index = c["index"]
            terms = extract_candidate_terms(c_text, language=language)

            chunk_texts.append(c_text)
            chunk_meta = {
                "source": path.name,
                "title": title,
                "language": language,
                "chunk_index": c_index,
                # Do NOT store 'terms' list in Chroma metadata; it only allows primitives.
            }
            chunk_metas.append(chunk_meta)

            chunk_id = f"{doc_id}::chunk-{c_index}"
            for term in terms:
                concordance_batch.append(
                    {
                        "term": term,
                        "doc_id": doc_id,
                        "chunk_id": chunk_id,
                        "chunk_index": c_index,
                        "language": language,
                        "title": title,
                    }
                )

        # Prepare texts to embed: all chunk texts + summary
        summary_text = _summarize_document(full_text, language=language, title=title)
        all_texts = chunk_texts + [summary_text]

        embeddings = embed_texts(all_texts)
        if not embeddings:
            logger.warning(f"{path.name}: embedding call returned no vectors; skipping.")
            continue

        chunk_embeddings = embeddings[: len(chunk_texts)]
        summary_embedding = embeddings[-1]

        # Upsert document summary
        summary_metadata = {
            "source": _clean_string(path.name),
            "title": _clean_string(title),
            "language": _clean_string(language),
            "page_count": page_count,
        }

        docs_col.upsert(
            ids=[_clean_string(doc_id)],
            documents=_clean_documents([summary_text]),
            metadatas=_clean_metadatas([summary_metadata]),
            embeddings=[summary_embedding],
        )

        # Upsert chunks (now without 'terms' in metadata)
        chunk_ids = [f"{doc_id}::chunk-{c['index']}" for c in chunks]

        clean_chunk_ids = [_clean_string(cid) for cid in chunk_ids]
        clean_chunk_texts = _clean_documents(chunk_texts)
        clean_chunk_metas = _clean_metadatas(chunk_metas)

        chunks_col.upsert(
            ids=clean_chunk_ids,
            documents=clean_chunk_texts,
            metadatas=clean_chunk_metas,
            embeddings=chunk_embeddings,
        )

        # Update semantic graph relationships for this document.
        _update_semantic_graph_for_doc(
            docs_col=docs_col,
            doc_id=doc_id,
            title=title,
            language=language,
            summary_embedding=summary_embedding,
        )

        # Append concordance occurrences for this document.
        append_occurrences(concordance_batch)

        elapsed = time.time() - t0
        total_elapsed = time.time() - start_time
        avg_per_pdf = total_elapsed / max(1, idx + 1)
        remaining = avg_per_pdf * (len(to_ingest) - (idx + 1))

        logger.info(
            f"Finished {path.name} in {elapsed:.1f}s "
            f"(elapsed total {total_elapsed/60:.1f} min, "
            f"ETA {remaining/60:.1f} min for remaining PDFs)."
        )

    save_manifest(updated_manifest)
    total = time.time() - start_time
    logger.info(f"Ingest complete. Total time: {total/60:.1f} minutes.")


if __name__ == "__main__":
    ingest_pdfs()
