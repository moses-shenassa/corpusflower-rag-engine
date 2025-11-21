# CorpusFlower Architecture

CorpusFlower is a small, production-grade concordance + RAG engine designed to be:
- easy to read and reason about,
- simple to run locally,
- straightforward to extend for more advanced retrieval and analysis.

At a high level:

```text
PDFs
  │
  ▼
StemParser         (ingest, clean, chunk)
  │
  ▼
RootIndex          (embeddings + vector store)
  │
  ▼
PetalRetriever     (similarity + concordance)
  │
  ▼
BloomSynthesizer   (LLM-based synthesis)
  │
  ▼
Flask UI           (local research interface)
```

## Components

### 1. StemParser (Ingestion Layer)

Responsibilities:
- Walk a configured directory of PDFs.
- Extract raw text from each document.
- Normalize whitespace, handle simple cleaning tasks.
- Chunk long documents into passages of manageable size.
- Persist the passages to disk as structured JSON.

Typical implementation location:
- `indexer/ingest_pdfs.py`
- `indexer/metadata.py`

### 2. RootIndex (Vector Index Layer)

Responsibilities:
- Load passages produced by StemParser.
- Generate embeddings for each passage using an OpenAI embedding model.
- Store vectors and references in a local index (e.g., a simple on-disk structure or lightweight vector store).
- Expose similarity search: given a query embedding, return top-k relevant passages.

Typical implementation location:
- `backend/retrieval.py`
- `data/` (for on-disk index artifacts; this directory is usually gitignored).

### 3. PetalRetriever (Retrieval / Concordance Layer)

Responsibilities:
- Accept user queries: keyword searches, phrases, or natural-language questions.
- Convert queries into embedding space and/or use lexical filtering.
- Retrieve relevant passages from RootIndex.
- Optionally structure results as:
  - a simple list of passages,
  - a concordance (term with left/right context),
  - grouped results by document or section.

Typical implementation location:
- `backend/retrieval.py`
- `concordance/` for any concordance-specific helpers.

### 4. BloomSynthesizer (Synthesis Layer)

Responsibilities:
- Take retrieved passages from PetalRetriever.
- Apply prompt templates to generate:
  - summaries,
  - thematic overviews,
  - comparative analyses,
  - carefully grounded answers that cite or anchor to retrieved passages.
- Ensure outputs remain grounded in the corpus (avoiding free-floating hallucinations).

Typical implementation location:
- `backend/prompts.py`
- `backend/reasoning.py`

### 5. Flask UI (Presentation Layer)

Responsibilities:
- Provide an HTTP interface for:
  - entering queries,
  - viewing results,
  - triggering synthesis calls.
- Render server-side templates (and optionally light client-side enhancements) for a minimal, distraction-free interface.
- Serve as the main entry point for non-developer users.

Typical implementation location:
- `backend/main.py`
- `frontend/templates/` (if separated).

## Data Flow

1. **Index Time**
   - PDFs → StemParser → JSON passages → RootIndex builds embeddings and saves a local index.

2. **Query Time**
   - User query → PetalRetriever → uses RootIndex for similarity search → (optionally) passes results to BloomSynthesizer → returns results to Flask UI.

## Production-Grade Considerations

While CorpusFlower is intentionally compact, several best practices are followed:

- Clear separation of concerns (ingest vs index vs retrieval vs synthesis vs presentation).
- Environment-driven configuration (API keys, model names, and paths).
- Text-based, inspectable artifacts (JSON, logs, and simple on-disk indexes).
- Straightforward logging and extension points.

This makes the project well-suited both as:
- a personal research tool, and
- a portfolio example for RAG/concordance-style engineering work.
