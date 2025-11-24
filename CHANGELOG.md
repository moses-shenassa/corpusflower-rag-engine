# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2025-11-19
### Added
- Initial public release of **CorpusFlower**, evolved from an earlier internal prototype.
- PDF ingestion pipeline (StemParser) for walking a directory of PDFs, extracting and chunking text.
- Local embedding and vector index layer (RootIndex) for semantic similarity search.
- Retrieval layer (PetalRetriever) for concordance-style lookup and context window exploration.
- Synthesis layer (BloomSynthesizer) wired to OpenAI models for summaries and thematic analysis.
- Flask-based local research UI for querying the corpus and inspecting results.
- Dark-academia botanical branding, README, and documentation (`architecture.md`, `branding.md`, `configuration.md`).
- MIT license and initial project metadata suitable for professional portfolio use.

### Notes
- This release is focused on clarity, readability, and serving as a production-quality reference implementation
  of a small RAG-style concordance engine.
