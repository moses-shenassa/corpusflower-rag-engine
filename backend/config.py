import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve project root as parent of this file's directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"

# Load environment variables from the .env at project root
load_dotenv(dotenv_path=DOTENV_PATH)


def require_api_key() -> str:
    """Return the OpenAI API key or raise a clear error.

    This helper centralizes API key checks so both the backend and the
    indexer fail fast with a helpful message if configuration is missing.
    """
    # Re-load in case the environment changed after import (e.g. in REPL)
    load_dotenv(dotenv_path=DOTENV_PATH)
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. "
            "Create a .env file in the project root with OPENAI_API_KEY=..."
        )
    return key


# Chat / completion model used for answering questions and summarization
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Embedding model used for vector indexing and retrieval
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Persistent Chroma index location
# Primary env vars are CORPUSFLOWER_*; legacy names are still accepted as a fallback.
INDEX_PATH = Path(
    os.getenv("CORPUSFLOWER_INDEX_PATH")
    or os.getenv("LEGACY_INDEX_PATH", "./data/index")
)

# Location of PDFs to ingest
PDF_PATH = Path(
    os.getenv("CORPUSFLOWER_PDF_PATH")
    or os.getenv("LEGACY_PDF_PATH", "./data/pdfs")
)
