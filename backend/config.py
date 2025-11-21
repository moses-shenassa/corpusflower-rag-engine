import os
from pathlib import Path
from dotenv import load_dotenv

# Resolve project root as parent of this file's directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOTENV_PATH = PROJECT_ROOT / ".env"

# Load environment variables from the .env at project root
load_dotenv(dotenv_path=DOTENV_PATH)


def require_api_key() -> str:
    """
    Ensure OPENAI_API_KEY is available.

    This function reloads the .env on each call so that changes on disk are picked up
    even if the process is long-lived (e.g., during development servers).
    """
    load_dotenv(dotenv_path=DOTENV_PATH)
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set. "
                           "Create a .env file in the project root with OPENAI_API_KEY=...")
    return key


OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

INDEX_PATH = os.getenv("BOB_INDEX_PATH", "./data/index")
PDF_PATH = os.getenv("BOB_PDF_PATH", "./data/pdfs")
