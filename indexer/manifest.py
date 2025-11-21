
from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Dict, Any

MANIFEST_PATH = Path("data/index/manifest.json")


def file_fingerprint(path: Path) -> Dict[str, Any]:
    """
    Compute a simple fingerprint for a file based on:
    - size in bytes
    - last modified time
    - sha256 hash

    This is robust enough for incremental ingest in a local library.
    """
    stat = path.stat()
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return {
        "size": stat.st_size,
        "mtime": stat.st_mtime,
        "sha256": h.hexdigest(),
    }


def load_manifest() -> Dict[str, Any]:
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_manifest(data: Dict[str, Any]) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
