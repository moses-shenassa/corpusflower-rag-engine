from __future__ import annotations

from typing import Dict, Any

from langdetect import detect, LangDetectException


def detect_language(text: str) -> str:
    """
    Lightweight language detection for a document or chunk.

    We only need a rough label (e.g. 'en', 'es', 'fr', 'la'); it's fine if this
    occasionally misfires, because the embedding model handles cross-language
    similarity and CorpusFlower's prompts handle cross-language synthesis.
    """
    sample = text[:2000]
    try:
        lang = detect(sample)
        return lang
    except LangDetectException:
        return "unknown"


def guess_tradition_from_text(text: str) -> str:
    """
    Extremely simple heuristic to tag a text with a rough domain/context label.

    This is intentionally conservative. It does not attempt to be authoritative;
    it only gives CorpusFlower a hint about which stream(s) of thought a given PDF might
    belong to, so that it can speak more clearly about context.
    """
    lowered = text.lower()

    if any(k in lowered for k in ["kabbalah", "sefirot", "sephiroth", "tiferet", "binah", "yesod"]):
        return "Jewish mysticism / Kabbalah (heuristic)"
    if any(k in lowered for k in ["psalm", "psalms", "jesus", "mary", "saint", "angel"]):
        return "Christian / folk Catholic devotional / liturgical material (heuristic)"
    if any(k in lowered for k in ["hoodoo", "conjure", "rootwork", "mojo", "jack ball"]):
        return "African American hoodoo / conjure (heuristic)"
    if any(k in lowered for k in ["pentacle", "pentagram", "solomon", "goetia",  "seal of"]):
        return "Solomonic / ceremonial esoteric text (heuristic)"
    if any(k in lowered for k in ["thelema", "crowley", "a\'a", "ordo templi orientis"]):
        return "Thelemic / modern ceremonial (heuristic)"
    if any(k in lowered for k in ["golden dawn", "lbrp", "rose cross", "shemesh", "mizrah"]):
        return "Hermetic Order of the Golden Dawn (heuristic)"
    if any(k in lowered for k in ["spiritism", "espiritismo", "mesa blanca"]):
        return "Spiritism / Espiritismo (heuristic)"

    return "Unknown / mixed domain (heuristic)"


SYMBOL_KEYWORDS = [
    "figure",
    "seal",
    "sigil",
    "pentacle",
    "diagram",
    "plate",
    "illustration",
    "engraving",
    "talisman",
    "amulet",
]


def detect_symbol_hint(text: str) -> bool:
    """
    Very crude symbol detection: if the text mentions any of a small list of
    keywords (figure, sigil, pentacle, seal, etc.), we mark the chunk / doc
    as having a 'symbol_hint'.

    CorpusFlower's prompts then explain how to use this:
    - He describes the symbol verbally,
    - And tells the reader to consult the cited page in the original PDF.
    """
    lowered = text.lower()
    return any(k in lowered for k in SYMBOL_KEYWORDS)


def build_document_metadata(
    raw_text: str,
    source_file: str,
    title: str | None = None,
) -> Dict[str, Any]:
    """
    Compute metadata for a full document.

    We store:
    - source_id / source_file
    - title
    - language (rough)
    - tradition (rough)
    - symbol_hint (bool)
    """
    language = detect_language(raw_text)
    tradition = guess_tradition_from_text(raw_text)
    symbol_hint = detect_symbol_hint(raw_text)

    return {
        "source_id": source_file,
        "source_file": source_file,
        "title": title or source_file,
        "language": language,
        "tradition": tradition,
        "symbol_hint": symbol_hint,
    }


def build_chunk_metadata(
    doc_metadata: Dict[str, Any],
    page_range: str,
    chunk_index: int,
    chunk_text: str,
) -> Dict[str, Any]:
    """
    Build metadata for a chunk by extending the parent document metadata.

    We add:
    - page_range
    - chunk_index
    - chunk-specific language (if different)
    - symbol_hint if the chunk itself appears to reference symbols/figures.
    """  # noqa: D401
    md = dict(doc_metadata)
    md["page_range"] = page_range
    md["chunk_index"] = chunk_index

    lang = detect_language(chunk_text)
    if lang != "unknown":
        md["language"] = lang

    if detect_symbol_hint(chunk_text):
        md["symbol_hint"] = True

    return md
