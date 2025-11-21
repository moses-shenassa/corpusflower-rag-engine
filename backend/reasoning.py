from __future__ import annotations

import logging
from typing import List

from openai import OpenAI

from .config import require_api_key, OPENAI_MODEL
from .prompts import SYSTEM_PROMPT, build_answer_prompt, build_reflection_prompt

logger = logging.getLogger(__name__)


def _get_client() -> OpenAI:
    return OpenAI(api_key=require_api_key())


def answer_question_with_rag(
    question: str,
    context_blocks: List[str],
) -> str:
    """
    Core reasoning function.

    1. Build an answer prompt that:
       - Includes all context blocks as numbered sources.
       - Explains CorpusFlower's expectations (citations, cross-language, symbol handling).
    2. Generate a first-pass answer.
    3. Ask the model to self-critique and revise the answer for:
       - Better synthesis,
       - Proper citations,
       - Safety and clarity.

    This structure is intentionally explicit and verbose so that you (the developer)
    can read and learn from the prompt shaping.
    """
    client = _get_client()

    # First pass
    answer_prompt = build_answer_prompt(context_blocks, question)
    logger.info("Requesting first-pass answer from LLM.")
    first = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": answer_prompt},
        ],
        temperature=0.3,
    )
    draft = first.choices[0].message.content or ""

    # Reflection pass
    reflection_prompt = build_reflection_prompt(draft, question)
    logger.info("Requesting self-critique / refinement pass from LLM.")
    second = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": reflection_prompt},
        ],
        temperature=0.2,
    )
    final = second.choices[0].message.content or draft

    return final.strip()
