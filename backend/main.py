from __future__ import annotations

import logging
from typing import Any, Dict

from fastapi import FastAPI
from pydantic import BaseModel

from .graphrag import graph_rag_retrieve
from .reasoning import answer_question_with_rag

logger = logging.getLogger(__name__)

app = FastAPI(title="CorpusFlower RAG API (Graph-RAG enabled)")


class AnswerRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    answer: str
    raw_context: Dict[str, Any]


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str


def _run_graph_rag(question: str) -> Dict[str, Any]:
    """Shared retrieval + answer generation pipeline.

    This helper is used by both /answer and /api/ask so that all chat entry
    points go through the same Graph-RAG logic.
    """
    question = question.strip()
    if not question:
        return {
            "answer": "Ask me something about the documents you have indexed.",
            "raw_context": {},
        }

    # Graph-aware retrieval -> doc_summaries + passages
    doc_summaries, passages = graph_rag_retrieve(
        question,
        n_doc_summaries=6,
        n_passages=18,
    )

    # Build context blocks using the existing helper from retrieval.py
    from .retrieval import build_context_blocks  # type: ignore

    context_blocks = build_context_blocks(doc_summaries, passages)
    answer = answer_question_with_rag(question, context_blocks)

    return {
        "answer": answer,
        "raw_context": {
            "doc_summaries": doc_summaries,
            "passages": passages,
        },
    }


@app.post("/answer", response_model=AnswerResponse)
async def answer(req: AnswerRequest) -> AnswerResponse:
    """Primary chat endpoint for CorpusFlower.

    Returns both the final answer and the raw retrieval context so you can
    introspect what CorpusFlower actually used to reason.
    """
    result = _run_graph_rag(req.question)
    return AnswerResponse(**result)


@app.post("/api/ask", response_model=AskResponse)
async def api_ask(req: AskRequest) -> AskResponse:
    """Backward-compatible endpoint for the existing frontend.

    The frontend expects to POST to /api/ask with:

        { "question": "..." }

    and receive:

        { "answer": "..." }

    We reuse the same Graph-RAG pipeline under the hood but only return the
    answer field to keep the contract simple.
    """
    result = _run_graph_rag(req.question)
    return AskResponse(answer=result["answer"])
