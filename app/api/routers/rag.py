"""Retrieval-Augmented Generation (RAG) router."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.api.routers.search import get_retriever
from reviewiq.rag.rag_engine import ReviewRAGEngine

router = APIRouter()

_rag_engine: ReviewRAGEngine | None = None


def get_rag_engine() -> ReviewRAGEngine:
    global _rag_engine
    if _rag_engine is None:
        retriever = get_retriever()
        _rag_engine = ReviewRAGEngine(retriever=retriever)
    return _rag_engine


class AskRequest(BaseModel):
    query: str = Field(..., min_length=2, description="Natural language question")
    top_k: int = Field(5, ge=1, le=10, description="Evidence reviews to retrieve")


class CitationItem(BaseModel):
    citation_id: int
    doc_id: str | int
    text: str
    relevance_score: float


class AskResponse(BaseModel):
    query: str
    answer: str
    citations: list[CitationItem]
    model_name: str
    is_grounded: bool


@router.post("/query", response_model=AskResponse)
async def ask_reviews(req: AskRequest) -> AskResponse:
    """Answer question strictly grounded in verified customer review citations."""
    engine = get_rag_engine()
    response = engine.answer_query(req.query, top_k=req.top_k)

    citations = [
        CitationItem(
            citation_id=c.citation_id,
            doc_id=c.doc_id,
            text=c.text,
            relevance_score=round(c.relevance_score, 4),
        )
        for c in response.citations
    ]

    return AskResponse(
        query=response.query,
        answer=response.answer,
        citations=citations,
        model_name=response.model_name,
        is_grounded=response.is_grounded,
    )
