"""Search and entity extraction API router."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from reviewiq.retrieval.hybrid_search import HybridRetriever, SearchResult
from reviewiq.retrieval.ner import LaptopNER

router = APIRouter()

# Shared lazy instances
_retriever: HybridRetriever | None = None
_ner: LaptopNER | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
        # Seed demo corpus if not yet indexed
        demo_texts = [
            "Battery lasts 14 hours easily. Amazing battery performance and fast charging.",
            "Screen display is vibrant colorful OLED with 120Hz refresh rate and bright HDR.",
            "Keyboard is responsive and great for typing code. Sturdy aluminum chassis build.",
            "Terrible thermal cooling, fans get extremely loud and laptop gets super hot.",
            "Blazing fast performance with 16GB RAM and RTX 4060 graphics.",
            "Affordable budget laptop with great specs for students and office work.",
        ]
        demo_ids = [f"demo_{i}" for i in range(len(demo_texts))]
        _retriever.index(demo_texts, doc_ids=demo_ids)
    return _retriever


def get_ner() -> LaptopNER:
    global _ner
    if _ner is None:
        _ner = LaptopNER()
    return _ner


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search query string")
    top_k: int = Field(5, ge=1, le=50, description="Number of results to return")
    rerank: bool = Field(False, description="Whether to apply neural CrossEncoder reranking")


class SearchItem(BaseModel):
    rank: int
    doc_id: str | int
    text: str
    score: float
    metadata: dict[str, Any] = {}


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: list[SearchItem]


class NERRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Input text for hardware entity extraction")


class NEREntity(BaseModel):
    text: str
    label: str
    start_char: int
    end_char: int


class NERResponse(BaseModel):
    text: str
    entities: list[NEREntity]
    grouped: dict[str, list[str]]


@router.post("/query", response_model=SearchResponse)
async def search_reviews(req: SearchRequest) -> SearchResponse:
    """Execute hybrid search combining TF-IDF and dense embeddings with RRF."""
    retriever = get_retriever()
    results = retriever.search(req.query, final_k=req.top_k, rerank=req.rerank)
    items = [
        SearchItem(
            rank=r.rank,
            doc_id=r.doc_id,
            text=r.text,
            score=round(r.score, 4),
            metadata=r.metadata,
        )
        for r in results
    ]
    return SearchResponse(query=req.query, total_results=len(items), results=items)


@router.post("/ner", response_model=NERResponse)
async def extract_entities(req: NERRequest) -> NERResponse:
    """Extract laptop brands, CPU, GPU, RAM, screen size, and hardware entities."""
    ner = get_ner()
    spans = ner.extract_entities(req.text)
    grouped = ner.extract_entities_grouped(req.text)
    items = [
        NEREntity(
            text=s.text,
            label=s.label,
            start_char=s.start_char,
            end_char=s.end_char,
        )
        for s in spans
    ]
    return NERResponse(text=req.text, entities=items, grouped=grouped)
