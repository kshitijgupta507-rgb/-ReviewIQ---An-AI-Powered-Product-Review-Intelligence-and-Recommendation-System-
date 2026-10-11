"""Extractive and abstractive summarization router."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from reviewiq.summarization.abstractive import AbstractiveSummarizer, evaluate_rouge
from reviewiq.summarization.textrank import TextRankSummarizer

router = APIRouter()

_textrank: TextRankSummarizer | None = None
_abstractive: AbstractiveSummarizer | None = None


def get_textrank() -> TextRankSummarizer:
    global _textrank
    if _textrank is None:
        _textrank = TextRankSummarizer()
    return _textrank


def get_abstractive() -> AbstractiveSummarizer:
    global _abstractive
    if _abstractive is None:
        _abstractive = AbstractiveSummarizer()
    return _abstractive


class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Customer review text or collection of reviews")
    num_sentences: int = Field(3, ge=1, le=10, description="Target extracted sentences")


class SummarizeResponse(BaseModel):
    method: str
    summary: str
    sentence_count: int
    word_count: int


@router.post("/textrank", response_model=SummarizeResponse)
async def summarize_textrank(req: SummarizeRequest) -> SummarizeResponse:
    """Generate extractive summary using sentence similarity graph PageRank."""
    summarizer = get_textrank()
    summary = summarizer.summarize(req.text, num_sentences=req.num_sentences)
    sents = summarizer.split_sentences(summary)
    return SummarizeResponse(
        method="TextRank (Extractive Graph PageRank)",
        summary=summary,
        sentence_count=len(sents),
        word_count=len(summary.split()),
    )


@router.post("/abstractive", response_model=SummarizeResponse)
async def summarize_abstractive(req: SummarizeRequest) -> SummarizeResponse:
    """Generate neural abstractive summary using Seq2Seq Transformer."""
    summarizer = get_abstractive()
    summary = summarizer.summarize(req.text)
    words = summary.split()
    return SummarizeResponse(
        method=f"Seq2Seq Transformer ({summarizer.model_name})",
        summary=summary,
        sentence_count=len(get_textrank().split_sentences(summary)),
        word_count=len(words),
    )
