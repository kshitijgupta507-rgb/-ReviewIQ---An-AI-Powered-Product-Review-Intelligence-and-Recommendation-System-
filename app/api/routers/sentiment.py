"""Sentiment and Aspect-Based Sentiment Analysis (ABSA) router."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from reviewiq.absa.aspect_extractor import AspectExtractor
from reviewiq.absa.sentiment_scorer import AspectSentimentScorer

router = APIRouter()

_extractor: AspectExtractor | None = None
_scorer: AspectSentimentScorer | None = None


def get_extractor() -> AspectExtractor:
    global _extractor
    if _extractor is None:
        _extractor = AspectExtractor()
    return _extractor


def get_scorer() -> AspectSentimentScorer:
    global _scorer
    if _scorer is None:
        _scorer = AspectSentimentScorer(use_vader=True, use_transformer=False)
    return _scorer


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Review text string")


class ExtractedAspectItem(BaseModel):
    aspect: str
    term: str
    opinion_words: list[str]
    sentence: str
    is_negated: bool


class AspectScoreItem(BaseModel):
    aspect: str
    sentiment: str
    score: float
    confidence: float
    mention_count: int
    evidence_sentences: list[str]


class ABSAResponse(BaseModel):
    text: str
    extracted_aspects: list[ExtractedAspectItem]
    scored_aspects: dict[str, AspectScoreItem]


class ClassifyResponse(BaseModel):
    text: str
    sentiment: str
    score: float
    confidence: float


@router.post("/classify", response_model=ClassifyResponse)
async def classify_sentiment(req: TextRequest) -> ClassifyResponse:
    """Classify overall sentiment of review text into positive, negative, or neutral."""
    scorer = get_scorer()
    assert scorer.vader is not None
    scores = scorer.vader.polarity_scores(req.text)
    compound = float(scores["compound"])
    if compound >= 0.05:
        sentiment = "positive"
        conf = float(scores.get("pos", 0.5) + compound * 0.5)
    elif compound <= -0.05:
        sentiment = "negative"
        conf = float(scores.get("neg", 0.5) + abs(compound) * 0.5)
    else:
        sentiment = "neutral"
        conf = float(scores.get("neu", 0.7))

    return ClassifyResponse(
        text=req.text,
        sentiment=sentiment,
        score=round(compound, 4),
        confidence=round(min(max(conf, 0.5), 1.0), 4),
    )


@router.post("/aspects", response_model=ABSAResponse)
async def analyze_aspects(req: TextRequest) -> ABSAResponse:
    """Extract laptop aspects, modifiers, and score per-aspect sentiment."""
    extractor = get_extractor()
    scorer = get_scorer()

    extracted = extractor.extract(req.text)
    scored = scorer.score_review(req.text)

    ext_items = [
        ExtractedAspectItem(
            aspect=e.aspect,
            term=e.term,
            opinion_words=e.opinion_words,
            sentence=e.sentence,
            is_negated=e.is_negated,
        )
        for e in extracted
    ]

    scored_items = {
        k: AspectScoreItem(
            aspect=v.aspect,
            sentiment=v.sentiment,
            score=round(v.score, 4),
            confidence=round(v.confidence, 4),
            mention_count=v.mention_count,
            evidence_sentences=v.evidence_sentences,
        )
        for k, v in scored.items()
    }

    return ABSAResponse(
        text=req.text,
        extracted_aspects=ext_items,
        scored_aspects=scored_items,
    )
