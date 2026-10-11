"""Aspect-Based Sentiment Analysis (ABSA) module."""

from reviewiq.absa.aspect_extractor import AspectExtractor, ExtractedAspect
from reviewiq.absa.sentiment_scorer import AspectSentiment, AspectSentimentScorer

__all__ = [
    "AspectExtractor",
    "ExtractedAspect",
    "AspectSentiment",
    "AspectSentimentScorer",
]
