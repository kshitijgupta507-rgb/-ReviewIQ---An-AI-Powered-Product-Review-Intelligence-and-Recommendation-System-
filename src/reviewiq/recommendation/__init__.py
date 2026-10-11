"""Product Recommendation Engine based on user-weighted aspect sentiment intelligence."""

from reviewiq.recommendation.ranker import AspectWeightedRanker, RecommendationResult
from reviewiq.recommendation.scorer import (
    AspectAggregate,
    ProductAspectProfile,
    ProductAspectScorer,
)

__all__ = [
    "AspectAggregate",
    "ProductAspectProfile",
    "ProductAspectScorer",
    "AspectWeightedRanker",
    "RecommendationResult",
]
