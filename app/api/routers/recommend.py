"""Product recommendation API router."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from reviewiq.recommendation.ranker import AspectWeightedRanker, RecommendationResult
from reviewiq.recommendation.scorer import (
    AspectAggregate,
    ProductAspectProfile,
    ProductAspectScorer,
)

router = APIRouter()

# In-memory default catalog profiles for demonstration
_profiles: dict[str, ProductAspectProfile] = {
    "DELL_XPS_15": ProductAspectProfile(
        product_id="DELL_XPS_15",
        title="Dell XPS 15 (OLED 3.5K, i7-13700H)",
        total_reviews=45,
        aspect_scores={
            "display": AspectAggregate("display", 0.92, 38, 0.95, 1.0),
            "build_quality": AspectAggregate("build_quality", 0.88, 30, 0.92, 1.0),
            "performance": AspectAggregate("performance", 0.82, 35, 0.88, 1.0),
            "keyboard": AspectAggregate("keyboard", 0.75, 25, 0.80, 1.0),
            "thermal": AspectAggregate("thermal", -0.20, 20, 0.40, 1.0),
            "battery": AspectAggregate("battery", 0.35, 28, 0.65, 1.0),
        },
    ),
    "MACBOOK_AIR_M2": ProductAspectProfile(
        product_id="MACBOOK_AIR_M2",
        title="Apple MacBook Air 15 (M2, 16GB, 512GB)",
        total_reviews=60,
        aspect_scores={
            "battery": AspectAggregate("battery", 0.96, 52, 0.98, 1.0),
            "weight": AspectAggregate("weight", 0.90, 40, 0.95, 1.0),
            "build_quality": AspectAggregate("build_quality", 0.94, 45, 0.96, 1.0),
            "thermal": AspectAggregate("thermal", 0.85, 30, 0.90, 1.0),
            "display": AspectAggregate("display", 0.88, 42, 0.90, 1.0),
            "price": AspectAggregate("price", -0.15, 25, 0.45, 1.0),
        },
    ),
    "ASUS_ROG_ZEPHYRUS": ProductAspectProfile(
        product_id="ASUS_ROG_ZEPHYRUS",
        title="ASUS ROG Zephyrus G14 (RTX 4070, Ryzen 9)",
        total_reviews=35,
        aspect_scores={
            "performance": AspectAggregate("performance", 0.95, 32, 0.97, 1.0),
            "thermal": AspectAggregate("thermal", 0.70, 25, 0.85, 1.0),
            "display": AspectAggregate("display", 0.90, 28, 0.92, 1.0),
            "battery": AspectAggregate("battery", -0.40, 22, 0.30, 1.0),
            "keyboard": AspectAggregate("keyboard", 0.80, 18, 0.85, 1.0),
        },
    ),
    "LENOVO_THINKPAD_T14": ProductAspectProfile(
        product_id="LENOVO_THINKPAD_T14",
        title="Lenovo ThinkPad T14 Gen 4 (AMD Ryzen 7)",
        total_reviews=40,
        aspect_scores={
            "keyboard": AspectAggregate("keyboard", 0.95, 36, 0.98, 1.0),
            "build_quality": AspectAggregate("build_quality", 0.92, 34, 0.95, 1.0),
            "battery": AspectAggregate("battery", 0.75, 28, 0.85, 1.0),
            "performance": AspectAggregate("performance", 0.78, 25, 0.82, 1.0),
            "display": AspectAggregate("display", 0.40, 20, 0.65, 1.0),
        },
    ),
}

_ranker: AspectWeightedRanker | None = None


def get_ranker() -> AspectWeightedRanker:
    global _ranker
    if _ranker is None:
        _ranker = AspectWeightedRanker()
    return _ranker


class RecommendRequest(BaseModel):
    user_weights: dict[str, float] = Field(
        default_factory=dict,
        description="Aspect weights mapping, e.g. {'battery': 2.0, 'display': 1.5}",
    )
    top_n: int = Field(5, ge=1, le=20, description="Max recommendation results")


class ProductRecItem(BaseModel):
    rank: int
    product_id: str
    title: str
    composite_score: float
    coverage: float
    aspect_breakdown: dict[str, float]
    pros: list[str]
    cons: list[str]
    total_reviews: int
    explanation: str


class RecommendResponse(BaseModel):
    total_products: int
    recommendations: list[ProductRecItem]


@router.post("/rank", response_model=RecommendResponse)
async def rank_products(req: RecommendRequest) -> RecommendResponse:
    """Rank laptops using user-weighted aspect preferences and transparent explanations."""
    ranker = get_ranker()
    try:
        results = ranker.rank(_profiles, user_weights=req.user_weights, top_n=req.top_n)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    items = [
        ProductRecItem(
            rank=r.rank,
            product_id=r.product_id,
            title=r.title,
            composite_score=round(r.composite_score, 2),
            coverage=round(r.coverage, 4),
            aspect_breakdown={k: round(v, 2) for k, v in r.aspect_breakdown.items()},
            pros=r.pros,
            cons=r.cons,
            total_reviews=r.total_reviews,
            explanation=r.explanation,
        )
        for r in results
    ]

    return RecommendResponse(total_products=len(items), recommendations=items)
