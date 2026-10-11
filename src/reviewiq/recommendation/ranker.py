"""User-weighted aspect recommendation ranker with transparent explanations.

Ranks products by matching user-specified aspect priority weights against
aggregated product aspect profiles:
- Supports arbitrary user preference weighting across the 12 canonical laptop aspects
- Computes transparent dot-product composite scores with coverage penalty
- Employs Bayesian confidence shrinkage to avoid cold-start bias (1 review beating 100)
- Generates natural-language explainable recommendation justifications
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from reviewiq.recommendation.scorer import AspectAggregate, ProductAspectProfile
from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RecommendationResult:
    """Detailed recommendation outcome for a single ranked product."""

    rank: int
    product_id: str
    composite_score: float  # Normalized score on [0.0, 100.0] scale
    aspect_breakdown: dict[str, float]  # Contribution per aspect
    coverage: float  # Aspect coverage fraction in [0.0, 1.0]
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)
    total_reviews: int = 0
    title: str = ""
    explanation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "product_id": self.product_id,
            "composite_score": round(self.composite_score, 2),
            "aspect_breakdown": {k: round(v, 2) for k, v in self.aspect_breakdown.items()},
            "coverage": round(self.coverage, 4),
            "pros": self.pros,
            "cons": self.cons,
            "total_reviews": self.total_reviews,
            "title": self.title,
            "explanation": self.explanation,
            "metadata": self.metadata,
        }


class AspectWeightedRanker:
    """Ranks products based on user aspect preferences and sentiment profiles.

    Parameters
    ----------
    default_weights:
        Default aspect weights dictionary. Defaults to configs/default.yaml.
    coverage_penalty:
        Penalty coefficient applied when requested aspects lack data. Default 0.25.
    min_confidence_reviews:
        Review threshold for 100% confidence weighting (mitigates cold-start). Default 5.
    """

    def __init__(
        self,
        default_weights: dict[str, float] | None = None,
        coverage_penalty: float = 0.25,
        min_confidence_reviews: int = 5,
    ) -> None:
        cfg = load_config()
        rec_cfg = getattr(cfg, "recommendation", None)

        config_weights: dict[str, float] = {}
        if rec_cfg and hasattr(rec_cfg, "default_weights"):
            config_weights = dict(rec_cfg.default_weights)
        elif isinstance(rec_cfg, dict) and "default_weights" in rec_cfg:
            config_weights = dict(rec_cfg["default_weights"])

        self.default_weights = (
            dict(default_weights)
            if default_weights is not None
            else config_weights or {
                "battery": 1.0, "display": 1.0, "performance": 1.0,
                "build_quality": 1.0, "keyboard": 1.0, "price": 1.0,
                "thermal": 1.0, "storage": 1.0, "weight": 1.0,
                "software": 1.0, "customer_service": 0.5, "delivery": 0.5,
            }
        )
        self.coverage_penalty = max(0.0, min(1.0, coverage_penalty))
        self.min_confidence_reviews = max(1, min_confidence_reviews)

    def _normalize_score_100(self, score: float) -> float:
        """Map raw sentiment score in [-1.0, 1.0] to a [0.0, 100.0] consumer scale."""
        clamped = max(-1.0, min(1.0, score))
        return (clamped + 1.0) * 50.0

    def _generate_explanation(
        self,
        product_id: str,
        title: str,
        composite_score: float,
        user_weights: dict[str, float],
        aspect_scores: dict[str, AspectAggregate],
        pros: list[str],
        cons: list[str],
        coverage: float,
    ) -> str:
        """Generate human-readable justification for the product recommendation."""
        name = title if title else f"Product {product_id}"

        # Identify top aspects matching highest user weights
        sorted_weights = sorted(user_weights.items(), key=lambda x: x[1], reverse=True)
        top_user_priorities = [k for k, w in sorted_weights if w > 0.0][:2]

        priority_phrases = []
        for p in top_user_priorities:
            if p in aspect_scores:
                sent = aspect_scores[p].mean_score
                if sent >= 0.2:
                    priority_phrases.append(f"strong {p.replace('_', ' ')} (score: {sent:+.2f})")
                elif sent <= -0.2:
                    priority_phrases.append(f"weak {p.replace('_', ' ')} (score: {sent:+.2f})")

        reasons = []
        if priority_phrases:
            reasons.append(f"Delivers {', and '.join(priority_phrases)} matching your high priorities.")

        if pros:
            reasons.append(f"Notable strengths: {', '.join(pros[:2])}.")
        if cons:
            reasons.append(f"Potential drawback: {', '.join(cons[:1])}.")

        if coverage < 0.6:
            reasons.append(f"Note: limited review mentions across {int((1.0 - coverage) * 100)}% of desired aspects.")

        if not reasons:
            reasons.append(f"Overall solid performance with a composite score of {composite_score:.1f}/100.")

        return " ".join(reasons)

    def rank(
        self,
        profiles: Sequence[ProductAspectProfile] | dict[str, ProductAspectProfile],
        user_weights: dict[str, float] | None = None,
        top_n: int = 5,
        min_total_reviews: int = 0,
    ) -> list[RecommendationResult]:
        """Rank products according to user aspect preferences.

        Parameters
        ----------
        profiles:
            Collection of ProductAspectProfile instances or mapping product_id -> profile.
        user_weights:
            Optional custom dictionary of aspect weights (e.g. {'battery': 2.0, 'price': 1.5}).
            Aspects omitted default to self.default_weights or 0.0 if empty.
        top_n:
            Number of recommendations to return.
        min_total_reviews:
            Minimum review count required to be considered for recommendation.

        Returns
        -------
        list[RecommendationResult]:
            Ranked list of products with scores, breakdowns, and explanations.
        """
        if isinstance(profiles, dict):
            profile_list = list(profiles.values())
        else:
            profile_list = list(profiles)

        if not profile_list:
            return []

        # Determine active weights
        active_weights: dict[str, float] = {}
        if user_weights is not None:
            for k, w in user_weights.items():
                if w < 0:
                    raise ValueError(f"Aspect weight for '{k}' cannot be negative: {w}")
                active_weights[k] = float(w)
        else:
            active_weights = dict(self.default_weights)

        # Target aspects with non-zero weight
        target_aspects = [a for a, w in active_weights.items() if w > 0.0]
        total_target_weight = sum(active_weights[a] for a in target_aspects)
        if total_target_weight <= 0:
            total_target_weight = 1.0

        scored_candidates: list[RecommendationResult] = []

        for prof in profile_list:
            if prof.total_reviews < min_total_reviews:
                continue

            # Compute weighted sum
            aspect_breakdown: dict[str, float] = {}
            weighted_sum = 0.0
            sum_weights = 0.0

            pros: list[str] = []
            cons: list[str] = []

            for aspect in target_aspects:
                weight = active_weights[aspect]
                agg = prof.aspect_scores.get(aspect)

                if agg is not None:
                    norm_score = self._normalize_score_100(agg.mean_score)
                    aspect_contrib = norm_score * weight
                    aspect_breakdown[aspect] = norm_score
                    weighted_sum += aspect_contrib
                    sum_weights += weight

                    if agg.mean_score >= 0.15:
                        pros.append(f"{aspect.replace('_', ' ')} (+{agg.mean_score:.2f})")
                    elif agg.mean_score <= -0.15:
                        cons.append(f"{aspect.replace('_', ' ')} ({agg.mean_score:.2f})")

            # Coverage across targeted aspects
            coverage = prof.get_coverage(target_aspects)

            # Base score from observed aspects
            if sum_weights > 0:
                base_score = weighted_sum / sum_weights
            else:
                base_score = 50.0  # Default neutral midpoint

            # Apply coverage penalty: penalize products that lack data for requested aspects
            coverage_factor = 1.0 - (self.coverage_penalty * (1.0 - coverage))
            penalized_score = base_score * coverage_factor

            # Cold-start shrinkage: pull score gently towards neutral (50.0) if few reviews
            sample_confidence = min(1.0, prof.total_reviews / self.min_confidence_reviews)
            final_composite = float(
                sample_confidence * penalized_score + (1.0 - sample_confidence) * 50.0
            )

            explanation = self._generate_explanation(
                product_id=prof.product_id,
                title=prof.title,
                composite_score=final_composite,
                user_weights=active_weights,
                aspect_scores=prof.aspect_scores,
                pros=pros,
                cons=cons,
                coverage=coverage,
            )

            scored_candidates.append(
                RecommendationResult(
                    rank=0,
                    product_id=prof.product_id,
                    composite_score=final_composite,
                    aspect_breakdown=aspect_breakdown,
                    coverage=coverage,
                    pros=pros,
                    cons=cons,
                    total_reviews=prof.total_reviews,
                    title=prof.title,
                    explanation=explanation,
                    metadata=prof.metadata,
                )
            )

        # Sort descending by composite score, then total reviews as tie-breaker
        scored_candidates.sort(
            key=lambda x: (x.composite_score, x.total_reviews),
            reverse=True,
        )

        # Assign ranks
        results: list[RecommendationResult] = []
        for i, cand in enumerate(scored_candidates[:top_n], start=1):
            cand.rank = i
            results.append(cand)

        return results
