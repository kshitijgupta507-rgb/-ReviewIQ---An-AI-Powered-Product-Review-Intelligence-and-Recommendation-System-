"""Aspect score aggregation module for product recommendations.

Aggregates review-level Aspect-Based Sentiment Analysis (ABSA) outputs into
product-level aspect profiles:
- Computes mean sentiment score, mention count, and positive ratio per aspect
- Applies Bayesian shrinkage / statistical confidence based on mention count
- Explicitly distinguishes between missing aspects (None) and neutral sentiment (0.0)
- Serializes and deserializes structured ProductAspectProfile objects
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from reviewiq.absa.sentiment_scorer import AspectSentimentScorer
from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AspectAggregate:
    """Aggregated sentiment score and statistics for a specific product aspect."""

    aspect: str
    mean_score: float  # In [-1.0, 1.0]
    mention_count: int
    positive_ratio: float
    confidence: float  # Statistical sample confidence in [0.0, 1.0]
    evidence_samples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "aspect": self.aspect,
            "mean_score": float(self.mean_score),
            "mention_count": int(self.mention_count),
            "positive_ratio": float(self.positive_ratio),
            "confidence": float(self.confidence),
            "evidence_samples": self.evidence_samples,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AspectAggregate":
        return cls(
            aspect=data["aspect"],
            mean_score=float(data["mean_score"]),
            mention_count=int(data["mention_count"]),
            positive_ratio=float(data["positive_ratio"]),
            confidence=float(data.get("confidence", 1.0)),
            evidence_samples=data.get("evidence_samples", []),
        )


@dataclass
class ProductAspectProfile:
    """Aggregated aspect intelligence profile for a single product."""

    product_id: str
    aspect_scores: dict[str, AspectAggregate] = field(default_factory=dict)
    total_reviews: int = 0
    title: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_score(self, aspect: str) -> float | None:
        """Return the mean sentiment score if the aspect was mentioned, else None."""
        if aspect in self.aspect_scores:
            return self.aspect_scores[aspect].mean_score
        return None

    def get_coverage(self, target_aspects: Sequence[str]) -> float:
        """Fraction of target aspects present in this product's reviews."""
        if not target_aspects:
            return 1.0
        covered = sum(1 for a in target_aspects if a in self.aspect_scores)
        return float(covered / len(target_aspects))

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "aspect_scores": {k: v.to_dict() for k, v in self.aspect_scores.items()},
            "total_reviews": self.total_reviews,
            "title": self.title,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProductAspectProfile":
        scores = {
            k: AspectAggregate.from_dict(v)
            for k, v in data.get("aspect_scores", {}).items()
        }
        return cls(
            product_id=data["product_id"],
            aspect_scores=scores,
            total_reviews=data.get("total_reviews", 0),
            title=data.get("title", ""),
            metadata=data.get("metadata", {}),
        )


class ProductAspectScorer:
    """Aggregates review-level aspect sentiment into product aspect profiles.

    Parameters
    ----------
    sentiment_scorer:
        Optional pre-configured AspectSentimentScorer instance.
    min_mentions_for_full_confidence:
        Number of mentions required for 100% statistical confidence. Default 5.
    """

    def __init__(
        self,
        sentiment_scorer: AspectSentimentScorer | None = None,
        min_mentions_for_full_confidence: int = 5,
    ) -> None:
        self.scorer = sentiment_scorer or AspectSentimentScorer()
        self.min_mentions = max(1, min_mentions_for_full_confidence)

    def score_product_reviews(
        self,
        product_id: str,
        reviews: Sequence[str | dict[str, Any]],
        title: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> ProductAspectProfile:
        """Extract and aggregate aspect sentiment across a product's reviews.

        Parameters
        ----------
        product_id:
            Unique product identifier (e.g. ASIN or parent_asin).
        reviews:
            Sequence of review text strings or review dictionaries.
        title:
            Optional product name/title.
        metadata:
            Optional dictionary of metadata attributes.

        Returns
        -------
        ProductAspectProfile:
            Aggregated profile with scores for each detected aspect.
        """
        review_texts: list[str] = []
        for r in reviews:
            if isinstance(r, str):
                if r.strip():
                    review_texts.append(r.strip())
            elif isinstance(r, dict):
                text = r.get("review_clean") or r.get("text") or r.get("reviewText") or ""
                if text and str(text).strip():
                    review_texts.append(str(text).strip())

        total_reviews = len(review_texts)
        if total_reviews == 0:
            return ProductAspectProfile(
                product_id=product_id,
                aspect_scores={},
                total_reviews=0,
                title=title,
                metadata=metadata or {},
            )

        # Collect aspect scores and evidence
        aspect_scores_list: dict[str, list[float]] = {}
        aspect_evidence: dict[str, list[str]] = {}

        for text in review_texts:
            scored_aspects = self.scorer.score_review(text)
            for aspect_name, aspect_sent in scored_aspects.items():
                if aspect_name not in aspect_scores_list:
                    aspect_scores_list[aspect_name] = []
                    aspect_evidence[aspect_name] = []
                aspect_scores_list[aspect_name].append(aspect_sent.score)
                aspect_evidence[aspect_name].extend(aspect_sent.evidence_sentences)

        # Aggregate per aspect
        aspect_aggregates: dict[str, AspectAggregate] = {}
        for aspect, score_vals in aspect_scores_list.items():
            if not score_vals:
                continue
            mean_score = float(np.mean(score_vals))
            pos_ratio = float(sum(1 for s in score_vals if s >= 0.05) / len(score_vals))
            # Statistical confidence scales from 0.5 to 1.0 as mentions reach min_mentions
            count = len(score_vals)
            conf = min(1.0, 0.5 + 0.5 * (count / self.min_mentions))

            # Deduplicate and limit evidence sentences
            unique_evidence: list[str] = []
            seen = set()
            for s in aspect_evidence[aspect]:
                if s.lower() not in seen:
                    seen.add(s.lower())
                    unique_evidence.append(s)
                if len(unique_evidence) >= 3:
                    break

            aspect_aggregates[aspect] = AspectAggregate(
                aspect=aspect,
                mean_score=round(mean_score, 4),
                mention_count=count,
                positive_ratio=round(pos_ratio, 4),
                confidence=round(conf, 4),
                evidence_samples=unique_evidence,
            )

        return ProductAspectProfile(
            product_id=product_id,
            aspect_scores=aspect_aggregates,
            total_reviews=total_reviews,
            title=title,
            metadata=metadata or {},
        )

    def score_corpus_dataframe(
        self,
        df: pd.DataFrame,
        product_id_col: str = "parent_asin",
        text_col: str = "review_clean",
        title_col: str | None = "title",
    ) -> dict[str, ProductAspectProfile]:
        """Aggregate aspect scores across an entire DataFrame grouped by product.

        Parameters
        ----------
        df:
            Review DataFrame.
        product_id_col:
            Column name identifying products. Falls back to 'asin' if missing.
        text_col:
            Column containing review text. Falls back to 'text' if missing.
        title_col:
            Optional column containing product title.

        Returns
        -------
        dict[str, ProductAspectProfile]:
            Dictionary mapping product_id -> ProductAspectProfile.
        """
        pid_col = product_id_col if product_id_col in df.columns else ("asin" if "asin" in df.columns else None)
        if pid_col is None:
            raise ValueError(f"Neither '{product_id_col}' nor 'asin' found in DataFrame columns")

        t_col = text_col if text_col in df.columns else ("text" if "text" in df.columns else "reviewText")
        if t_col not in df.columns:
            raise ValueError(f"Neither '{text_col}' nor 'text' found in DataFrame columns")

        profiles: dict[str, ProductAspectProfile] = {}
        grouped = df.groupby(pid_col)

        logger.info("Scoring aspects for %d unique products in DataFrame...", len(grouped))
        for pid, group in grouped:
            texts = group[t_col].dropna().tolist()
            prod_title = ""
            if title_col and title_col in group.columns:
                non_empty = group[title_col].dropna()
                if not non_empty.empty:
                    prod_title = str(non_empty.iloc[0])

            profile = self.score_product_reviews(
                product_id=str(pid),
                reviews=texts,
                title=prod_title,
            )
            profiles[str(pid)] = profile

        return profiles

    @staticmethod
    def save_profiles(
        profiles: dict[str, ProductAspectProfile] | Sequence[ProductAspectProfile],
        filepath: str | Path,
    ) -> None:
        """Save product aspect profiles to a JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(profiles, dict):
            data = {k: v.to_dict() for k, v in profiles.items()}
        else:
            data = {p.product_id: p.to_dict() for p in profiles}

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved %d product profiles to %s", len(data), path)

    @staticmethod
    def load_profiles(filepath: str | Path) -> dict[str, ProductAspectProfile]:
        """Load product aspect profiles from a JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Profiles file not found at {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        profiles = {k: ProductAspectProfile.from_dict(v) for k, v in data.items()}
        logger.info("Loaded %d product profiles from %s", len(profiles), path)
        return profiles
