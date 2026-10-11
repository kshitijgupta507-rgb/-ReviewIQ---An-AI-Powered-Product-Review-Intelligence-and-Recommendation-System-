"""Data contracts for ReviewIQ.

These dataclasses define the canonical shape of a review and product metadata
throughout the pipeline. All modules should accept/return these types,
not raw dicts, so that type checking is meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Review:
    """A single product review after loading and basic validation."""

    asin: str
    """Amazon Standard Identification Number — unique product ID."""

    reviewer_id: str
    """Unique reviewer identifier (hashed/anonymised in Amazon data)."""

    text: str
    """Raw review text as written by the reviewer."""

    rating: float
    """Star rating 1.0 – 5.0."""

    title: str = ""
    """Review headline (may be empty)."""

    verified: bool = False
    """True if this is a verified purchase review."""

    timestamp: int = 0
    """Unix timestamp of the review."""

    @property
    def sentiment_label(self) -> str:
        """Coarse sentiment derived from star rating.

        1–2 stars → negative
        3 stars   → neutral
        4–5 stars → positive
        """
        if self.rating <= 2.0:
            return "negative"
        elif self.rating == 3.0:
            return "neutral"
        else:
            return "positive"


@dataclass
class ProductMeta:
    """Product metadata from the Amazon item metadata file."""

    asin: str
    title: str
    brand: str = ""
    price: float | None = None
    categories: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class ProcessedReview:
    """A review after the full preprocessing pipeline."""

    asin: str
    reviewer_id: str
    raw_text: str
    clean_tokens: list[str]
    """Lowercased, lemmatized, stopword-removed tokens."""
    rating: float
    sentiment_label: str
    has_negation: bool = False
    detected_language: str = "en"
