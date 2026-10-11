"""Pytest configuration and shared fixtures for ReviewIQ tests."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def sample_reviews() -> list[dict]:
    """A tiny list of raw review dicts for testing preprocessing without real data."""
    return [
        {
            "asin": "B001",
            "reviewText": "Great laptop! Battery lasts forever. Very fast too.",
            "overall": 5.0,
            "reviewerID": "U001",
        },
        {
            "asin": "B001",
            "reviewText": "Not good at all. Battery dies in 2 hours. Wouldn't recommend.",
            "overall": 1.0,
            "reviewerID": "U002",
        },
        {
            "asin": "B002",
            "reviewText": "Average product. Display is ok but keyboard feels cheap :(",
            "overall": 3.0,
            "reviewerID": "U003",
        },
        {
            "asin": "B002",
            "reviewText": "I luv this laptop!! 10/10 would buy again 😍",
            "overall": 5.0,
            "reviewerID": "U004",
        },
        {
            "asin": "B003",
            "reviewText": "Terribel build quality. Hinges broke aftr 1 month. Total waste.",
            "overall": 1.0,
            "reviewerID": "U005",
        },
    ]


@pytest.fixture(scope="session")
def sample_texts(sample_reviews) -> list[str]:
    """Raw review texts extracted from sample_reviews."""
    return [r["reviewText"] for r in sample_reviews]
