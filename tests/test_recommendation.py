"""Unit and integration tests for ReviewIQ Recommendation Engine.

Covers:
- ProductAspectScorer & ProductAspectProfile aggregation
- Handling missing aspects (None vs neutral 0.0)
- AspectWeightedRanker ranking with customized user priorities
- Coverage penalty and cold-start shrinkage
- Transparent explanation generation
- Profile serialization and deserialization
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from reviewiq.recommendation.ranker import AspectWeightedRanker, RecommendationResult
from reviewiq.recommendation.scorer import (
    AspectAggregate,
    ProductAspectProfile,
    ProductAspectScorer,
)


@pytest.fixture
def product_scorer() -> ProductAspectScorer:
    return ProductAspectScorer()


@pytest.fixture
def sample_profiles() -> list[ProductAspectProfile]:
    """Synthetic product profiles representing distinct laptop archetype tradeoffs."""
    # Product A: Great battery, weak thermals, average display
    p_a = ProductAspectProfile(
        product_id="PROD_BATTERY_KING",
        title="UltraBook Marathon 14",
        total_reviews=20,
        aspect_scores={
            "battery": AspectAggregate(
                aspect="battery", mean_score=0.85, mention_count=18, positive_ratio=0.95, confidence=1.0
            ),
            "thermal": AspectAggregate(
                aspect="thermal", mean_score=-0.50, mention_count=12, positive_ratio=0.10, confidence=1.0
            ),
            "display": AspectAggregate(
                aspect="display", mean_score=0.20, mention_count=10, positive_ratio=0.60, confidence=1.0
            ),
        },
    )

    # Product B: Cool thermals & insane performance, mediocre battery
    p_b = ProductAspectProfile(
        product_id="PROD_GAMING_BEAST",
        title="Titan Vapor 16 RTX",
        total_reviews=25,
        aspect_scores={
            "battery": AspectAggregate(
                aspect="battery", mean_score=-0.30, mention_count=15, positive_ratio=0.20, confidence=1.0
            ),
            "thermal": AspectAggregate(
                aspect="thermal", mean_score=0.75, mention_count=20, positive_ratio=0.90, confidence=1.0
            ),
            "performance": AspectAggregate(
                aspect="performance", mean_score=0.90, mention_count=22, positive_ratio=0.95, confidence=1.0
            ),
        },
    )

    # Product C: Budget laptop with missing thermal data
    p_c = ProductAspectProfile(
        product_id="PROD_BUDGET",
        title="ValueBook 15",
        total_reviews=15,
        aspect_scores={
            "price": AspectAggregate(
                aspect="price", mean_score=0.80, mention_count=14, positive_ratio=0.90, confidence=1.0
            ),
            "battery": AspectAggregate(
                aspect="battery", mean_score=0.10, mention_count=8, positive_ratio=0.50, confidence=1.0
            ),
        },
    )

    # Product D: Cold start with only 1 review
    p_d = ProductAspectProfile(
        product_id="PROD_COLD_START",
        title="Unknown New Laptop",
        total_reviews=1,
        aspect_scores={
            "battery": AspectAggregate(
                aspect="battery", mean_score=0.95, mention_count=1, positive_ratio=1.0, confidence=0.6
            ),
        },
    )

    return [p_a, p_b, p_c, p_d]


class TestProductAspectScorer:
    def test_score_product_reviews(self, product_scorer: ProductAspectScorer) -> None:
        reviews = [
            "The battery life is amazing and lasts 14 hours. Excellent screen display.",
            "Great battery performance! The OLED display is bright and vivid.",
            "Keyboard feels mushy and cheap, but battery is solid.",
        ]
        profile = product_scorer.score_product_reviews(
            product_id="TEST_LAPTOP_01",
            reviews=reviews,
            title="Super Laptop Pro",
        )

        assert profile.product_id == "TEST_LAPTOP_01"
        assert profile.total_reviews == 3
        assert "battery" in profile.aspect_scores
        assert profile.aspect_scores["battery"].mean_score > 0.1
        assert profile.aspect_scores["battery"].mention_count == 3
        assert "display" in profile.aspect_scores
        assert profile.aspect_scores["display"].mean_score > 0.1
        # Thermal was not mentioned; must be None rather than 0.0
        assert profile.get_score("thermal") is None

    def test_empty_reviews_handling(self, product_scorer: ProductAspectScorer) -> None:
        profile = product_scorer.score_product_reviews(product_id="EMPTY_01", reviews=[])
        assert profile.total_reviews == 0
        assert profile.aspect_scores == {}
        assert profile.get_score("battery") is None

    def test_score_corpus_dataframe(self, product_scorer: ProductAspectScorer) -> None:
        df = pd.DataFrame({
            "parent_asin": ["LAPTOP_A", "LAPTOP_A", "LAPTOP_B"],
            "review_clean": [
                "Awesome battery life and fast performance.",
                "Battery is great.",
                "Screen display is beautiful, but battery drains fast.",
            ],
            "title": ["Laptop A Model", "Laptop A Model", "Laptop B Model"],
        })
        profiles = product_scorer.score_corpus_dataframe(df)
        assert "LAPTOP_A" in profiles
        assert "LAPTOP_B" in profiles
        assert profiles["LAPTOP_A"].total_reviews == 2
        assert profiles["LAPTOP_B"].total_reviews == 1
        assert profiles["LAPTOP_A"].aspect_scores["battery"].mention_count == 2

    def test_save_and_load_profiles(self, sample_profiles: list[ProductAspectProfile]) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            ProductAspectScorer.save_profiles(sample_profiles, temp_path)
            loaded = ProductAspectScorer.load_profiles(temp_path)
            assert len(loaded) == len(sample_profiles)
            assert "PROD_BATTERY_KING" in loaded
            king = loaded["PROD_BATTERY_KING"]
            assert king.title == "UltraBook Marathon 14"
            assert king.aspect_scores["battery"].mean_score == 0.85
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestAspectWeightedRanker:
    def test_rank_battery_focused_user(
        self, sample_profiles: list[ProductAspectProfile]
    ) -> None:
        ranker = AspectWeightedRanker()
        # User strongly prioritizes battery life
        results = ranker.rank(
            sample_profiles,
            user_weights={"battery": 3.0, "thermal": 0.2, "display": 0.5},
            top_n=3,
        )

        assert len(results) <= 3
        top_product = results[0]
        # Battery king should win
        assert top_product.product_id == "PROD_BATTERY_KING"
        assert top_product.rank == 1
        assert top_product.composite_score > 60.0
        assert "battery" in top_product.aspect_breakdown
        assert len(top_product.explanation) > 0

    def test_rank_thermal_performance_focused_user(
        self, sample_profiles: list[ProductAspectProfile]
    ) -> None:
        ranker = AspectWeightedRanker()
        # User is a gamer/creator who prioritizes cooling and performance over battery
        results = ranker.rank(
            sample_profiles,
            user_weights={"thermal": 3.0, "performance": 3.0, "battery": 0.1},
            top_n=3,
        )

        top_product = results[0]
        assert top_product.product_id == "PROD_GAMING_BEAST"
        assert top_product.rank == 1

    def test_negative_weight_validation(
        self, sample_profiles: list[ProductAspectProfile]
    ) -> None:
        ranker = AspectWeightedRanker()
        with pytest.raises(ValueError, match="cannot be negative"):
            ranker.rank(sample_profiles, user_weights={"battery": -1.5})

    def test_cold_start_shrinkage(
        self, sample_profiles: list[ProductAspectProfile]
    ) -> None:
        ranker = AspectWeightedRanker(min_confidence_reviews=5)
        # Even though PROD_COLD_START has 0.95 battery score, it only has 1 review.
        # PROD_BATTERY_KING has 0.85 score with 20 reviews.
        results = ranker.rank(
            sample_profiles,
            user_weights={"battery": 2.0},
            top_n=4,
        )

        prod_map = {r.product_id: r for r in results}
        assert prod_map["PROD_BATTERY_KING"].composite_score > prod_map["PROD_COLD_START"].composite_score

    def test_coverage_penalty(
        self, sample_profiles: list[ProductAspectProfile]
    ) -> None:
        ranker = AspectWeightedRanker(coverage_penalty=0.4)
        # User requires battery, thermal, and performance
        results = ranker.rank(
            sample_profiles,
            user_weights={"battery": 1.0, "thermal": 1.0, "performance": 1.0},
            top_n=4,
        )
        prod_map = {r.product_id: r for r in results}
        # PROD_BUDGET only has price and battery (lacks thermal and performance)
        assert prod_map["PROD_BUDGET"].coverage < 1.0

    def test_recommendation_result_to_dict(
        self, sample_profiles: list[ProductAspectProfile]
    ) -> None:
        ranker = AspectWeightedRanker()
        results = ranker.rank(sample_profiles, top_n=1)
        res_dict = results[0].to_dict()
        assert "rank" in res_dict
        assert "composite_score" in res_dict
        assert "aspect_breakdown" in res_dict
        assert "explanation" in res_dict
        assert "pros" in res_dict
        assert "cons" in res_dict
