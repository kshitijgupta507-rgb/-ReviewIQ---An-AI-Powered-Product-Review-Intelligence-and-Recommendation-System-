"""Unit and integration tests for ReviewIQ Aspect-Based Sentiment Analysis (ABSA).

Covers:
- AspectExtractor (12 canonical aspects, seed keywords, opinion words, negation detection)
- AspectSentimentScorer (VADER polarity scoring, confidence, review aggregation)
- SemEval-2014 XML parsing and evaluation
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from reviewiq.absa.aspect_extractor import AspectExtractor, ExtractedAspect
from reviewiq.absa.sentiment_scorer import AspectSentiment, AspectSentimentScorer


@pytest.fixture
def extractor() -> AspectExtractor:
    return AspectExtractor()


@pytest.fixture
def scorer() -> AspectSentimentScorer:
    # Use VADER (lightweight, deterministic, offline) for unit testing
    return AspectSentimentScorer(use_vader=True, use_transformer=False)


class TestAspectExtractor:
    def test_extract_battery_aspect(self, extractor: AspectExtractor) -> None:
        text = "The battery life on this laptop is incredible, lasting over 12 hours."
        aspects = extractor.extract(text)
        assert len(aspects) >= 1
        aspect_names = [a.aspect for a in aspects]
        assert "battery" in aspect_names
        battery_aspect = next(a for a in aspects if a.aspect == "battery")
        assert battery_aspect.is_negated is False
        assert "incredible" in battery_aspect.opinion_words or "lasting" in battery_aspect.opinion_words

    def test_extract_display_aspect(self, extractor: AspectExtractor) -> None:
        text = "The OLED screen is vivid and bright with deep blacks."
        aspects = extractor.extract(text)
        aspect_names = [a.aspect for a in aspects]
        assert "display" in aspect_names

    def test_extract_multiple_aspects_single_text(self, extractor: AspectExtractor) -> None:
        text = "Great performance and fast processor, but the keyboard feels cheap and mushy."
        aspects = extractor.extract(text)
        aspect_names = set(a.aspect for a in aspects)
        assert "performance" in aspect_names
        assert "keyboard" in aspect_names

    def test_extract_negation(self, extractor: AspectExtractor) -> None:
        text = "The battery is not good at all and drains very quickly."
        aspects = extractor.extract(text)
        battery_aspects = [a for a in aspects if a.aspect == "battery"]
        assert len(battery_aspects) >= 1
        assert any(a.is_negated for a in battery_aspects)

    def test_extract_canonical_aspect_coverage(self, extractor: AspectExtractor) -> None:
        canonical_samples = [
            ("The battery lasts 10 hours", "battery"),
            ("The 4K display is breathtaking", "display"),
            ("The CPU speed and gaming performance are insane", "performance"),
            ("Solid aluminum build quality and durable chassis", "build_quality"),
            ("Mechanical keyboard with smooth keystrokes", "keyboard"),
            ("Affordable price and great budget value", "price"),
            ("Laptop runs hot and the cooling fans are noisy", "thermal"),
            ("Huge 1TB NVMe SSD storage", "storage"),
            ("Very lightweight and portable ultrabook", "weight"),
            ("Clean Windows 11 OS without bloatware", "software"),
            ("Helpful customer support and fast warranty replacement", "customer_service"),
            ("Fast Prime shipping and package arrived safely", "delivery"),
        ]
        for text, expected_aspect in canonical_samples:
            aspects = extractor.extract(text)
            aspect_names = [a.aspect for a in aspects]
            assert expected_aspect in aspect_names, f"Failed to extract {expected_aspect} from '{text}'"

    def test_batch_extraction(self, extractor: AspectExtractor) -> None:
        texts = [
            "Super long battery.",
            "Terrible display quality.",
            "Completely irrelevant sentence with no keywords.",
        ]
        batch_results = extractor.extract_batch(texts)
        assert len(batch_results) == 3
        assert len(batch_results[0]) >= 1
        assert batch_results[0][0].aspect == "battery"
        assert len(batch_results[1]) >= 1
        assert batch_results[1][0].aspect == "display"
        assert len(batch_results[2]) == 0

    def test_empty_and_whitespace_input(self, extractor: AspectExtractor) -> None:
        assert extractor.extract("") == []
        assert extractor.extract("   \n\t  ") == []


class TestAspectSentimentScorer:
    def test_positive_aspect_sentiment(self, scorer: AspectSentimentScorer) -> None:
        aspect = ExtractedAspect(
            aspect="battery",
            term="battery life",
            opinion_words=["excellent", "superb"],
            sentence="The battery life is excellent and superb.",
            is_negated=False,
        )
        result = scorer.score_aspect(aspect)
        assert result.sentiment == "positive"
        assert result.score > 0.05
        assert result.aspect == "battery"

    def test_negative_aspect_sentiment(self, scorer: AspectSentimentScorer) -> None:
        aspect = ExtractedAspect(
            aspect="thermal",
            term="fans",
            opinion_words=["terrible", "horrible"],
            sentence="The fans are terrible and loud.",
            is_negated=False,
        )
        result = scorer.score_aspect(aspect)
        assert result.sentiment == "negative"
        assert result.score < -0.05

    def test_negated_positive_inversion(self, scorer: AspectSentimentScorer) -> None:
        aspect_pos = ExtractedAspect(
            aspect="battery",
            term="battery",
            opinion_words=["good"],
            sentence="The battery is good.",
            is_negated=False,
        )
        aspect_neg = ExtractedAspect(
            aspect="battery",
            term="battery",
            opinion_words=["good"],
            sentence="The battery is not good.",
            is_negated=True,
        )
        res_pos = scorer.score_aspect(aspect_pos)
        res_neg = scorer.score_aspect(aspect_neg)
        assert res_pos.score > res_neg.score
        assert res_neg.sentiment == "negative" or res_neg.score <= 0.0

    def test_score_review(self, scorer: AspectSentimentScorer) -> None:
        text = "Incredible battery and bright display, but horrible keyboard."
        results = scorer.score_review(text)
        assert "battery" in results
        assert "display" in results
        assert "keyboard" in results

        assert results["battery"].sentiment == "positive"
        assert results["display"].sentiment == "positive"
        assert results["keyboard"].sentiment == "negative"

    def test_score_corpus_aggregate(self, scorer: AspectSentimentScorer) -> None:
        reviews = [
            "Great battery life, lasts all day.",
            "Battery is phenomenal!",
            "Terrible thermal throttling, gets super hot.",
        ]
        summary = scorer.score_corpus(reviews)
        assert "battery" in summary
        assert "thermal" in summary
        assert summary["battery"].mention_count == 2
        assert summary["battery"].sentiment == "positive"
        assert summary["thermal"].mention_count == 1
        assert summary["thermal"].sentiment == "negative"

    def test_semeval_evaluation(self, scorer: AspectSentimentScorer) -> None:
        # Create a synthetic SemEval-2014 Task 4 XML benchmark file
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<sentences>
    <sentence id="1">
        <text>The battery life is exceptional.</text>
        <aspectTerms>
            <aspectTerm term="battery life" polarity="positive" from="4" to="16"/>
        </aspectTerms>
    </sentence>
    <sentence id="2">
        <text>Terrible display resolution and washed out colors.</text>
        <aspectTerms>
            <aspectTerm term="display" polarity="negative" from="9" to="16"/>
        </aspectTerms>
    </sentence>
</sentences>
"""
        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8") as f:
            f.write(xml_content)
            temp_path = f.name

        try:
            metrics = scorer.evaluate_semeval(temp_path)
            assert "accuracy" in metrics
            assert "total_aspects" in metrics
            assert metrics["total_aspects"] == 2
            assert metrics["accuracy"] >= 0.5
            assert "confusion_matrix" in metrics
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_semeval_clause_scoped_negation(self, scorer: AspectSentimentScorer) -> None:
        """Verify that negation in one clause does not affect an aspect term in another clause."""
        xml_content = """<?xml version="1.0" encoding="UTF-8"?>
<sentences>
    <sentence id="1">
        <text>The battery life is great, but the trackpad does not work well.</text>
        <aspectTerms>
            <aspectTerm term="battery life" polarity="positive" from="4" to="16"/>
            <aspectTerm term="trackpad" polarity="negative" from="39" to="47"/>
        </aspectTerms>
    </sentence>
</sentences>
"""
        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8") as f:
            f.write(xml_content)
            temp_path = f.name

        try:
            metrics = scorer.evaluate_semeval(temp_path)
            assert metrics["total_aspects"] == 2
            # Both terms should be correctly predicted
            assert metrics["accuracy"] == 1.0
        finally:
            Path(temp_path).unlink(missing_ok=True)
