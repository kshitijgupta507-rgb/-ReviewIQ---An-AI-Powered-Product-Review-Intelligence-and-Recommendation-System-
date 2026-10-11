"""Unit and integration tests for ReviewIQ Summarization modules.

Covers:
- TextRank extractive summarizer (sentence segmentation, TF-IDF cosine graph, PageRank, chronological ordering)
- Abstractive summarizer interface and pipeline invocation
- ROUGE-1, ROUGE-2, and ROUGE-L precision, recall, and F1 evaluation
- Head-to-head comparison helper (compare_summarizers)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from reviewiq.summarization.abstractive import (
    AbstractiveSummarizer,
    compare_summarizers,
    evaluate_rouge,
    evaluate_rouge_batch,
)
from reviewiq.summarization.textrank import TextRankSummarizer


@pytest.fixture
def textrank() -> TextRankSummarizer:
    return TextRankSummarizer(num_sentences=2, similarity_threshold=0.05)


@pytest.fixture
def multi_sentence_review() -> str:
    return (
        "This laptop delivers exceptional performance for video editing and programming tasks. "
        "The battery life easily lasts a full ten-hour workday on a single charge. "
        "Unfortunately, the chassis gets noticeably warm when rendering heavy 3D animations. "
        "The OLED 120Hz display is stunning with deep blacks and great contrast. "
        "Customer service was prompt and replaced my power adapter within two days."
    )


class TestTextRankSummarizer:
    def test_sentence_splitting(self, textrank: TextRankSummarizer) -> None:
        text = "First sentence here. Second sentence follows! Is this the third sentence?"
        sentences = textrank.split_sentences(text)
        assert len(sentences) == 3
        assert "First sentence here." in sentences[0]
        assert "Second sentence follows!" in sentences[1]
        assert "Is this the third sentence?" in sentences[2]

    def test_textrank_summarize_length(
        self, textrank: TextRankSummarizer, multi_sentence_review: str
    ) -> None:
        summary = textrank.summarize(multi_sentence_review, num_sentences=2)
        assert isinstance(summary, str)
        assert len(summary) > 0
        extracted_sentences = textrank.split_sentences(summary)
        assert len(extracted_sentences) == 2

    def test_textrank_with_scores(
        self, textrank: TextRankSummarizer, multi_sentence_review: str
    ) -> None:
        result = textrank.summarize_with_scores(multi_sentence_review, num_sentences=3)
        assert "summary" in result
        assert "ranked_sentences" in result
        assert "scores" in result
        assert "num_input_sentences" in result
        assert result["num_input_sentences"] == 5
        assert len(result["scores"]) == 5
        assert len(result["ranked_sentences"]) == 5

        # Check PageRank scores are non-negative and sum to approx 1.0
        scores_sum = sum(result["scores"].values())
        assert pytest.approx(scores_sum, abs=0.05) == 1.0

    def test_textrank_preserves_chronological_order(
        self, textrank: TextRankSummarizer, multi_sentence_review: str
    ) -> None:
        result = textrank.summarize_with_scores(multi_sentence_review, num_sentences=2)
        summary = result["summary"]
        all_sentences = textrank.split_sentences(multi_sentence_review)

        # Get positions of extracted sentences in original text
        indices = [
            all_sentences.index(s)
            for s in textrank.split_sentences(summary)
            if s in all_sentences
        ]
        # Must be strictly increasing (chronological)
        assert indices == sorted(indices)

    def test_textrank_empty_and_short_inputs(self, textrank: TextRankSummarizer) -> None:
        assert textrank.summarize("") == ""
        assert textrank.summarize("Only one sentence.") == "Only one sentence."

    def test_textrank_multi_review_input(self, textrank: TextRankSummarizer) -> None:
        reviews = [
            "Battery lasts 12 hours easily. The build quality is solid.",
            "Display is gorgeous and bright. Trackpad is very smooth.",
        ]
        summary = textrank.summarize(reviews, num_sentences=2)
        assert len(textrank.split_sentences(summary)) <= 2


class TestRougeEvaluation:
    def test_evaluate_rouge_exact_match(self) -> None:
        reference = "The battery life on this laptop is incredible and lasts all day long."
        hypothesis = "The battery life on this laptop is incredible and lasts all day long."
        scores = evaluate_rouge(reference, hypothesis)

        assert scores["rouge1"]["fmeasure"] == 1.0
        assert scores["rouge2"]["fmeasure"] == 1.0
        assert scores["rougeL"]["fmeasure"] == 1.0

    def test_evaluate_rouge_partial_overlap(self) -> None:
        reference = "The battery life is excellent and the screen is bright."
        hypothesis = "The battery life is great and the display is bright."
        scores = evaluate_rouge(reference, hypothesis)

        assert 0.0 < scores["rouge1"]["fmeasure"] < 1.0
        assert "precision" in scores["rouge1"]
        assert "recall" in scores["rouge1"]

    def test_evaluate_rouge_empty(self) -> None:
        scores = evaluate_rouge("", "")
        assert scores["rouge1"]["fmeasure"] == 0.0

    def test_evaluate_rouge_batch(self) -> None:
        refs = ["Great laptop battery life.", "Super fast gaming performance."]
        hyps = ["Great laptop battery.", "Fast gaming laptop performance."]
        batch_scores = evaluate_rouge_batch(refs, hyps)

        assert "rouge1" in batch_scores
        assert "rouge2" in batch_scores
        assert "rougeL" in batch_scores
        assert batch_scores["rouge1"]["fmeasure"] > 0.0


class TestAbstractiveSummarizer:
    def test_short_input_bypass(self) -> None:
        summarizer = AbstractiveSummarizer()
        short_text = "This is a brief review."
        # Should return short text without loading heavy seq2seq pipeline
        result = summarizer.summarize(short_text)
        assert result == short_text

    def test_mocked_transformer_generation(self) -> None:
        import torch

        summarizer = AbstractiveSummarizer(model_name="mock-bart")
        mock_tok = MagicMock()
        mock_model = MagicMock()

        mock_tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tok.decode.return_value = "This laptop offers strong battery life and high performance."
        mock_model.generate.return_value = torch.tensor([[10, 20, 30]])

        summarizer._tokenizer = mock_tok
        summarizer._model = mock_model

        long_review = (
            "The battery life is really impressive, lasting more than 10 hours of continuous web browsing. "
            "Furthermore, the Intel Core i7 processor handles demanding multitasking and compiling code effortlessly. "
            "The keyboard feels responsive and comfortable for long writing sessions."
        )
        summary = summarizer.summarize(long_review)
        assert summary == "This laptop offers strong battery life and high performance."
        mock_model.generate.assert_called_once()

    def test_compare_summarizers(self, multi_sentence_review: str) -> None:
        import torch

        abs_summarizer = AbstractiveSummarizer(model_name="mock-bart")
        mock_tok = MagicMock()
        mock_model = MagicMock()
        mock_tok.return_value = {"input_ids": torch.tensor([[1, 2, 3]])}
        mock_tok.decode.return_value = "High performance laptop with great battery and OLED screen."
        mock_model.generate.return_value = torch.tensor([[10, 20, 30]])
        abs_summarizer._tokenizer = mock_tok
        abs_summarizer._model = mock_model

        ext_summarizer = TextRankSummarizer(num_sentences=2)

        comparison = compare_summarizers(
            [multi_sentence_review],
            reference_summary="High performance laptop with good battery life.",
            extractive_summarizer=ext_summarizer,
            abstractive_summarizer=abs_summarizer,
        )

        assert "extractive" in comparison
        assert "abstractive" in comparison
        assert "summary" in comparison["extractive"]
        assert "summary" in comparison["abstractive"]
        assert "rouge" in comparison["extractive"]
        assert "rouge" in comparison["abstractive"]
        assert comparison["abstractive"]["rouge"]["rouge1"]["fmeasure"] > 0.0

