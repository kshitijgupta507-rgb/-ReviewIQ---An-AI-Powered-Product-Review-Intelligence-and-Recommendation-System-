"""Abstractive text summarization module for ReviewIQ.

Implements neural sequence-to-sequence summarization using HuggingFace Transformers
(e.g., facebook/bart-large-cnn or sshleifer/distilbart-cnn-12-6):
- Concatenates and prepares customer review text up to model token budget
- Generates fluent abstractive summaries using beam search
- Computes ROUGE-1, ROUGE-2, and ROUGE-L precision, recall, and F1 metrics
- Supports head-to-head comparison between TextRank (extractive) and BART (abstractive)
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

import torch
from rouge_score import rouge_scorer

from reviewiq.summarization.textrank import TextRankSummarizer
from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)


def evaluate_rouge(
    reference: str,
    hypothesis: str,
    metrics: Sequence[str] = ("rouge1", "rouge2", "rougeL"),
) -> dict[str, dict[str, float]]:
    """Compute ROUGE scores comparing a hypothesis summary against a reference.

    Parameters
    ----------
    reference:
        Ground truth or reference summary text.
    hypothesis:
        Generated candidate summary text.
    metrics:
        Tuple of ROUGE metrics to compute. Default ('rouge1', 'rouge2', 'rougeL').

    Returns
    -------
    dict[str, dict[str, float]]:
        Dictionary mapping metric names to precision, recall, and fmeasure.
    """
    if not reference.strip() or not hypothesis.strip():
        return {
            m: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
            for m in metrics
        }

    scorer = rouge_scorer.RougeScorer(list(metrics), use_stemmer=True)
    scores = scorer.score(reference, hypothesis)

    return {
        metric: {
            "precision": round(score.precision, 4),
            "recall": round(score.recall, 4),
            "fmeasure": round(score.fmeasure, 4),
        }
        for metric, score in scores.items()
    }


def evaluate_rouge_batch(
    references: Sequence[str],
    hypotheses: Sequence[str],
    metrics: Sequence[str] = ("rouge1", "rouge2", "rougeL"),
) -> dict[str, dict[str, float]]:
    """Compute average ROUGE scores across multiple reference/hypothesis pairs.

    Parameters
    ----------
    references:
        List of reference summary texts.
    hypotheses:
        List of generated candidate summary texts.
    metrics:
        Tuple of ROUGE metrics to compute.

    Returns
    -------
    dict[str, dict[str, float]]:
        Macro-averaged precision, recall, and fmeasure for each metric.
    """
    if len(references) != len(hypotheses):
        raise ValueError("Number of references must match number of hypotheses")

    if not references:
        return {
            m: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0}
            for m in metrics
        }

    accum: dict[str, dict[str, float]] = {
        m: {"precision": 0.0, "recall": 0.0, "fmeasure": 0.0} for m in metrics
    }

    n = len(references)
    for ref, hyp in zip(references, hypotheses):
        pair_scores = evaluate_rouge(ref, hyp, metrics=metrics)
        for m in metrics:
            accum[m]["precision"] += pair_scores[m]["precision"]
            accum[m]["recall"] += pair_scores[m]["recall"]
            accum[m]["fmeasure"] += pair_scores[m]["fmeasure"]

    return {
        m: {
            "precision": round(accum[m]["precision"] / n, 4),
            "recall": round(accum[m]["recall"] / n, 4),
            "fmeasure": round(accum[m]["fmeasure"] / n, 4),
        }
        for m in metrics
    }


class AbstractiveSummarizer:
    """Abstractive review summarizer using seq2seq Transformer models.

    Parameters
    ----------
    model_name:
        HuggingFace model identifier (e.g. 'facebook/bart-large-cnn' or 'sshleifer/distilbart-cnn-12-6').
    max_input_length:
        Maximum input tokens before truncation. Default 1024.
    max_output_length:
        Maximum output tokens generated. Default 150.
    min_output_length:
        Minimum output tokens generated. Default 50.
    num_beams:
        Number of beams for beam search decoding. Default 4.
    device:
        Torch device identifier ('cuda', 'cpu', or None for auto-detection).
    """

    def __init__(
        self,
        model_name: str | None = None,
        max_input_length: int | None = None,
        max_output_length: int | None = None,
        min_output_length: int | None = None,
        num_beams: int | None = None,
        device: str | None = None,
    ) -> None:
        cfg = load_config().get("summarization", {}).get("abstractive", {})
        self.model_name = model_name or cfg.get("model_name", "facebook/bart-large-cnn")
        self.max_input_length = max_input_length or cfg.get("max_input_length", 1024)
        self.max_output_length = max_output_length or cfg.get("max_output_length", 150)
        self.min_output_length = min_output_length or cfg.get("min_output_length", 50)
        self.num_beams = num_beams or cfg.get("num_beams", 4)

        if device is not None:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self._pipeline = None
        self._tokenizer = None
        self._model = None

    def _load_model(self) -> None:
        """Lazily initialize and load the HuggingFace transformer tokenizer and model."""
        if self._tokenizer is not None and self._model is not None:
            return

        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        logger.info("Loading abstractive summarization model: %s on %s", self.model_name, self.device)
        try:
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name).to(self.device)
            self._model.eval()
        except Exception as exc:
            logger.warning(
                "Failed to load %s (%s). Attempting fallback to 'sshleifer/distilbart-cnn-12-6'",
                self.model_name,
                exc,
            )
            fallback_model = "sshleifer/distilbart-cnn-12-6"
            try:
                self._tokenizer = AutoTokenizer.from_pretrained(fallback_model)
                self._model = AutoModelForSeq2SeqLM.from_pretrained(fallback_model).to(self.device)
                self._model.eval()
                self.model_name = fallback_model
            except Exception:
                raise exc

    def summarize(
        self,
        text_or_reviews: str | Sequence[str],
        max_length: int | None = None,
        min_length: int | None = None,
    ) -> str:
        """Generate an abstractive summary of input text or multiple reviews.

        Parameters
        ----------
        text_or_reviews:
            Single review text string, or sequence of review strings.
        max_length:
            Max generated tokens. Defaults to configured max_output_length.
        min_length:
            Min generated tokens. Defaults to configured min_output_length.

        Returns
        -------
        str:
            Generated abstractive summary text.
        """
        # Format and aggregate input text
        if isinstance(text_or_reviews, str):
            input_text = text_or_reviews.strip()
        else:
            input_text = " ".join(s.strip() for s in text_or_reviews if s and s.strip())

        if not input_text:
            return ""

        # Check if input text is very brief; return as-is
        words = input_text.split()
        if len(words) < 25:
            return input_text

        self._load_model()
        assert self._tokenizer is not None and self._model is not None

        gen_max = max_length or self.max_output_length
        gen_min = min_length or min(self.min_output_length, max(10, len(words) // 2))

        # Clamp min_length to avoid generation conflicts
        if gen_min >= gen_max:
            gen_min = max(5, gen_max // 2)

        try:
            inputs = self._tokenizer(
                input_text,
                max_length=self.max_input_length,
                truncation=True,
                return_tensors="pt",
            )
            raw_ids = inputs["input_ids"] if isinstance(inputs, dict) else inputs.input_ids
            input_ids = raw_ids.to(self.device) if hasattr(raw_ids, "to") else raw_ids

            with torch.no_grad():
                summary_ids = self._model.generate(
                    input_ids,
                    max_length=gen_max,
                    min_length=gen_min,
                    num_beams=self.num_beams,
                    length_penalty=2.0,
                    early_stopping=True,
                    no_repeat_ngram_size=3,
                )

            summary = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True).strip()
            return summary
        except Exception as exc:
            logger.error("Abstractive summarization failed: %s; falling back to lead sentences", exc)
            raw_sents = re.split(r"(?<=[.!?])\s+|\n+", input_text)
            lead_sents: list[str] = []
            word_count = 0
            for s in raw_sents:
                s_clean = s.strip()
                if not s_clean:
                    continue
                lead_sents.append(s_clean)
                word_count += len(s_clean.split())
                if word_count >= gen_max or len(lead_sents) >= 3:
                    break
            return " ".join(lead_sents) if lead_sents else input_text


def compare_summarizers(
    reviews: Sequence[str],
    reference_summary: str | None = None,
    extractive_summarizer: TextRankSummarizer | None = None,
    abstractive_summarizer: AbstractiveSummarizer | None = None,
) -> dict[str, Any]:
    """Run both TextRank and Abstractive summarizers side-by-side for comparison.

    Parameters
    ----------
    reviews:
        Input reviews to summarize.
    reference_summary:
        Optional ground-truth reference summary for ROUGE score computation.
    extractive_summarizer:
        Optional instance of TextRankSummarizer.
    abstractive_summarizer:
        Optional instance of AbstractiveSummarizer.

    Returns
    -------
    dict[str, Any]:
        Dictionary with extractive and abstractive summaries, word counts, and ROUGE metrics.
    """
    ext_sum = extractive_summarizer or TextRankSummarizer()
    abs_sum = abstractive_summarizer or AbstractiveSummarizer()

    ext_text = ext_sum.summarize(reviews)
    abs_text = abs_sum.summarize(reviews)

    results: dict[str, Any] = {
        "extractive": {
            "method": "TextRank (PageRank on TF-IDF cosine graph)",
            "summary": ext_text,
            "word_count": len(ext_text.split()),
            "sentence_count": len(ext_sum.split_sentences(ext_text)),
        },
        "abstractive": {
            "method": f"Seq2Seq Transformer ({abs_sum.model_name})",
            "summary": abs_text,
            "word_count": len(abs_text.split()),
            "sentence_count": len(ext_sum.split_sentences(abs_text)),
        },
    }

    if reference_summary and reference_summary.strip():
        results["extractive"]["rouge"] = evaluate_rouge(reference_summary, ext_text)
        results["abstractive"]["rouge"] = evaluate_rouge(reference_summary, abs_text)

    return results
