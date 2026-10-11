"""Aspect-Based Sentiment Analysis (ABSA) scoring module for ReviewIQ.

Scores sentiment for extracted laptop aspects:
- Dual-engine architecture:
  - VADER Lexicon Sentiment Analyzer: Fast rule-based sentiment with polarity compound scores
  - DistilBERT Transformer Classifier: Neural contextual sentiment classification
- Produces normalized scores (-1.0 to +1.0) and categorical labels (positive, negative, neutral)
- Evaluator for SemEval-2014 Task 4 (Laptops) benchmark XML files
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from reviewiq.absa.aspect_extractor import AspectExtractor, ExtractedAspect
from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class AspectSentiment:
    """Sentiment evaluation result for a single aspect."""

    aspect: str
    sentiment: str  # 'positive', 'negative', 'neutral'
    score: float  # Normalized compound score in [-1.0, 1.0]
    confidence: float
    mention_count: int = 1
    evidence_sentences: list[str] = field(default_factory=list)
    positive_ratio: float | None = None

    def to_dict(self) -> dict[str, Any]:
        result = {
            "aspect": self.aspect,
            "sentiment": self.sentiment,
            "score": float(self.score),
            "confidence": float(self.confidence),
            "mention_count": self.mention_count,
            "evidence_sentences": self.evidence_sentences,
        }
        if self.positive_ratio is not None:
            result["positive_ratio"] = float(self.positive_ratio)
        return result

    def __getitem__(self, key: str) -> Any:
        d = self.to_dict()
        if key in d:
            return d[key]
        raise KeyError(key)


class AspectSentimentScorer:
    """Scores sentiment for specific product aspects across reviews.

    Parameters
    ----------
    use_vader:
        Whether to use VADER lexicon analyzer. Default True.
    use_transformer:
        Whether to load Transformer model (DistilBERT). Default False for rapid CPU execution.
    model_name:
        HuggingFace model name for transformer classifier.
    """

    def __init__(
        self,
        use_vader: bool = True,
        use_transformer: bool = False,
        model_name: str | None = None,
    ) -> None:
        cfg = load_config()
        a_cfg = getattr(cfg, "absa", None)

        self.use_vader = use_vader
        self.use_transformer = use_transformer
        self.model_name = (
            model_name
            if model_name is not None
            else getattr(a_cfg, "sentiment_model", "distilbert-base-uncased-finetuned-sst-2-english")
        )

        self.vader = SentimentIntensityAnalyzer() if self.use_vader else None
        self._transformer_pipeline = None
        self.extractor = AspectExtractor()

    def _get_transformer_pipeline(self) -> Any:
        """Lazily load HuggingFace sentiment pipeline."""
        if self._transformer_pipeline is None and self.use_transformer:
            try:
                from transformers import pipeline

                logger.info(f"Loading ABSA transformer pipeline: '{self.model_name}'...")
                self._transformer_pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    device=-1,  # CPU by default
                    truncation=True,
                )
            except Exception as e:
                logger.warning(
                    f"Could not load transformer '{self.model_name}': {e}. Falling back to VADER."
                )
                self._transformer_pipeline = None
        return self._transformer_pipeline

    NEG_OPINIONS: set[str] = {
        "bad", "poor", "terrible", "horrible", "awful", "dreadful", "disappointing",
        "frustrating", "slow", "sluggish", "laggy", "dim", "washed", "flimsy", "fragile",
        "loose", "wobbly", "hot", "loud", "noisy", "heavy", "bulky", "expensive",
        "overpriced", "useless", "broken", "defective", "mushy", "unresponsive", "weak",
    }

    POS_OPINIONS: set[str] = {
        "amazing", "awesome", "excellent", "great", "good", "decent", "fine",
        "satisfactory", "impressive", "outstanding", "superb", "terrific", "fantastic",
        "crisp", "vibrant", "sharp", "sturdy", "solid", "durable", "fast", "quick",
        "speedy", "snappy", "smooth", "quiet", "cool", "light", "portable", "bright",
        "affordable", "cheap", "responsive", "powerful", "incredible", "exceptional",
        "wonderful", "brilliant", "stellar", "lasting",
    }

    def score_sentence(self, aspect_item: ExtractedAspect) -> tuple[str, float, float]:
        """Compute (sentiment_label, compound_score, confidence) for an extracted aspect item.

        Isolates the aspect clause when contrastive conjunctions exist.
        Score range is [-1.0, 1.0].
        """
        sentence = aspect_item.sentence

        # Contrastive clause isolation for compound sentences
        target_clause = sentence
        clauses = re.split(
            r"(?<=[,\s])(?:but|however|although|though|whereas|while|yet|except)\b|[;]",
            sentence,
            flags=re.IGNORECASE,
        )
        for clause in clauses:
            clause_clean = clause.strip()
            if aspect_item.term.lower() in clause_clean.lower():
                target_clause = clause_clean
                break

        transformer = self._get_transformer_pipeline()
        if transformer is not None:
            # Transformer classification
            res = transformer(target_clause)[0]
            label = res["label"].lower()  # 'positive' or 'negative'
            score = float(res["score"])  # confidence in [0.5, 1.0]

            compound = score if "pos" in label else -score
            cat_label = "positive" if compound >= 0.05 else ("negative" if compound <= -0.05 else "neutral")
            return cat_label, compound, score

        # VADER scoring on targeted clause
        assert self.vader is not None
        scores = self.vader.polarity_scores(target_clause)
        compound = float(scores["compound"])

        # Check domain opinion words associated with the aspect
        neg_count = sum(1 for op in aspect_item.opinion_words if op in self.NEG_OPINIONS)
        pos_count = sum(1 for op in aspect_item.opinion_words if op in self.POS_OPINIONS)

        # Domain correction: if aspect has unambiguous negative opinion words and no positive words
        if neg_count > 0 and pos_count == 0 and compound > 0:
            compound = -0.5 if compound < 0.5 else -compound

        # Domain correction: if aspect has unambiguous positive opinion words and no negative words
        if pos_count > 0 and neg_count == 0 and not aspect_item.is_negated and compound < 0:
            compound = 0.5 if abs(compound) < 0.5 else abs(compound)

        # If negation was detected near the aspect, invert positive sentiment
        if aspect_item.is_negated:
            if compound > 0.05:
                compound = -abs(compound)
            elif compound >= -0.05:
                compound = -0.5

        # Determine discrete category
        if compound >= 0.05:
            cat_label = "positive"
            conf = float(scores.get("pos", 0.5) + (compound * 0.5))
        elif compound <= -0.05:
            cat_label = "negative"
            conf = float(scores.get("neg", 0.5) + (abs(compound) * 0.5))
        else:
            cat_label = "neutral"
            conf = float(scores.get("neu", 0.7))

        conf = min(max(conf, 0.5), 1.0)
        return cat_label, compound, conf

    def score_aspect(self, aspect_item: ExtractedAspect) -> AspectSentiment:
        """Compute structured AspectSentiment for an extracted aspect item."""
        cat_label, compound, conf = self.score_sentence(aspect_item)
        return AspectSentiment(
            aspect=aspect_item.aspect,
            sentiment=cat_label,
            score=compound,
            confidence=conf,
            mention_count=1,
            evidence_sentences=[aspect_item.sentence],
        )

    def score_review(self, text: str) -> dict[str, AspectSentiment]:
        """Extract all aspects from review text and score sentiment per aspect.

        Returns
        -------
        Dictionary mapping aspect_name -> AspectSentiment.
        """
        extracted_items = self.extractor.extract_from_text(text)
        if not extracted_items:
            return {}

        aspect_scores: dict[str, list[tuple[float, float, str]]] = {}
        for item in extracted_items:
            cat_label, compound, conf = self.score_sentence(item)
            if item.aspect not in aspect_scores:
                aspect_scores[item.aspect] = []
            aspect_scores[item.aspect].append((compound, conf, item.sentence))

        results: dict[str, AspectSentiment] = {}
        for aspect, score_list in aspect_scores.items():
            mean_score = float(sum(s[0] for s in score_list) / len(score_list))
            mean_conf = float(sum(s[1] for s in score_list) / len(score_list))
            evidence = [s[2] for s in score_list]

            if mean_score >= 0.05:
                sentiment = "positive"
            elif mean_score <= -0.05:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            results[aspect] = AspectSentiment(
                aspect=aspect,
                sentiment=sentiment,
                score=mean_score,
                confidence=mean_conf,
                mention_count=len(score_list),
                evidence_sentences=evidence,
            )

        return results

    def score_corpus(
        self, reviews: Sequence[str]
    ) -> dict[str, AspectSentiment]:
        """Compute aggregated average aspect sentiment scores across a corpus of reviews.

        Returns
        -------
        Dictionary mapping aspect -> AspectSentiment (with dict access compatibility).
        """
        aspect_aggregates: dict[str, list[float]] = {}
        aspect_evidences: dict[str, list[str]] = {}

        for rev in reviews:
            review_aspects = self.score_review(rev)
            for aspect, sent in review_aspects.items():
                if aspect not in aspect_aggregates:
                    aspect_aggregates[aspect] = []
                    aspect_evidences[aspect] = []
                aspect_aggregates[aspect].append(sent.score)
                aspect_evidences[aspect].extend(sent.evidence_sentences)

        summary: dict[str, AspectSentiment] = {}
        for aspect, scores in aspect_aggregates.items():
            if scores:
                mean_val = float(sum(scores) / len(scores))
                pos_count = sum(1 for s in scores if s >= 0.05)
                ratio = float(pos_count / len(scores))
                if mean_val >= 0.05:
                    sent_label = "positive"
                elif mean_val <= -0.05:
                    sent_label = "negative"
                else:
                    sent_label = "neutral"

                summary[aspect] = AspectSentiment(
                    aspect=aspect,
                    sentiment=sent_label,
                    score=mean_val,
                    confidence=ratio,
                    mention_count=len(scores),
                    evidence_sentences=aspect_evidences.get(aspect, [])[:5],
                    positive_ratio=ratio,
                )

        return summary

    def evaluate_semeval(
        self, xml_filepath: str | Path
    ) -> dict[str, Any]:
        """Evaluate ABSA accuracy against official SemEval-2014 Task 4 (Laptops) XML annotations.

        Parameters
        ----------
        xml_filepath:
            Path to SemEval 2014 XML dataset (train or test).

        Returns
        -------
        Dictionary with accuracy, macro_f1, total_terms, total_aspects, and confusion_matrix.
        """
        path = Path(xml_filepath)
        if not path.exists():
            raise FileNotFoundError(f"SemEval XML file not found at {path}")

        tree = ET.parse(path)
        root = tree.getroot()

        y_true: list[str] = []
        y_pred: list[str] = []

        for sentence_elem in root.findall(".//sentence"):
            text_elem = sentence_elem.find("text")
            if text_elem is None or not text_elem.text:
                continue
            text = text_elem.text

            aspect_terms_elem = sentence_elem.find("aspectTerms")
            if aspect_terms_elem is None:
                continue

            for aspect_term in aspect_terms_elem.findall("aspectTerm"):
                term = aspect_term.get("term")
                polarity = aspect_term.get("polarity", "").lower()

                # SemEval uses: positive, negative, neutral, conflict
                if polarity not in ["positive", "negative", "neutral"]:
                    continue

                # Isolate the clause containing the current term to check for negation
                term_clause = text
                clauses = re.split(
                    r"(?<=[,\s])(?:but|however|although|though|whereas|while|yet|except)\b|[;]",
                    text,
                    flags=re.IGNORECASE,
                )
                if term:
                    for clause in clauses:
                        if term.lower() in clause.lower():
                            term_clause = clause.strip()
                            break

                # Run sentiment scoring on the clause containing the term
                is_neg = bool(re.search(r"\bnot\b|n['’]t\b", term_clause, flags=re.IGNORECASE))
                simulated_aspect = ExtractedAspect(
                    aspect="generic",
                    term=term or "",
                    opinion_words=[],
                    sentence=text,
                    is_negated=is_neg,
                )
                pred_label, _, _ = self.score_sentence(simulated_aspect)

                y_true.append(polarity)
                y_pred.append(pred_label)

        if not y_true:
            return {"accuracy": 0.0, "total_terms": 0.0, "total_aspects": 0}

        correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
        accuracy = correct / len(y_true)

        # Macro F1
        classes = ["positive", "negative", "neutral"]
        f1_list = []
        confusion_matrix: dict[str, dict[str, int]] = {
            c_true: {c_pred: 0 for c_pred in classes} for c_true in classes
        }
        for yt, yp in zip(y_true, y_pred):
            if yt in confusion_matrix and yp in confusion_matrix[yt]:
                confusion_matrix[yt][yp] += 1

        for cls in classes:
            tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp == cls)
            fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != cls and yp == cls)
            fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == cls and yp != cls)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
            f1_list.append(f1)

        macro_f1 = sum(f1_list) / len(f1_list)
        return {
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "total_terms": float(len(y_true)),
            "total_aspects": int(len(y_true)),
            "confusion_matrix": confusion_matrix,
        }



if __name__ == "__main__":
    scorer = AspectSentimentScorer()
    sample_review = (
        "The battery life is amazing and easily lasts 14 hours. "
        "The OLED screen is sharp and colorful. "
        "However, the trackpad feels loose and the laptop gets very hot under heavy load."
    )

    results = scorer.score_review(sample_review)
    print("--- ABSA Sentiment Analysis ---")
    for aspect, sent in results.items():
        print(f"[{aspect.upper()}] Sentiment: {sent.sentiment.upper()} (score: {sent.score:+.3f}, conf: {sent.confidence:.2f})")
        print(f"  Evidence: {sent.evidence_sentences}")
