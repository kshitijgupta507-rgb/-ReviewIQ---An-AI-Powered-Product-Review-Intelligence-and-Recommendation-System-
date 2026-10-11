"""Full preprocessing pipeline for ReviewIQ.

This module orchestrates all preprocessing steps in the correct order
and is the single entry point used by every downstream module.

Pipeline order
--------------
1. Emoji → text          (emoji_handler)
2. HTML / URL cleaning   (cleaner)
3. Contraction expansion (normalizer)
4. Slang replacement     (normalizer)
5. Lowercase             (normalizer)
6. Sentence tokenization (tokenizer)
7. Word tokenization     (tokenizer)
8. Negation handling    (negation_handler — runs before punct removal to keep clause boundaries)
9. Punctuation filter    (pipeline — strips non-alpha tokens while keeping negation-marked tokens)
10. Stopword removal     (stopwords)
11. POS tagging          (tokenizer)
12. Lemmatization       (lemmatizer)
13. Spell correction    (spell_corrector — optional, config-controlled)

Two token lists are produced:
- ``clean_tokens``: lemmatized, stopwords removed, with negation marks.
  Used for BoW, TF-IDF, Word2Vec, classical classifiers.
- ``transformer_text``: lowercased but otherwise minimally processed.
  Used for DistilBERT and other transformers that do their own tokenization.

Usage
-----
    from reviewiq.preprocessing.pipeline import PreprocessingPipeline
    pipe = PreprocessingPipeline()
    result = pipe.process("I wouldn't buy this again. Battery DIES in 2 hrs 😤")
    print(result.clean_tokens)
    # ['would', 'not_NEG', 'buy_NEG', 'battery', 'die']
    print(result.transformer_text)
    # 'i would not buy this again . battery dies in 2 hours face with steam from nose'
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from tqdm import tqdm

from reviewiq.preprocessing.cleaner import clean_text
from reviewiq.preprocessing.emoji_handler import emojis_to_text
from reviewiq.preprocessing.lemmatizer import lemmatize_tokens
from reviewiq.preprocessing.negation_handler import handle_negation, has_negation
from reviewiq.preprocessing.normalizer import normalize_text
from reviewiq.preprocessing.spell_corrector import SpellCorrector
from reviewiq.preprocessing.stopwords import get_stopwords, remove_stopwords
from reviewiq.preprocessing.tokenizer import pos_tag, word_tokenize
from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)

# Tokens that are purely punctuation or single characters — remove from classical list
_PUNCT_RE = re.compile(r"^[^\w]+$")


@dataclass
class PipelineResult:
    """Output of the preprocessing pipeline for a single review."""

    raw_text: str
    """Original unmodified review text."""

    transformer_text: str
    """Minimally processed text for transformer models."""

    clean_tokens: list[str]
    """Fully processed tokens for classical NLP models.

    Includes:
    - Lemmatization
    - Stopword removal
    - _NEG suffix for negated tokens
    """

    sentences: list[str] = field(default_factory=list)
    """Original sentences (used by TextRank summarization)."""

    has_negation: bool = False
    """True if the review contains at least one negation word."""

    detected_language: str = "en"
    """ISO 639-1 language code detected by langdetect."""

    word_count: int = 0
    """Word count of the transformer_text (post-normalization)."""


class PreprocessingPipeline:
    """Full text preprocessing pipeline.

    Parameters
    ----------
    spell_correction:
        Enable SymSpell spell correction. Adds ~2–3 ms per review.
        Controlled by ``configs/default.yaml`` preprocessing.spell_correction.
    negation_window:
        Number of tokens to tag after a negation word. Default 5.
    use_pos_tagging:
        Use POS-aware lemmatization. Default True.
    min_tokens:
        Minimum number of clean tokens for a result to be considered valid.
        Reviews below this threshold will still be returned but flagged.
    language_detection:
        Run langdetect to filter non-English reviews. Default True.

    Examples
    --------
    >>> pipe = PreprocessingPipeline()
    >>> result = pipe.process("Gr8 batt life! Not bad at all 😍")
    >>> result.clean_tokens
    ['great', 'battery', 'life', 'not', 'bad_NEG']
    """

    def __init__(
        self,
        spell_correction: bool = True,
        negation_window: int = 5,
        use_pos_tagging: bool = True,
        min_tokens: int = 3,
        language_detection: bool = True,
    ) -> None:
        self._spell_correction = spell_correction
        self._negation_window = negation_window
        self._use_pos_tagging = use_pos_tagging
        self._min_tokens = min_tokens
        self._language_detection = language_detection

        self._stopwords = get_stopwords()
        self._spell_corrector = SpellCorrector() if spell_correction else None
        logger.debug(
            "PreprocessingPipeline initialised (spell=%s, neg_window=%d, pos=%s).",
            spell_correction,
            negation_window,
            use_pos_tagging,
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    def _detect_language(self, text: str) -> str:
        """Detect language of text. Returns ISO code or 'unknown'."""
        try:
            from langdetect import detect, LangDetectException
            return detect(text)
        except Exception:
            return "unknown"

    def _filter_punct_tokens(self, tokens: list[str]) -> list[str]:
        """Remove pure-punctuation tokens while preserving words and negation-marked tokens."""
        filtered = []
        for t in tokens:
            if t.endswith("_NEG"):
                base = t[:-4]
                if not _PUNCT_RE.match(base) and len(base) >= 1:
                    filtered.append(t)
            else:
                if not _PUNCT_RE.match(t) and len(t) > 1:
                    filtered.append(t)
        return filtered

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, text: str) -> PipelineResult:
        """Run the full pipeline on a single review string.

        Parameters
        ----------
        text:
            Raw review text.

        Returns
        -------
        PipelineResult
            Structured result with clean_tokens and transformer_text.
        """
        if not text or not isinstance(text, str):
            return PipelineResult(
                raw_text=text or "",
                transformer_text="",
                clean_tokens=[],
            )

        raw_text = text

        # ── Step 1: Emoji → text ──────────────────────────────────────────────
        text = emojis_to_text(text)

        # ── Step 2: HTML / URL / email cleaning ──────────────────────────────
        text = clean_text(text)

        # ── Step 3 & 4: Contraction expansion + slang replacement ─────────────
        text = normalize_text(text, lowercase=False)  # keep case for langdetect

        # ── Language detection (before lowercasing for accuracy) ───────────────
        lang = "en"
        if self._language_detection and len(text) > 20:
            lang = self._detect_language(text)

        # ── Step 5: Lowercase ─────────────────────────────────────────────────
        text_lower = text.lower()

        # ── Step 6: Sentence tokenization (for TextRank) ──────────────────────
        from reviewiq.preprocessing.tokenizer import sent_tokenize
        sentences = sent_tokenize(text)   # use original case for sentences

        # ── Step 7: Word tokenization ─────────────────────────────────────────
        tokens = word_tokenize(text_lower)

        # Transformer text = lowercase sentence, minimally cleaned
        transformer_text = " ".join(tokens)

        # ── Step 8: Negation handling (runs before punctuation removal) ─────
        negation_present = has_negation(tokens)
        tokens = handle_negation(tokens, window=self._negation_window)

        # ── Step 9: Remove punctuation tokens ────────────────────────────────
        tokens = self._filter_punct_tokens(tokens)

        # ── Step 10: Stopword removal ──────────────────────────────────────────
        tokens = remove_stopwords(tokens, self._stopwords)

        # ── Step 11: POS tagging + Lemmatization ──────────────────────────────
        if self._use_pos_tagging:
            tagged = pos_tag(tokens)
            tokens = lemmatize_tokens(tagged)
        # else: skip POS tagging for speed (tokens remain un-lemmatized)

        # ── Step 12: Spell correction (optional) ──────────────────────────────
        if self._spell_corrector is not None:
            tokens = self._spell_corrector.correct_tokens(tokens)

        return PipelineResult(
            raw_text=raw_text,
            transformer_text=transformer_text,
            clean_tokens=tokens,
            sentences=sentences,
            has_negation=negation_present,
            detected_language=lang,
            word_count=len(transformer_text.split()),
        )

    def process_dataframe(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
        *,
        show_progress: bool = True,
        language_filter: Optional[str] = "en",
    ) -> pd.DataFrame:
        """Run the pipeline on every row of a DataFrame.

        Adds columns: ``clean_tokens``, ``transformer_text``, ``sentences``,
        ``has_negation``, ``detected_language``, ``word_count``.

        Parameters
        ----------
        df:
            DataFrame with a text column.
        text_column:
            Name of the column containing raw review text.
        show_progress:
            Show a tqdm progress bar. Default True.
        language_filter:
            If set, drop rows where detected language ≠ this value.
            Set to None to keep all languages.

        Returns
        -------
        pd.DataFrame
            Input DataFrame with new preprocessing columns added.
            Original index is preserved.
        """
        logger.info(
            "Running preprocessing pipeline on %d reviews (spell=%s)...",
            len(df),
            self._spell_correction,
        )

        results = []
        iterator = df[text_column].fillna("").tolist()
        if show_progress:
            iterator = tqdm(iterator, desc="Preprocessing", unit="review")

        for text in iterator:
            results.append(self.process(text))

        df = df.copy()
        df["clean_tokens"] = [r.clean_tokens for r in results]
        df["transformer_text"] = [r.transformer_text for r in results]
        df["sentences"] = [r.sentences for r in results]
        df["has_negation"] = [r.has_negation for r in results]
        df["detected_language"] = [r.detected_language for r in results]
        df["word_count"] = [r.word_count for r in results]

        before = len(df)
        if language_filter:
            df = df[df["detected_language"] == language_filter]
            dropped = before - len(df)
            if dropped:
                logger.info("Dropped %d non-%s reviews.", dropped, language_filter)

        # Drop reviews with too few tokens (preserve original index)
        df = df[df["clean_tokens"].map(len) >= self._min_tokens]
        logger.info("Pipeline complete. %d reviews after filtering.", len(df))
        return df

    @classmethod
    def from_config(cls) -> "PreprocessingPipeline":
        """Create a pipeline instance from ``configs/default.yaml``.

        Returns
        -------
        PreprocessingPipeline
            Configured pipeline instance.
        """
        cfg = load_config()
        pp = cfg.preprocessing
        return cls(
            spell_correction=pp.spell_correction,
            negation_window=pp.negation_window,
            use_pos_tagging=pp.use_pos_tagging,
        )


# ── CLI entry point ───────────────────────────────────────────────────────────
def _main() -> None:
    """Run preprocessing pipeline from command line.

    Usage:
        python -m reviewiq.preprocessing.pipeline \
            --input data/interim/laptop_reviews_raw.parquet \
            --output data/processed/laptop_reviews.parquet
    """
    import argparse

    from reviewiq.utils.logger import setup_logging
    from reviewiq.utils.seeds import fix_seeds

    setup_logging()
    fix_seeds()

    parser = argparse.ArgumentParser(description="ReviewIQ preprocessing pipeline")
    parser.add_argument("--input", required=True, help="Input parquet file")
    parser.add_argument(
        "--output",
        default="data/processed/laptop_reviews.parquet",
        help="Output parquet file",
    )
    parser.add_argument(
        "--no-spell",
        action="store_true",
        help="Disable spell correction (faster)",
    )
    args = parser.parse_args()

    df = pd.read_parquet(args.input)
    logger.info("Loaded %d rows from %s.", len(df), args.input)

    pipe = PreprocessingPipeline(spell_correction=not args.no_spell)
    df_processed = pipe.process_dataframe(df)

    import pathlib
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_processed.to_parquet(args.output, index=False)
    logger.info("Saved %d processed reviews to %s.", len(df_processed), args.output)


if __name__ == "__main__":
    _main()
