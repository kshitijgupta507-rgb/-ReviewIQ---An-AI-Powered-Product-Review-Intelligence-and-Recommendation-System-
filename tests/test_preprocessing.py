"""Real unit tests for the Phase 1 preprocessing pipeline.

Each module is tested independently so failures are easy to diagnose.
Tests do NOT require the full Amazon dataset — they use the fixtures
defined in conftest.py and short inline strings.

Run with:
    pytest tests/test_preprocessing.py -v
"""

from __future__ import annotations

import pytest


# ─────────────────────────────────────────────────────────────────────────────
# 1. Cleaner
# ─────────────────────────────────────────────────────────────────────────────
class TestCleaner:
    def test_strips_html_tags(self) -> None:
        from reviewiq.preprocessing.cleaner import clean_text

        result = clean_text("<b>Great</b> <i>laptop!</i>")
        assert "<b>" not in result
        assert "Great" in result
        assert "laptop" in result

    def test_strips_urls(self) -> None:
        from reviewiq.preprocessing.cleaner import clean_text

        result = clean_text("Check https://amazon.com/product for details")
        assert "https" not in result
        assert "details" in result

    def test_strips_emails(self) -> None:
        from reviewiq.preprocessing.cleaner import clean_text

        result = clean_text("Contact support@company.com for help")
        assert "@" not in result

    def test_collapses_repeated_punctuation(self) -> None:
        from reviewiq.preprocessing.cleaner import clean_text

        result = clean_text("Terrible!!!!!!")
        assert "!!!!!!" not in result
        assert "!" in result

    def test_handles_empty_string(self) -> None:
        from reviewiq.preprocessing.cleaner import clean_text

        assert clean_text("") == ""
        assert clean_text(None) == ""  # type: ignore[arg-type]

    def test_normalises_unicode_entities(self) -> None:
        from reviewiq.preprocessing.cleaner import clean_text

        result = clean_text("Price &amp; quality are great")
        assert "&amp;" not in result
        assert "&" in result

    def test_unicode_normalization_fi_ligature(self) -> None:
        from reviewiq.preprocessing.cleaner import normalize_unicode

        # ﬁ (U+FB01) should become fi after NFKC
        result = normalize_unicode("proﬁt")
        assert result == "profit"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Emoji Handler
# ─────────────────────────────────────────────────────────────────────────────
class TestEmojiHandler:
    def test_converts_emoji_to_text(self) -> None:
        from reviewiq.preprocessing.emoji_handler import emojis_to_text

        result = emojis_to_text("I love this laptop 😍")
        assert "😍" not in result
        # Should contain some description of the emoji
        assert len(result) > len("I love this laptop ")

    def test_handles_no_emoji(self) -> None:
        from reviewiq.preprocessing.emoji_handler import emojis_to_text

        text = "This is a plain text review."
        assert emojis_to_text(text) == text

    def test_handles_empty_string(self) -> None:
        from reviewiq.preprocessing.emoji_handler import emojis_to_text

        assert emojis_to_text("") == ""

    def test_negative_emoji(self) -> None:
        from reviewiq.preprocessing.emoji_handler import emojis_to_text

        result = emojis_to_text("Battery is bad 😤")
        assert "😤" not in result


# ─────────────────────────────────────────────────────────────────────────────
# 3. Normalizer
# ─────────────────────────────────────────────────────────────────────────────
class TestNormalizer:
    def test_expands_contraction_dont(self) -> None:
        from reviewiq.preprocessing.normalizer import expand_contractions

        result = expand_contractions("I don't like this")
        assert "do not" in result
        assert "don't" not in result

    def test_expands_contraction_wouldnt(self) -> None:
        from reviewiq.preprocessing.normalizer import expand_contractions

        result = expand_contractions("I wouldn't buy this again")
        assert "would not" in result

    def test_expands_wont(self) -> None:
        from reviewiq.preprocessing.normalizer import expand_contractions

        result = expand_contractions("It won't turn on")
        assert "will not" in result

    def test_replaces_slang_gr8(self) -> None:
        from reviewiq.preprocessing.normalizer import replace_slang

        result = replace_slang("gr8 batt life")
        assert "great" in result
        assert "battery" in result

    def test_replaces_slang_luv(self) -> None:
        from reviewiq.preprocessing.normalizer import replace_slang

        result = replace_slang("i luv this laptop")
        assert "love" in result

    def test_lowercase(self) -> None:
        from reviewiq.preprocessing.normalizer import normalize_text

        result = normalize_text("GREAT Battery LIFE", lowercase=True)
        assert result == result.lower()

    def test_normalize_text_combined(self) -> None:
        from reviewiq.preprocessing.normalizer import normalize_text

        result = normalize_text("Gr8 batt, don't buy")
        assert "great" in result
        assert "battery" in result
        assert "do not" in result


# ─────────────────────────────────────────────────────────────────────────────
# 4. Tokenizer
# ─────────────────────────────────────────────────────────────────────────────
class TestTokenizer:
    def test_word_tokenize_basic(self) -> None:
        from reviewiq.preprocessing.tokenizer import word_tokenize

        tokens = word_tokenize("great laptop, but heavy")
        assert "great" in tokens
        assert "laptop" in tokens
        assert "heavy" in tokens

    def test_word_tokenize_empty(self) -> None:
        from reviewiq.preprocessing.tokenizer import word_tokenize

        assert word_tokenize("") == []

    def test_sent_tokenize_multiple_sentences(self) -> None:
        from reviewiq.preprocessing.tokenizer import sent_tokenize

        sents = sent_tokenize("Great display. Battery is poor. Would not buy.")
        assert len(sents) == 3

    def test_pos_tag_returns_tuples(self) -> None:
        from reviewiq.preprocessing.tokenizer import pos_tag

        tags = pos_tag(["great", "battery"])
        assert len(tags) == 2
        assert all(isinstance(t, tuple) and len(t) == 2 for t in tags)

    def test_pos_tag_noun(self) -> None:
        from reviewiq.preprocessing.tokenizer import pos_tag

        tags = dict(pos_tag(["battery", "laptop"]))
        # Both should be tagged as nouns (NN*)
        assert tags["battery"].startswith("N")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Stopwords
# ─────────────────────────────────────────────────────────────────────────────
class TestStopwords:
    def test_removes_common_stopwords(self) -> None:
        from reviewiq.preprocessing.stopwords import get_stopwords, remove_stopwords

        sw = get_stopwords()
        result = remove_stopwords(["the", "laptop", "is", "great"], sw)
        assert "the" not in result
        assert "laptop" in result
        assert "great" in result

    def test_preserves_negation_word(self) -> None:
        from reviewiq.preprocessing.stopwords import get_stopwords, remove_stopwords

        sw = get_stopwords()
        result = remove_stopwords(["i", "do", "not", "recommend"], sw)
        assert "not" in result

    def test_no_stopword_preserves_sentiment_words(self) -> None:
        """'very', 'never' must survive stopword removal."""
        from reviewiq.preprocessing.stopwords import get_stopwords, remove_stopwords

        sw = get_stopwords()
        tokens = ["this", "is", "very", "bad", "never", "buy"]
        result = remove_stopwords(tokens, sw)
        assert "very" in result
        assert "never" in result

    def test_stopword_set_is_frozenset(self) -> None:
        from reviewiq.preprocessing.stopwords import get_stopwords

        sw = get_stopwords()
        assert isinstance(sw, frozenset)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Lemmatizer
# ─────────────────────────────────────────────────────────────────────────────
class TestLemmatizer:
    def test_lemmatizes_plural_noun(self) -> None:
        from reviewiq.preprocessing.lemmatizer import lemmatize_tokens

        result = lemmatize_tokens([("batteries", "NNS")])
        assert result == ["battery"]

    def test_lemmatizes_verb(self) -> None:
        from reviewiq.preprocessing.lemmatizer import lemmatize_tokens

        result = lemmatize_tokens([("running", "VBG")])
        assert result == ["run"]

    def test_does_not_lemmatize_meeting_as_noun(self) -> None:
        """'meeting' as NN should NOT become 'meet'."""
        from reviewiq.preprocessing.lemmatizer import lemmatize_tokens

        result = lemmatize_tokens([("meeting", "NN")])
        assert result == ["meeting"]

    def test_lemmatize_word_list(self) -> None:
        from reviewiq.preprocessing.lemmatizer import lemmatize_word_list

        result = lemmatize_word_list(["batteries", "charges", "displays"])
        assert "battery" in result

    def test_empty_input(self) -> None:
        from reviewiq.preprocessing.lemmatizer import lemmatize_word_list

        assert lemmatize_word_list([]) == []


# ─────────────────────────────────────────────────────────────────────────────
# 7. Negation Handler
# ─────────────────────────────────────────────────────────────────────────────
class TestNegationHandler:
    def test_tags_tokens_after_not(self) -> None:
        from reviewiq.preprocessing.negation_handler import handle_negation

        tokens = ["battery", "is", "not", "great"]
        result = handle_negation(tokens, window=5)
        assert "great_NEG" in result
        assert "not" in result

    def test_stops_at_clause_boundary(self) -> None:
        from reviewiq.preprocessing.negation_handler import handle_negation

        tokens = ["not", "great", ",", "but", "display", "is", "good"]
        result = handle_negation(tokens, window=5)
        assert "great_NEG" in result
        # After the comma, "but", "display", "good" should be untagged
        assert "good" in result
        assert "good_NEG" not in result

    def test_no_negation_unchanged(self) -> None:
        from reviewiq.preprocessing.negation_handler import handle_negation

        tokens = ["great", "battery", "life"]
        result = handle_negation(tokens, window=5)
        assert result == tokens

    def test_has_negation_true(self) -> None:
        from reviewiq.preprocessing.negation_handler import has_negation

        assert has_negation(["i", "would", "not", "buy"]) is True

    def test_has_negation_false(self) -> None:
        from reviewiq.preprocessing.negation_handler import has_negation

        assert has_negation(["great", "laptop"]) is False

    def test_window_limits_tagging(self) -> None:
        from reviewiq.preprocessing.negation_handler import handle_negation

        tokens = ["not", "a", "b", "c", "d", "e", "f"]
        result = handle_negation(tokens, window=3)
        # Only first 3 after "not" get tagged
        assert "a_NEG" in result
        assert "b_NEG" in result
        assert "c_NEG" in result
        assert "d_NEG" not in result


# ─────────────────────────────────────────────────────────────────────────────
# 8. Full Pipeline (integration test)
# ─────────────────────────────────────────────────────────────────────────────
class TestPipeline:
    @pytest.fixture(scope="class")
    def pipeline(self):
        """Create pipeline once for the whole class (slow to init)."""
        from reviewiq.preprocessing.pipeline import PreprocessingPipeline

        return PreprocessingPipeline(spell_correction=False)  # faster for tests

    def test_returns_pipeline_result(self, pipeline) -> None:
        from reviewiq.preprocessing.pipeline import PipelineResult

        result = pipeline.process("This is a great laptop with amazing battery life!")
        assert isinstance(result, PipelineResult)

    def test_clean_tokens_non_empty(self, pipeline) -> None:
        result = pipeline.process("Great laptop, amazing battery life!")
        assert len(result.clean_tokens) > 0

    def test_emoji_converted(self, pipeline) -> None:
        result = pipeline.process("I luv this laptop 😍 battery is bad 😤")
        # Raw text preserved
        assert "😍" in result.raw_text
        # Tokens contain text descriptions, not emoji
        combined = " ".join(result.clean_tokens)
        assert "😍" not in combined

    def test_negation_in_tokens(self, pipeline) -> None:
        result = pipeline.process("Battery is not great but display is good")
        tokens = result.clean_tokens
        neg_tokens = [t for t in tokens if "_NEG" in t]
        assert len(neg_tokens) > 0
        assert result.has_negation is True

    def test_contraction_expanded(self, pipeline) -> None:
        result = pipeline.process("I wouldn't recommend this product")
        # "wouldn't" → "would not" → negation tagging
        assert result.has_negation is True

    def test_sentences_populated(self, pipeline) -> None:
        result = pipeline.process("Battery is great. Display is poor. Build quality is okay.")
        assert len(result.sentences) >= 2

    def test_transformer_text_lowercase(self, pipeline) -> None:
        result = pipeline.process("GREAT BATTERY LIFE")
        assert result.transformer_text == result.transformer_text.lower()

    def test_handles_empty_string(self, pipeline) -> None:
        result = pipeline.process("")
        assert result.clean_tokens == []
        assert result.transformer_text == ""

    def test_handles_pure_punctuation(self, pipeline) -> None:
        result = pipeline.process("... !!!! ???")
        assert result.clean_tokens == []

    def test_process_dataframe(self, pipeline, sample_reviews) -> None:
        import pandas as pd

        df = pd.DataFrame(sample_reviews)
        df = df.rename(columns={"reviewText": "text"})
        result = pipeline.process_dataframe(df, text_column="text", show_progress=False)
        assert "clean_tokens" in result.columns
        assert "transformer_text" in result.columns
        assert "has_negation" in result.columns
        assert len(result) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 9. Data Schema
# ─────────────────────────────────────────────────────────────────────────────
class TestDataSchema:
    def test_review_sentiment_label_positive(self) -> None:
        from reviewiq.data.schema import Review

        r = Review(asin="B001", reviewer_id="U1", text="great", rating=5.0)
        assert r.sentiment_label == "positive"

    def test_review_sentiment_label_negative(self) -> None:
        from reviewiq.data.schema import Review

        r = Review(asin="B001", reviewer_id="U1", text="bad", rating=1.0)
        assert r.sentiment_label == "negative"

    def test_review_sentiment_label_neutral(self) -> None:
        from reviewiq.data.schema import Review

        r = Review(asin="B001", reviewer_id="U1", text="ok", rating=3.0)
        assert r.sentiment_label == "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Loader (no real data required — tests schema validation logic)
# ─────────────────────────────────────────────────────────────────────────────
class TestLoader:
    def test_add_sentiment_label(self) -> None:
        import pandas as pd

        from reviewiq.data.loader import add_sentiment_label

        df = pd.DataFrame({"rating": [5.0, 1.0, 3.0, 4.0, 2.0]})
        df = add_sentiment_label(df)
        assert list(df["sentiment_label"]) == [
            "positive",
            "negative",
            "neutral",
            "positive",
            "negative",
        ]

    def test_validation_error_on_missing_columns(self) -> None:
        import pandas as pd
        import pytest

        from reviewiq.data.loader import DataValidationError, _validate, _REVIEW_SCHEMA

        df = pd.DataFrame({"some_col": [1, 2, 3]})
        with pytest.raises(DataValidationError):
            _validate(df, _REVIEW_SCHEMA, "test")

    def test_load_sample_raises_descriptive_error_when_missing(self) -> None:
        import pytest
        from reviewiq.data.loader import load_sample

        with pytest.raises(FileNotFoundError, match="Sample data missing"):
            load_sample()


# ─────────────────────────────────────────────────────────────────────────────
# 11. Downloader and Utility Verification
# ─────────────────────────────────────────────────────────────────────────────
class TestDownloader:
    def test_is_laptop_word_boundary_and_singular_forms(self) -> None:
        from reviewiq.data.download import _is_laptop

        include = ["laptop", "notebooks"]
        exclude = ["accessories", "cases", "bags", "sleeves", "charger", "adapter", "mouse", "keyboard"]

        # Matches singular "notebook" even though "notebooks" is configured
        assert _is_laptop(["Electronics", "Computers", "Notebook"], include, exclude) is True
        # Whole word match: "showcase" should NOT match "cases" exclude
        assert _is_laptop(["Laptop", "Showcase"], include, exclude) is True
        # "Laptop Bag" should be excluded via singular "bag" of "bags"
        assert _is_laptop(["Laptop", "Laptop Bag"], include, exclude) is False

    def test_progress_reader_bounded_read(self) -> None:
        from unittest.mock import MagicMock
        from reviewiq.data.download import _ProgressReader

        mock_resp = MagicMock()
        mock_resp.iter_content.return_value = iter([b"abcdefghij", b"klmnopqrst"])
        mock_bar = MagicMock()

        reader = _ProgressReader(mock_resp, mock_bar)
        chunk1 = reader.read(5)
        assert chunk1 == b"abcde"
        assert len(chunk1) == 5

        chunk2 = reader.read(8)
        assert chunk2 == b"fghijklm"
        assert len(chunk2) == 8

        # Unbounded read gets all remaining bytes
        rest = reader.read(-1)
        assert rest == b"nopqrst"

    def test_logging_config_resolves_and_exists(self) -> None:
        from reviewiq.utils.logger import _LOGGING_CONFIG

        assert _LOGGING_CONFIG.exists(), f"{_LOGGING_CONFIG} does not exist"
        assert _LOGGING_CONFIG.name == "logging.yaml"


class TestFixes:
    def test_strip_emoji_fallback_preserves_chinese(self) -> None:
        from reviewiq.preprocessing.emoji_handler import _strip_emoji_fallback

        text = "这款笔记本电脑非常好！😍 强烈推荐！"
        cleaned = _strip_emoji_fallback(text)
        assert "这款笔记本电脑非常好" in cleaned
        assert "强烈推荐" in cleaned
        assert "😍" not in cleaned

    def test_sentiment_words_excluded_from_custom_stopwords(self) -> None:
        from reviewiq.preprocessing.stopwords import get_stopwords

        sw = get_stopwords()
        for term in ["recommend", "recommended", "worth", "like"]:
            assert term not in sw, f"{term} should not be in stopwords"

    def test_process_dataframe_preserves_index(self) -> None:
        import pandas as pd
        from reviewiq.preprocessing.pipeline import PreprocessingPipeline

        pipe = PreprocessingPipeline(spell_correction=False, min_tokens=2)
        df = pd.DataFrame(
            {"text": ["Great laptop with excellent vibrant screen", "Terrible battery life and awful build", "..."]},
            index=[10, 20, 30],
        )
        processed = pipe.process_dataframe(df, show_progress=False)
        # Row 30 has 0 tokens and is dropped; remaining rows retain original indices [10, 20] (not reset to 0, 1)
        assert list(processed.index) == [10, 20]

    def test_negation_scope_delimited_by_punctuation(self) -> None:
        from reviewiq.preprocessing.pipeline import PreprocessingPipeline

        pipe = PreprocessingPipeline(spell_correction=False)
        res = pipe.process("not good, display is bright")
        tokens = res.clean_tokens
        assert "good_NEG" in tokens
        assert "bright_NEG" not in tokens
        assert "bright" in tokens

