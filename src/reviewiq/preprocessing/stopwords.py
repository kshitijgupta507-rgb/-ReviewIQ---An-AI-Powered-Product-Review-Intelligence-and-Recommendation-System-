"""Stopword management for ReviewIQ.

Combines NLTK's English stopword list with a custom domain-aware list.
Critically, sentiment-bearing words that NLTK marks as stopwords are
kept (e.g., "not", "no", "never") because they are essential for
negation handling and sentiment analysis.

Usage
-----
    from reviewiq.preprocessing.stopwords import get_stopwords, remove_stopwords
    sw = get_stopwords()
    tokens = remove_stopwords(["this", "laptop", "is", "not", "bad"], sw)
    # → ["laptop", "not", "bad"]   # "not" preserved
"""

from __future__ import annotations

from reviewiq.preprocessing.tokenizer import _ensure_nltk_resources


# Words that NLTK lists as stopwords but that carry sentiment / meaning
# in product reviews. We remove these from the stopword filter.
_SENTIMENT_PRESERVING: frozenset[str] = frozenset(
    [
        # Negation — critical for sentiment
        "no",
        "not",
        "nor",
        "never",
        "neither",
        "nothing",
        "nowhere",
        "nobody",
        "none",
        # Degree modifiers — change intensity
        "very",
        "too",
        "more",
        "most",
        "less",
        "least",
        "only",
        "just",
        "almost",
        "quite",
        "rather",
        "really",
        "highly",
        "extremely",
        "barely",
        "hardly",
        "nearly",
        "perfectly",
        "completely",
        "absolutely",
        # Comparatives that affect recommendation meaning
        "better",
        "worse",
        "best",
        "worst",
        "good",
        "bad",
    ]
)

# Extra domain stopwords — common review filler not in NLTK
_CUSTOM_STOPWORDS: frozenset[str] = frozenset(
    [
        # Review boilerplate
        "product",
        "item",
        "purchase",
        "bought",
        "buy",
        "order",
        "ordered",
        "received",
        "arrived",
        "delivery",
        "shipping",
        "amazon",
        "seller",
        "review",
        "star",
        "stars",
        "rating",
        "price",
        "paid",
        "cost",
        "money",
        "value",
        "use",
        "used",
        "using",
        "get",
        "got",
        "one",
        "two",
        "first",
        "day",
        "week",
        "month",
        "year",
        "time",
        "also",
        "even",
        "well",
        "still",
        "back",
        "work",
        "works",
        "working",
        "need",
        "needs",
        "way",
        "thing",
        "things",
        "make",
        "made",
        "look",
        "looks",
        "feel",
        "feels",
        "come",
        "came",
        "go",
        "goes",
        "went",
    ]
)

_cached_stopwords: frozenset[str] | None = None


def get_stopwords() -> frozenset[str]:
    """Return the combined stopword set.

    Builds the set once and caches it for the process lifetime.
    NLTK stopwords minus sentiment-preserving words, plus custom domain words.

    Returns
    -------
    frozenset[str]
        Lowercase stopword strings.
    """
    global _cached_stopwords
    if _cached_stopwords is not None:
        return _cached_stopwords

    _ensure_nltk_resources()
    import nltk

    nltk_sw = frozenset(nltk.corpus.stopwords.words("english"))
    # Remove sentiment-preserving words from NLTK's list
    filtered_nltk = nltk_sw - _SENTIMENT_PRESERVING
    # Add custom domain stopwords
    _cached_stopwords = filtered_nltk | _CUSTOM_STOPWORDS
    return _cached_stopwords


def remove_stopwords(
    tokens: list[str],
    stopwords: frozenset[str] | None = None,
    *,
    keep_negation: bool = True,
) -> list[str]:
    """Remove stopwords from a token list.

    Parameters
    ----------
    tokens:
        List of lowercase word tokens.
    stopwords:
        Stopword set to use. If None, uses ``get_stopwords()``.
    keep_negation:
        If True (default), always preserve "not", "no", "never" regardless
        of the stopword set. Useful safety net.

    Returns
    -------
    list[str]
        Filtered token list.
    """
    if stopwords is None:
        stopwords = get_stopwords()

    negation_words = {"not", "no", "never", "nor", "neither"} if keep_negation else set()

    result = []
    for t in tokens:
        if t.endswith("_NEG"):
            result.append(t)
        elif t in negation_words or t not in stopwords:
            result.append(t)
    return result
