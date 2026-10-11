"""Tokenization wrappers for word and sentence splitting.

Uses NLTK's ``word_tokenize`` (Penn Treebank tokenizer) which handles
punctuation attachment correctly — important for negation detection
since "n't" is split from its host verb ("wouldn't" → ["would", "n't"]).

Sentence tokenization uses ``sent_tokenize`` (Punkt model) which is
needed for TextRank summarization (Phase 5).

Usage
-----
    from reviewiq.preprocessing.tokenizer import word_tokenize, sent_tokenize
    tokens = word_tokenize("It's a great laptop, but battery life is poor.")
    # → ['It', "'s", 'a', 'great', 'laptop', ',', 'but', 'battery', 'life', 'is', 'poor', '.']
"""

from __future__ import annotations

import threading

import nltk


# ─── NLTK resource bootstrap ──────────────────────────────────────────────────
# Download only what is needed; skip if already present.
_NLTK_LOCK = threading.Lock()
_NLTK_READY = False


def _ensure_nltk_resources() -> None:
    """Download required NLTK resources once per process."""
    global _NLTK_READY
    if _NLTK_READY:
        return
    with _NLTK_LOCK:
        if _NLTK_READY:
            return
        for resource, path in [
            ("punkt_tab", "tokenizers/punkt_tab"),
            ("punkt", "tokenizers/punkt"),
            ("averaged_perceptron_tagger_eng", "taggers/averaged_perceptron_tagger_eng"),
            ("averaged_perceptron_tagger", "taggers/averaged_perceptron_tagger"),
            ("wordnet", "corpora/wordnet"),
            ("stopwords", "corpora/stopwords"),
            ("omw-1.4", "corpora/omw-1.4"),
        ]:
            try:
                nltk.data.find(path)
            except LookupError:
                nltk.download(resource, quiet=True)
        _NLTK_READY = True


def word_tokenize(text: str) -> list[str]:
    """Tokenize text into words using NLTK's Penn Treebank tokenizer.

    Parameters
    ----------
    text:
        Normalized, cleaned text string.

    Returns
    -------
    list[str]
        List of tokens including punctuation marks.

    Examples
    --------
    >>> word_tokenize("great battery, but overheats")
    ['great', 'battery', ',', 'but', 'overheats']
    """
    _ensure_nltk_resources()
    if not text:
        return []
    return nltk.word_tokenize(text)


def sent_tokenize(text: str) -> list[str]:
    """Split text into sentences using NLTK's Punkt sentence tokenizer.

    Parameters
    ----------
    text:
        Multi-sentence text string.

    Returns
    -------
    list[str]
        List of sentence strings.

    Examples
    --------
    >>> sent_tokenize("Great display. Battery life is awful. Would not buy.")
    ['Great display.', 'Battery life is awful.', 'Would not buy.']
    """
    _ensure_nltk_resources()
    if not text:
        return []
    return nltk.sent_tokenize(text)


def pos_tag(tokens: list[str]) -> list[tuple[str, str]]:
    """Assign POS tags to a list of tokens.

    Uses NLTK's averaged perceptron tagger (Penn Treebank tagset).
    POS tags are needed for POS-aware lemmatization.

    Parameters
    ----------
    tokens:
        List of word tokens.

    Returns
    -------
    list[tuple[str, str]]
        List of (token, POS_tag) pairs.

    Examples
    --------
    >>> pos_tag(['great', 'battery', 'life'])
    [('great', 'JJ'), ('battery', 'NN'), ('life', 'NN')]
    """
    _ensure_nltk_resources()
    if not tokens:
        return []
    base_tokens = [t[:-4] if t.endswith("_NEG") else t for t in tokens]
    tagged = nltk.pos_tag(base_tokens)
    return [(tokens[i], tag) for i, (_, tag) in enumerate(tagged)]
