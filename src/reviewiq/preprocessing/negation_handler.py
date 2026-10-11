"""Negation handling: scope-based NOT-suffixing.

Classical NLP trick: tokens after a negation word get a ``_NEG`` suffix
appended until the next clause boundary (punctuation or end of window).

This is essential for bag-of-words and TF-IDF classifiers because they
treat "good" and "not_NEG good_NEG" as completely different features —
which is exactly what we want for sentiment.

Example:
    "battery is not great but display is good"
    →  ["battery", "is", "not", "great_NEG", "but", "display", "is", "good"]

Transformer models (DistilBERT) don't need this — they understand context
natively. The _NEG suffix only goes into the token list used for
classical models (BoW, TF-IDF, Naive Bayes, SVM).

Reference: Das & Chen (2001); Pang et al. (2002).
"""

from __future__ import annotations

import re


# Negation trigger words and patterns
_NEGATION_WORDS: frozenset[str] = frozenset(
    [
        "not",
        "no",
        "never",
        "neither",
        "nor",
        "nothing",
        "nobody",
        "nowhere",
        "none",
        "cannot",
        "without",
        "hardly",
        "barely",
        "scarcely",
    ]
)

# Clause boundary markers — negation scope ends at these
_CLAUSE_BOUNDARIES: frozenset[str] = frozenset(
    [".", ",", "!", "?", ";", ":", "but", "however", "although", "though", "yet"]
)

_NEG_SUFFIX = "_NEG"


def handle_negation(
    tokens: list[str],
    window: int = 5,
    *,
    mark_negation_word: bool = False,
) -> list[str]:
    """Apply negation scoping to a token list.

    Tokens within ``window`` positions after a negation word (up to the
    next clause boundary) are suffixed with ``_NEG``.

    Parameters
    ----------
    tokens:
        Lowercased word tokens (punctuation included — needed for
        clause boundary detection).
    window:
        Maximum number of tokens to tag after a negation word.
        Typical range: 4–8. Set in ``configs/default.yaml``.
    mark_negation_word:
        If True, also suffix the negation word itself (e.g., "not_NEG").
        Default False — the negation word is kept as-is so it can still
        be identified as a negation trigger.

    Returns
    -------
    list[str]
        Tokens with ``_NEG`` appended where appropriate.

    Examples
    --------
    >>> handle_negation(["battery", "is", "not", "great", ",", "but", "display", "is", "good"])
    ['battery', 'is', 'not', 'great_NEG', ',', 'but', 'display', 'is', 'good']

    >>> handle_negation(["i", "would", "not", "buy", "this", "ever", "again"])
    ['i', 'would', 'not', 'buy_NEG', 'this_NEG', 'ever_NEG', 'again_NEG']
    """
    result: list[str] = []
    negation_countdown: int = 0   # how many more tokens to tag

    for token in tokens:
        tok_lower = token.lower()

        if tok_lower in _CLAUSE_BOUNDARIES:
            negation_countdown = 0
            result.append(token)
        elif tok_lower in _NEGATION_WORDS:
            negation_countdown = window
            if mark_negation_word:
                result.append(token + _NEG_SUFFIX)
            else:
                result.append(token)
        elif negation_countdown > 0:
            result.append(token + _NEG_SUFFIX)
            negation_countdown -= 1
        else:
            result.append(token)

    return result


def has_negation(tokens: list[str]) -> bool:
    """Return True if the token list contains any negation word.

    Used as a quick flag stored on ``ProcessedReview.has_negation``.
    """
    return any(t.lower() in _NEGATION_WORDS for t in tokens)
