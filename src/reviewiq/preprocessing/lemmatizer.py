"""POS-aware lemmatization using NLTK's WordNetLemmatizer.

Why POS-aware lemmatization matters:
- "meeting" → "meet" (wrong if "meeting" is a noun: "the meeting was bad")
- "better" → "good" (adjective form) vs "better" → "well" (adverb form)
- "running" → "run" only if it's a verb; as an adjective it stays "running"

This module maps Penn Treebank POS tags to WordNet POS codes and uses
the correct code during lemmatization, so "meeting" as NN stays "meeting".

Usage
-----
    from reviewiq.preprocessing.lemmatizer import lemmatize_tokens
    tokens = [("great", "JJ"), ("battery", "NN"), ("life", "NN")]
    result = lemmatize_tokens(tokens)
    # → ["great", "battery", "life"]
"""

from __future__ import annotations

import nltk
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from reviewiq.preprocessing.tokenizer import _ensure_nltk_resources


# Penn Treebank tag prefix → WordNet POS mapping
_PTB_TO_WN: dict[str, str] = {
    "J": wordnet.ADJ,   # JJ, JJR, JJS → adjective
    "V": wordnet.VERB,  # VB, VBD, VBG, VBN, VBP, VBZ → verb
    "N": wordnet.NOUN,  # NN, NNS, NNP, NNPS → noun
    "R": wordnet.ADV,   # RB, RBR, RBS → adverb
}


def _ptb_to_wordnet(ptb_tag: str) -> str:
    """Convert a Penn Treebank POS tag to a WordNet POS tag.

    Parameters
    ----------
    ptb_tag:
        Penn Treebank tag (e.g., "NN", "VBZ", "JJ").

    Returns
    -------
    str
        WordNet POS constant. Defaults to NOUN for unknown tags.
    """
    return _PTB_TO_WN.get(ptb_tag[0] if ptb_tag else "N", wordnet.NOUN)


_lemmatizer: WordNetLemmatizer | None = None


def _get_lemmatizer() -> WordNetLemmatizer:
    """Return a singleton WordNetLemmatizer instance."""
    global _lemmatizer
    if _lemmatizer is None:
        _ensure_nltk_resources()
        _lemmatizer = WordNetLemmatizer()
    return _lemmatizer


def lemmatize_tokens(
    tagged_tokens: list[tuple[str, str]],
) -> list[str]:
    """Lemmatize tokens using their POS tags.

    Parameters
    ----------
    tagged_tokens:
        List of (token, penn_treebank_pos_tag) pairs, as returned by
        ``reviewiq.preprocessing.tokenizer.pos_tag``.

    Returns
    -------
    list[str]
        Lemmatized token strings.

    Examples
    --------
    >>> lemmatize_tokens([("batteries", "NNS"), ("running", "VBG"), ("better", "JJR")])
    ['battery', 'run', 'good']
    """
    lemmatizer = _get_lemmatizer()
    result = []
    for token, tag in tagged_tokens:
        has_neg = token.endswith("_NEG")
        base = token[:-4] if has_neg else token
        wn_pos = _ptb_to_wordnet(tag)
        lemma = lemmatizer.lemmatize(base, pos=wn_pos)
        result.append(f"{lemma}_NEG" if has_neg else lemma)
    return result


def lemmatize_word_list(tokens: list[str]) -> list[str]:
    """Lemmatize a plain list of tokens (POS tagging applied internally).

    Convenience wrapper that runs POS tagging then lemmatization in one call.
    Use this when you don't already have POS tags.

    Parameters
    ----------
    tokens:
        List of word strings (should be lowercased first).

    Returns
    -------
    list[str]
        Lemmatized tokens.
    """
    if not tokens:
        return []
    from reviewiq.preprocessing.tokenizer import pos_tag as _pos_tag  # avoid circular at module level

    tagged = _pos_tag(tokens)
    return lemmatize_tokens(tagged)
