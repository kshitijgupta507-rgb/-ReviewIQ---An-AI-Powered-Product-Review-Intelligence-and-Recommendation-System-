"""Text cleaning: HTML stripping, URL/email removal, unicode normalization.

This is the first step in the preprocessing pipeline. It operates on raw
review text and returns a string that is safe to pass to the normalizer.

Nothing language-specific happens here — just structural cleaning.
"""

from __future__ import annotations

import re
import unicodedata


# ─── Compiled regex patterns (compiled once at import time for speed) ─────────

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_URL_RE = re.compile(
    r"(?:https?://|www\.)\S+",
    re.IGNORECASE,
)
_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[a-z]{2,}", re.IGNORECASE)
_MULTI_SPACE_RE = re.compile(r"\s+")
_MULTI_PUNCT_RE = re.compile(r"([!?.]){3,}")   # "!!!!!!" → "!"


def strip_html(text: str) -> str:
    """Remove HTML tags from text.

    Parameters
    ----------
    text:
        Raw text that may contain HTML markup.

    Returns
    -------
    str
        Text with all HTML tags removed.

    Examples
    --------
    >>> strip_html("<b>Great</b> product!")
    'Great product!'
    """
    return _HTML_TAG_RE.sub(" ", text)


def strip_urls(text: str) -> str:
    """Replace URLs with a single space.

    Parameters
    ----------
    text:
        Text that may contain http/https/www URLs.
    """
    return _URL_RE.sub(" ", text)


def strip_emails(text: str) -> str:
    """Replace email addresses with a single space."""
    return _EMAIL_RE.sub(" ", text)


def normalize_unicode(text: str) -> str:
    """Normalize unicode to NFKC form and decode common HTML entities.

    NFKC converts compatibility characters (e.g., ﬁ → fi, ½ → 1/2)
    and standardises whitespace characters.

    Parameters
    ----------
    text:
        Raw unicode text.
    """
    # NFKC: Compatibility decomposition followed by canonical composition.
    text = unicodedata.normalize("NFKC", text)
    # Decode common HTML entities that survive HTML-tag stripping
    html_entities = {
        "&amp;": "&",
        "&lt;": "<",
        "&gt;": ">",
        "&quot;": '"',
        "&#39;": "'",
        "&nbsp;": " ",
        "&apos;": "'",
    }
    for entity, char in html_entities.items():
        text = text.replace(entity, char)
    return text


def collapse_repeated_punctuation(text: str) -> str:
    """Reduce sequences of 3+ identical punctuation marks to one.

    E.g. "GREAT!!!!!!" → "GREAT!"
    """
    return _MULTI_PUNCT_RE.sub(r"\1", text)


def collapse_whitespace(text: str) -> str:
    """Replace any run of whitespace (tabs, newlines, spaces) with a single space."""
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def clean_text(
    text: str,
    *,
    strip_html_tags: bool = True,
    strip_url: bool = True,
    strip_email: bool = True,
    unicode_normalize: bool = True,
    collapse_punct: bool = True,
) -> str:
    """Run the full cleaning sequence on a single review text string.

    Parameters
    ----------
    text:
        Raw review text.
    strip_html_tags:
        Remove HTML markup. Default True.
    strip_url:
        Remove URLs. Default True.
    strip_email:
        Remove email addresses. Default True.
    unicode_normalize:
        Apply NFKC unicode normalization. Default True.
    collapse_punct:
        Collapse runs of 3+ punctuation marks. Default True.

    Returns
    -------
    str
        Cleaned text, ready for normalization.

    Examples
    --------
    >>> clean_text("<p>Amazing laptop!!!!</p> Check https://example.com")
    'Amazing laptop! Check'
    """
    if not text or not isinstance(text, str):
        return ""

    if unicode_normalize:
        text = normalize_unicode(text)
    if strip_html_tags:
        text = strip_html(text)
    if strip_url:
        text = strip_urls(text)
    if strip_email:
        text = strip_emails(text)
    if collapse_punct:
        text = collapse_repeated_punctuation(text)

    return collapse_whitespace(text)
