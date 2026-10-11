"""Emoji handling: convert emoji characters to text descriptions.

Uses the ``emoji`` library which maps each emoji to its official CLDR
short name (e.g., 😍 → "smiling face with heart-eyes").

The text descriptions are meaningful for sentiment analysis — 😍 carries
positive sentiment, 😡 carries negative, etc.

Usage
-----
    from reviewiq.preprocessing.emoji_handler import emojis_to_text
    result = emojis_to_text("I luv this laptop 😍 battery is bad 😤")
    # → "I luv this laptop smiling face with heart-eyes battery is bad face with steam from nose"
"""

from __future__ import annotations

import re

try:
    import emoji as emoji_lib

    _EMOJI_AVAILABLE = True
except ImportError:
    _EMOJI_AVAILABLE = False


def emojis_to_text(text: str, separator: str = " ") -> str:
    """Replace emoji characters in text with their CLDR text descriptions.

    Parameters
    ----------
    text:
        Input string, possibly containing emoji.
    separator:
        String inserted before and after each converted emoji description.
        Default is a space.

    Returns
    -------
    str
        Text with emoji replaced by descriptive words.

    Examples
    --------
    >>> emojis_to_text("great battery 🔋 but heavy 😞")
    'great battery battery but heavy disappointed face'
    """
    if not text or not isinstance(text, str):
        return text or ""

    if not _EMOJI_AVAILABLE:
        # Graceful degradation: strip emoji instead of converting
        return _strip_emoji_fallback(text)

    # emoji.demojize wraps each emoji in colons: 😍 → :smiling_face_with_heart-eyes:
    demojized = emoji_lib.demojize(text, delimiters=(":", ":"))

    # Replace :colon_wrapped_names: with space-separated words
    def _clean_name(match: re.Match) -> str:
        name = match.group(1).replace("_", " ").replace("-", " ")
        return f"{separator}{name}{separator}"

    result = re.sub(r":([a-z0-9_\-]+):", _clean_name, demojized)
    # Collapse multiple spaces introduced by replacements
    return re.sub(r"\s+", " ", result).strip()


def _strip_emoji_fallback(text: str) -> str:
    """Remove emoji characters when the emoji library is not installed.

    Covers the BMP emoji ranges and common supplementary plane ranges.
    """
    # Basic emoji Unicode ranges
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
        "\U0001F680-\U0001F6FF"  # Transport and Map
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2"              # Enclosed Latin letter M
        "\U0001F200-\U0001F251"  # Enclosed Ideographic Supplement
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(" ", text)
