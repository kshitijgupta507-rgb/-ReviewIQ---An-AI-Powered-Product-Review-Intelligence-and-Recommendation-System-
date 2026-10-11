"""Text normalization: contractions, slang, and case folding.

Normalization happens after cleaning but before tokenization.
It operates at the string level (not token level) so multi-word
contractions and slang phrases are handled correctly.

Order in pipeline: clean → emoji → normalize → tokenize
"""

from __future__ import annotations

import re


# ─── Contractions dictionary ──────────────────────────────────────────────────
# Sorted by length (longest first) so "won't've" matches before "won't".
_CONTRACTIONS: dict[str, str] = {
    # Negatives
    "ain't": "are not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "daren't": "dare not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "isn't": "is not",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "wasn't": "was not",
    "weren't": "were not",
    "won't": "will not",
    "won't've": "will not have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    # Auxiliaries
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "o'clock": "of the clock",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she's": "she is",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there would",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "where'd": "where did",
    "where's": "where is",
    "who'd": "who would",
    "who'll": "who will",
    "who're": "who are",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
}

# Compiled pattern: word boundaries, case-insensitive
_CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_CONTRACTIONS, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


# ─── Laptop-review slang dictionary ──────────────────────────────────────────
# Common informal abbreviations and slang found in tech reviews.
_SLANG: dict[str, str] = {
    # Sentiment
    "luv": "love",
    "luvd": "loved",
    "gr8": "great",
    "gr8t": "great",
    "amazn": "amazing",
    "awsm": "awesome",
    "fab": "fabulous",
    "brill": "brilliant",
    "trrble": "terrible",
    "horbl": "horrible",
    "wtf": "very bad",
    "omg": "oh my god",
    "smh": "shaking my head",
    "tbh": "to be honest",
    "ngl": "not going to lie",
    "imo": "in my opinion",
    "imho": "in my honest opinion",
    # Tech abbreviations
    "batt": "battery",
    "batry": "battery",
    "kb": "keyboard",
    "kbd": "keyboard",
    "disp": "display",
    "scrn": "screen",
    "spd": "speed",
    "perf": "performance",
    "prfrm": "performance",
    "qual": "quality",
    "ssd": "ssd",
    "hdd": "hard drive",
    "ram": "ram",
    "cpu": "processor",
    "gpu": "graphics card",
    "wifi": "wifi",
    "bt": "bluetooth",
    "usb": "usb",
    "hdmi": "hdmi",
    # Informal abbreviations
    "pls": "please",
    "plz": "please",
    "thx": "thanks",
    "ty": "thank you",
    "rly": "really",
    "rlly": "really",
    "srsly": "seriously",
    "def": "definitely",
    "defntly": "definitely",
    "prob": "probably",
    "probs": "probably",
    "tho": "though",
    "thru": "through",
    "b4": "before",
    "2day": "today",
    "4ever": "forever",
    "w/": "with",
    "w/o": "without",
    "yr": "year",
    "mo": "month",
    "hrs": "hours",
    "min": "minutes",
}

# Compiled pattern for slang (whole words only)
_SLANG_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(_SLANG, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def expand_contractions(text: str) -> str:
    """Expand English contractions in text.

    Parameters
    ----------
    text:
        Input text, possibly containing contractions like "don't", "it's".

    Returns
    -------
    str
        Text with contractions expanded to their full forms.

    Examples
    --------
    >>> expand_contractions("I wouldn't buy this again. It's terrible.")
    "I would not buy this again. It is terrible."
    """
    def _replace(match: re.Match) -> str:
        original = match.group(0)
        key = original.lower()
        expanded = _CONTRACTIONS.get(key, original)
        # Preserve leading capitalisation
        if original[0].isupper():
            return expanded.capitalize()
        return expanded

    return _CONTRACTION_RE.sub(_replace, text)


def replace_slang(text: str) -> str:
    """Replace common internet/tech slang with standard English equivalents.

    Parameters
    ----------
    text:
        Input text possibly containing slang terms.

    Returns
    -------
    str
        Text with known slang replaced.

    Examples
    --------
    >>> replace_slang("gr8 batt life, luv the disp")
    'great battery life, love the display'
    """
    def _replace(match: re.Match) -> str:
        original = match.group(0)
        key = original.lower()
        replacement = _SLANG.get(key, original)
        if original[0].isupper():
            return replacement.capitalize()
        return replacement

    return _SLANG_RE.sub(_replace, text)


def to_lowercase(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def normalize_text(
    text: str,
    *,
    expand_contraction: bool = True,
    replace_slang_terms: bool = True,
    lowercase: bool = True,
) -> str:
    """Apply all normalization steps in the correct order.

    Parameters
    ----------
    text:
        Cleaned text (HTML/URL already stripped).
    expand_contraction:
        Expand "don't" → "do not", etc. Default True.
    replace_slang_terms:
        Replace informal abbreviations. Default True.
    lowercase:
        Convert to lowercase. Default True.

    Returns
    -------
    str
        Normalized text, ready for tokenization.
    """
    if not text:
        return ""

    if expand_contraction:
        text = expand_contractions(text)
    if replace_slang_terms:
        text = replace_slang(text)
    if lowercase:
        text = to_lowercase(text)

    return text
