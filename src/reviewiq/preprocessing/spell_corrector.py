"""Spell correction using SymSpell.

SymSpell uses a pre-computed delete-neighbourhood dictionary, giving
O(1) lookup for words within edit distance 2. For a 50,000-review
corpus this is fast enough to run without multiprocessing.

Design choices:
- Only correct tokens that are all-alphabetic and not already in the
  SymSpell dictionary. This avoids corrupting product model numbers
  (e.g., "i5-12500H", "RTX3050") and brand names.
- Max edit distance is 2 (config: preprocessing.symspell_max_edit_distance).
  Distance 2 catches common typos ("terribel", "bettary") without
  over-correcting intentional brand variations.
- Correction is applied per token, not per sentence, for speed.

Usage
-----
    from reviewiq.preprocessing.spell_corrector import SpellCorrector
    sc = SpellCorrector()
    sc.correct_tokens(["terribel", "batry", "laptop"])
    # → ["terrible", "battery", "laptop"]
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Tokens that look like product identifiers — skip spell correction
_TECH_PATTERN = re.compile(
    r"^[a-z0-9]*(i[3579]|ryzen|rtx|gtx|amd|intel|ddr[45]?|nvme|"
    r"ghz|mhz|usb|hdmi|ssd|hdd|ram|gpu|cpu|fps|hz|tb|gb|mb|kb|"
    r"wh|mah|ms|mm|inch|led|ips|oled)[a-z0-9\-]*$",
    re.IGNORECASE,
)


class SpellCorrector:
    """Lazy-initialised SymSpell-based spell corrector.

    The SymSpell dictionary is loaded on first use, not at import time,
    to keep startup fast when spell correction is disabled.

    Parameters
    ----------
    max_edit_distance:
        Maximum Damerau-Levenshtein distance for suggestions. Default 2.
    prefix_length:
        SymSpell prefix length (7 is the default; higher = more memory). Default 7.
    min_token_length:
        Tokens shorter than this are never corrected. Prevents "is" → "in", etc.
    """

    def __init__(
        self,
        max_edit_distance: int = 2,
        prefix_length: int = 7,
        min_token_length: int = 4,
    ) -> None:
        self._max_edit_distance = max_edit_distance
        self._prefix_length = prefix_length
        self._min_token_length = min_token_length
        self._sym_spell = None   # lazy init

    def _load(self) -> None:
        """Load SymSpell with its bundled frequency dictionary."""
        if self._sym_spell is not None:
            return

        try:
            from symspellpy import SymSpell, Verbosity  # noqa: F401

            sym_spell = SymSpell(
                max_dictionary_edit_distance=self._max_edit_distance,
                prefix_length=self._prefix_length,
            )
            # symspellpy ships its own frequency dictionary
            import importlib.resources as pkg_resources
            import symspellpy

            dict_path = Path(symspellpy.__file__).parent / "frequency_dictionary_en_82_765.txt"
            if not dict_path.exists():
                logger.warning(
                    "SymSpell frequency dictionary not found at %s. "
                    "Spell correction will be disabled.",
                    dict_path,
                )
                return

            sym_spell.load_dictionary(
                str(dict_path),
                term_index=0,
                count_index=1,
            )
            self._sym_spell = sym_spell
            logger.info("SymSpell dictionary loaded from %s.", dict_path)

        except ImportError:
            logger.warning(
                "symspellpy not installed. Spell correction disabled. "
                "Install with: pip install symspellpy"
            )

    def _should_correct(self, token: str) -> bool:
        """Return True if this token is a candidate for spell correction."""
        base = token[:-4] if token.endswith("_NEG") else token
        if len(base) < self._min_token_length:
            return False
        if not base.isalpha():
            return False
        if _TECH_PATTERN.match(base):
            return False
        return True

    def correct_token(self, token: str) -> str:
        """Correct a single token if it looks misspelled.

        Parameters
        ----------
        token:
            A single lowercased word token.

        Returns
        -------
        str
            Corrected token, or original if no suggestion found.
        """
        self._load()
        if self._sym_spell is None or not self._should_correct(token):
            return token

        from symspellpy import Verbosity

        has_neg = token.endswith("_NEG")
        base = token[:-4] if has_neg else token
        suggestions = self._sym_spell.lookup(
            base,
            Verbosity.CLOSEST,
            max_edit_distance=self._max_edit_distance,
            include_unknown=True,
        )
        if suggestions:
            res = suggestions[0].term
            return f"{res}_NEG" if has_neg else res
        return token

    def correct_tokens(self, tokens: list[str]) -> list[str]:
        """Correct a list of tokens.

        Parameters
        ----------
        tokens:
            Lowercased word tokens.

        Returns
        -------
        list[str]
            Tokens with spelling corrected where appropriate.

        Examples
        --------
        >>> sc = SpellCorrector()
        >>> sc.correct_tokens(["terribel", "bild", "qualety", "laptop"])
        ['terrible', 'build', 'quality', 'laptop']
        """
        self._load()
        if self._sym_spell is None:
            return tokens   # disabled gracefully

        return [self.correct_token(t) for t in tokens]
