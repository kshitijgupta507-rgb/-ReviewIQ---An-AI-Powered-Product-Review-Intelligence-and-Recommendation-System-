"""Aspect extraction module for ReviewIQ.

Extracts aspect terms and opinion modifiers from laptop customer reviews:
- Matches against the 12 canonical laptop aspects:
  battery, display, performance, build_quality, keyboard, price, thermal,
  storage, weight, software, customer_service, delivery
- Uses domain-curated keyword lexicons and syntactic dependency patterns
- Identifies opinion words (adjectives, adverbs) and negation scopes
- Returns structured ExtractedAspect objects with sentence contexts
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

from reviewiq.utils.config import load_config
from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ExtractedAspect:
    """Structured representation of an extracted review aspect."""

    aspect: str
    term: str
    opinion_words: list[str]
    sentence: str
    is_negated: bool = False
    start_char: int = -1
    end_char: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "aspect": self.aspect,
            "term": self.term,
            "opinion_words": self.opinion_words,
            "sentence": self.sentence,
            "is_negated": self.is_negated,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


class AspectExtractor:
    """Extracts laptop aspects and opinion modifiers from customer review text.

    Parameters
    ----------
    aspects:
        List of target aspects. Default from config (12 canonical laptop aspects).
    """

    # Comprehensive domain seed lexicon mapping keyword stems to canonical aspects
    ASPECT_LEXICON: dict[str, list[str]] = {
        "battery": [
            "battery", "battery life", "batteries", "charge", "charger", "charging",
            "runtime", "standby", "backup", "mah", "watt-hour", "wh", "power bank",
        ],
        "display": [
            "screen", "display", "monitor", "oled", "ips", "resolution", "brightness",
            "contrast", "hdr", "viewing angle", "refresh rate", "panel", "hz", "1080p",
            "1440p", "4k", "fhd", "qhd", "colors", "pixel", "glare", "anti-glare",
        ],
        "performance": [
            "performance", "speed", "fast", "slow", "sluggish", "lag", "laggy",
            "multitask", "multitasking", "processor", "cpu", "gpu", "ram", "memory",
            "fps", "gaming", "smooth", "snappy", "boot", "loading", "specs",
        ],
        "build_quality": [
            "build", "build quality", "chassis", "hinge", "hinges", "durable", "durability",
            "sturdy", "flimsy", "plastic", "aluminum", "metal", "body", "casing", "creak",
            "finish", "construction", "feel", "solid",
        ],
        "keyboard": [
            "keyboard", "keys", "key", "typing", "type", "trackpad", "touchpad",
            "click", "travel", "key travel", "backlight", "backlit", "spacebar",
            "layout", "numpad",
        ],
        "price": [
            "price", "cost", "expensive", "cheap", "value", "worth", "deal",
            "budget", "affordable", "dollar", "dollars", "pricing", "overpriced",
            "bang for buck", "money",
        ],
        "thermal": [
            "thermal", "thermals", "heat", "hot", "warm", "fan", "fans", "noise",
            "noisy", "loud", "throttling", "throttle", "cooling", "cool", "vent",
            "vents", "temperature", "overheating",
        ],
        "storage": [
            "storage", "ssd", "hdd", "hard drive", "nvme", "capacity", "space",
            "disk", "gigabytes", "terabytes", "512gb", "1tb", "256gb",
        ],
        "weight": [
            "weight", "light", "lightweight", "heavy", "portable", "portability",
            "compact", "bulky", "heaviness", "thin", "slim", "carry",
        ],
        "software": [
            "software", "windows", "macos", "os", "bloatware", "driver", "drivers",
            "update", "updates", "app", "apps", "bios", "bug", "bugs", "crash",
            "crashes", "system",
        ],
        "customer_service": [
            "customer service", "support", "warranty", "customer support", "tech support",
            "service", "repair", "replacement", "rma", "refund", "agent", "representative",
        ],
        "delivery": [
            "delivery", "shipping", "shipped", "package", "packaging", "arrived",
            "box", "courier", "carrier", "delivered", "transit", "damage in transit",
        ],
    }

    # Common opinion adjectives and descriptive verbs
    OPINION_WORDS: set[str] = {
        "amazing", "awesome", "excellent", "great", "good", "decent", "fine",
        "satisfactory", "impressive", "outstanding", "superb", "terrific", "fantastic",
        "incredible", "exceptional", "wonderful", "brilliant", "stellar", "lasting",
        "crisp", "vibrant", "sharp", "sturdy", "solid", "durable", "fast", "quick",
        "speedy", "snappy", "smooth", "quiet", "cool", "light", "portable", "bright",
        "affordable", "cheap", "bad", "poor", "terrible", "horrible", "awful", "dreadful",
        "disappointing", "frustrating", "slow", "sluggish", "laggy", "dim", "washed",
        "flimsy", "fragile", "loose", "wobbly", "hot", "loud", "noisy", "heavy", "bulky",
        "expensive", "overpriced", "useless", "broken", "defective", "mushy", "clicky",
        "responsive", "unresponsive", "weak", "powerful", "long", "short",
    }

    NEGATION_TERMS: set[str] = {
        "not", "never", "no", "hardly", "barely", "scarcely", "without", "rarely", "n't",
    }

    def __init__(self, aspects: Sequence[str] | None = None) -> None:
        cfg = load_config()
        a_cfg = getattr(cfg, "absa", None)

        self.aspects = (
            list(aspects)
            if aspects is not None
            else list(getattr(a_cfg, "aspects", list(self.ASPECT_LEXICON.keys())))
        )

        # Build optimized regex dictionary per aspect
        self._compiled_patterns: dict[str, list[re.Pattern]] = {}
        for aspect in self.aspects:
            terms = self.ASPECT_LEXICON.get(aspect, [aspect])
            # Sort terms by length descending to match multi-word phrases first
            sorted_terms = sorted(terms, key=len, reverse=True)
            self._compiled_patterns[aspect] = [
                re.compile(rf"\b{re.escape(term)}\b", re.IGNORECASE) for term in sorted_terms
            ]

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences while preserving meaningful review clauses."""
        return [s.strip() for s in re.split(r"[.!?;\n]+", text) if s.strip()]

    def _extract_opinions_and_negation(
        self, sentence: str, match_start: int, match_end: int
    ) -> tuple[list[str], bool]:
        """Extract nearby opinion modifiers and check for negation within the aspect's clause."""
        # Find clause boundaries enclosing [match_start, match_end]
        clause_start = 0
        clause_end = len(sentence)

        for m in re.finditer(
            r"(?:[,\s]+(?:but|however|although|though|whereas|while|yet)\b|[;])",
            sentence,
            flags=re.IGNORECASE,
        ):
            if m.end() <= match_start:
                clause_start = max(clause_start, m.end())
            elif m.start() >= match_end:
                clause_end = min(clause_end, m.start())
                break

        relevant_clause = sentence[clause_start:clause_end]
        clause_words = re.findall(r"\b[\w'-]+\b", relevant_clause.lower())

        # Match whole tokens and contractions ending in "n't"
        is_negated = any(
            w in self.NEGATION_TERMS or w.endswith("n't") or w.endswith("n’t")
            for w in clause_words
        )

        # Identify opinion words present in the relevant clause
        found_opinions: list[str] = []
        for word in clause_words:
            clean_w = word.strip("'-’")
            if clean_w in self.OPINION_WORDS:
                found_opinions.append(clean_w)

        return found_opinions, is_negated

    def extract_from_sentence(self, sentence: str) -> list[ExtractedAspect]:
        """Extract all aspect mentions and associated modifiers from a single sentence."""
        extracted: list[ExtractedAspect] = []
        seen_aspects: set[str] = set()

        for aspect in self.aspects:
            patterns = self._compiled_patterns.get(aspect, [])
            for pat in patterns:
                match = pat.search(sentence)
                if match and aspect not in seen_aspects:
                    term = match.group(0)
                    opinions, is_neg = self._extract_opinions_and_negation(
                        sentence, match.start(), match.end()
                    )

                    extracted.append(
                        ExtractedAspect(
                            aspect=aspect,
                            term=term,
                            opinion_words=opinions,
                            sentence=sentence,
                            is_negated=is_neg,
                            start_char=match.start(),
                            end_char=match.end(),
                        )
                    )
                    seen_aspects.add(aspect)
                    break

        return extracted

    def extract_from_text(self, text: str) -> list[ExtractedAspect]:
        """Extract aspects across all sentences in a complete review text."""
        if not text or not text.strip():
            return []

        sentences = self._split_sentences(text)
        all_aspects: list[ExtractedAspect] = []

        for sent in sentences:
            all_aspects.extend(self.extract_from_sentence(sent))

        return all_aspects

    def extract(self, text: str) -> list[ExtractedAspect]:
        """Extract aspects across all sentences in a review text (alias for extract_from_text)."""
        return self.extract_from_text(text)

    def extract_batch(self, texts: Sequence[str]) -> list[list[ExtractedAspect]]:
        """Extract aspects for a batch/sequence of review texts."""
        return [self.extract_from_text(t) for t in texts]

    def extract_grouped(self, text: str) -> dict[str, list[ExtractedAspect]]:
        """Group extracted aspects by aspect name."""
        aspects = self.extract_from_text(text)
        grouped: dict[str, list[ExtractedAspect]] = {a: [] for a in self.aspects}
        for item in aspects:
            if item.aspect in grouped:
                grouped[item.aspect].append(item)
        return {k: v for k, v in grouped.items() if v}


if __name__ == "__main__":
    extractor = AspectExtractor()
    sample_review = (
        "The battery life is amazing and lasts 14 hours. "
        "However, the OLED screen is a bit dim under direct sunlight. "
        "Fast performance with no lag, but the fans get super loud and hot under load. "
        "Build quality is solid with sturdy aluminum hinges. Excellent keyboard for typing!"
    )

    print("--- Extracted Aspects ---")
    for item in extractor.extract_from_text(sample_review):
        neg_str = " (NEGATED)" if item.is_negated else ""
        print(f"[{item.aspect.upper()}]{neg_str} Term: '{item.term}' | Opinions: {item.opinion_words}")
        print(f"  Sentence: \"{item.sentence}\"")
