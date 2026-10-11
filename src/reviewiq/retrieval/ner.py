"""Domain-specific Named Entity Recognition (NER) for laptop specifications.

Extracts hardware components, brands, and performance specs from review text:
- BRANDS: Apple, Dell, Lenovo, HP, ASUS, Acer, MSI, Razer, Microsoft, Samsung, etc.
- CPU: Intel Core (i3/i5/i7/i9, Core Ultra), AMD Ryzen, Apple Silicon (M1/M2/M3/M4), Snapdragon
- GPU: NVIDIA RTX (3050, 4060, 4070, 4080), Intel Iris Xe / Arc, AMD Radeon
- RAM: 8GB, 16GB, 32GB, 64GB DDR4/DDR5
- STORAGE: 256GB, 512GB, 1TB, 2TB NVMe SSD
- DISPLAY: Screen sizes (14", 15.6", 16"), resolution (FHD, 1440p, 4K, OLED), refresh rates (120Hz, 144Hz)
- OS: Windows 11, macOS, Linux, ChromeOS
- BATTERY: Watt-hours and battery duration mentions (70Wh, 12 hours)

Uses spaCy EntityRuler pattern matching and targeted regex for maximum precision.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

import spacy
from spacy.pipeline import EntityRuler

from reviewiq.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EntitySpan:
    """Detected entity with text span and label."""

    text: str
    label: str
    start_char: int
    end_char: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "label": self.label,
            "start_char": self.start_char,
            "end_char": self.end_char,
        }


class LaptopNER:
    """Domain-specific Named Entity Recognizer for laptop product specifications."""

    KNOWN_BRANDS = [
        "Apple", "Dell", "Lenovo", "HP", "ASUS", "Asus", "Acer", "MSI",
        "Razer", "Microsoft", "Samsung", "Alienware", "Gigabyte", "LG",
        "Framework", "Huawei", "Honor", "Toshiba", "Sony",
    ]

    def __init__(self) -> None:
        self.nlp = spacy.blank("en")
        self.ruler = self._build_entity_ruler()

    def _build_entity_ruler(self) -> EntityRuler:
        """Create and populate spaCy EntityRuler with curated laptop spec patterns."""
        ruler = self.nlp.add_pipe("entity_ruler")

        patterns = []

        # 1. Brands
        for brand in self.KNOWN_BRANDS:
            patterns.append({
                "label": "BRAND",
                "pattern": [{"LOWER": brand.lower()}],
            })

        # 2. Operating Systems
        os_list = [
            ("Windows", "11"), ("Windows", "10"), ("macOS",), ("ChromeOS",),
            ("Linux",), ("Ubuntu",),
        ]
        for item in os_list:
            patterns.append({
                "label": "OS",
                "pattern": [{"LOWER": term.lower()} for term in item],
            })

        # 3. CPU Families & Models
        # Intel Core i3/i5/i7/i9
        for i_num in ["i3", "i5", "i7", "i9"]:
            patterns.append({
                "label": "CPU",
                "pattern": [
                    {"LOWER": {"IN": ["intel", "core"]}, "OP": "?"},
                    {"LOWER": "core", "OP": "?"},
                    {"LOWER": i_num},
                ],
            })
            # Hyphenated e.g. "i7-13700H"
            patterns.append({
                "label": "CPU",
                "pattern": [{"TEXT": {"REGEX": rf"(?i)^{i_num}-?\d{{4,5}}[a-z]*$"}}],
            })

        # Intel Core Ultra
        patterns.append({
            "label": "CPU",
            "pattern": [
                {"LOWER": {"IN": ["intel", "core"]}, "OP": "?"},
                {"LOWER": "core"},
                {"LOWER": "ultra"},
                {"LOWER": {"IN": ["5", "7", "9"]}, "OP": "?"},
            ],
        })

        # AMD Ryzen
        for ryzen_num in ["3", "5", "7", "9"]:
            patterns.append({
                "label": "CPU",
                "pattern": [
                    {"LOWER": "amd", "OP": "?"},
                    {"LOWER": "ryzen"},
                    {"LOWER": ryzen_num},
                ],
            })
            # e.g. "Ryzen 7 7840HS" or "Ryzen-7840HS"
            patterns.append({
                "label": "CPU",
                "pattern": [
                    {"LOWER": "amd", "OP": "?"},
                    {"LOWER": "ryzen"},
                    {"TEXT": {"REGEX": r"^\d{4,5}[a-zA-Z]*$"}},
                ],
            })

        # Apple Silicon
        for m_chip in ["m1", "m2", "m3", "m4"]:
            patterns.append({
                "label": "CPU",
                "pattern": [
                    {"LOWER": m_chip},
                    {"LOWER": {"IN": ["pro", "max", "ultra"]}, "OP": "?"},
                ],
            })

        # Qualcomm Snapdragon
        patterns.append({
            "label": "CPU",
            "pattern": [
                {"LOWER": "snapdragon"},
                {"LOWER": "x", "OP": "?"},
                {"LOWER": "elite", "OP": "?"},
            ],
        })

        # 4. GPU / Graphics
        # NVIDIA RTX / GTX
        patterns.append({
            "label": "GPU",
            "pattern": [
                {"LOWER": {"IN": ["nvidia", "geforce"]}, "OP": "?"},
                {"LOWER": {"IN": ["rtx", "gtx"]}},
                {"TEXT": {"REGEX": r"^\d{4}(?:ti)?$"}},
            ],
        })
        patterns.append({
            "label": "GPU",
            "pattern": [{"TEXT": {"REGEX": r"(?i)^(?:rtx|gtx)\d{4}(?:ti)?$"}}],
        })
        # Intel Arc / Iris Xe
        patterns.append({
            "label": "GPU",
            "pattern": [{"LOWER": "iris"}, {"LOWER": "xe"}],
        })
        patterns.append({
            "label": "GPU",
            "pattern": [{"LOWER": "intel"}, {"LOWER": "arc"}],
        })
        # AMD Radeon
        patterns.append({
            "label": "GPU",
            "pattern": [
                {"LOWER": "radeon"},
                {"TEXT": {"REGEX": r"(?i)^(?:\d{3,4}[a-z]*|graphics)$"}, "OP": "?"},
            ],
        })

        # 5. RAM
        for ram_size in ["4gb", "8gb", "12gb", "16gb", "24gb", "32gb", "64gb"]:
            patterns.append({
                "label": "RAM",
                "pattern": [{"TEXT": {"REGEX": rf"(?i)^{ram_size}$"}}],
            })
            patterns.append({
                "label": "RAM",
                "pattern": [
                    {"TEXT": {"REGEX": rf"(?i)^{ram_size[:-2]}$"}},
                    {"LOWER": "gb"},
                    {"LOWER": {"IN": ["ram", "ddr4", "ddr5", "lpddr5", "memory"]}, "OP": "?"},
                ],
            })

        # 6. STORAGE
        for size in ["128gb", "256gb", "512gb", "1tb", "2tb", "4tb"]:
            patterns.append({
                "label": "STORAGE",
                "pattern": [
                    {"TEXT": {"REGEX": rf"(?i)^{size}$"}}],
            })
            patterns.append({
                "label": "STORAGE",
                "pattern": [
                    {"TEXT": {"REGEX": rf"(?i)^{size[:-2]}$"}},
                    {"LOWER": {"IN": ["gb", "tb"]}},
                    {"LOWER": {"IN": ["ssd", "nvme", "storage", "drive", "hdd"]}},
                ],
            })

        # 7. DISPLAY Specs
        for res in ["1080p", "1440p", "4k", "fhd", "qhd", "uhd", "oled", "ips"]:
            patterns.append({
                "label": "DISPLAY",
                "pattern": [{"LOWER": res}],
            })
        for hz in ["60hz", "90hz", "120hz", "144hz", "165hz", "240hz"]:
            patterns.append({
                "label": "DISPLAY",
                "pattern": [{"LOWER": hz}],
            })
        # Screen size e.g. 14-inch, 15.6 inch, 16" - required unit
        size_regex = r"(?i)^(?:13\.3|14|15\.6|16|17\.3)$"
        unit_regex = r"(?i)^(?:\"|inch|inches)$"
        patterns.append({
            "label": "DISPLAY",
            "pattern": [{"TEXT": {"REGEX": r"(?i)^(?:13\.3|14|15\.6|16|17\.3)(?:\"|-inch|-inches|inch|inches)$"}}],
        })
        patterns.append({
            "label": "DISPLAY",
            "pattern": [
                {"TEXT": {"REGEX": size_regex}},
                {"TEXT": {"REGEX": unit_regex}},
            ],
        })
        patterns.append({
            "label": "DISPLAY",
            "pattern": [
                {"TEXT": {"REGEX": size_regex}},
                {"TEXT": "-"},
                {"TEXT": {"REGEX": unit_regex}},
            ],
        })

        # 8. BATTERY
        patterns.append({
            "label": "BATTERY",
            "pattern": [{"TEXT": {"REGEX": r"(?i)^\d{2,3}wh$"}}],
        })

        ruler.add_patterns(patterns)
        return ruler

    def extract_entities(self, text: str) -> list[EntitySpan]:
        """Extract structured entity spans from text.

        Parameters
        ----------
        text:
            Input review or query string.

        Returns
        -------
        List of EntitySpan objects with text, label, start_char, end_char.
        """
        if not text:
            return []

        doc = self.nlp(text)
        entities: list[EntitySpan] = []

        # Deduplicate overlapping spans preferring longer matches
        seen_spans: set[tuple[int, int]] = set()
        for ent in doc.ents:
            span_tuple = (ent.start_char, ent.end_char)
            if span_tuple not in seen_spans:
                seen_spans.add(span_tuple)
                entities.append(
                    EntitySpan(
                        text=ent.text,
                        label=ent.label_,
                        start_char=ent.start_char,
                        end_char=ent.end_char,
                    )
                )

        # Additional regex sweep for standalone spec patterns (e.g. 16GB, RTX 4060)
        regex_patterns = [
            (r"\b(\d{1,2}\s*(?:hours|hrs))\s+(?:of\s+)?battery\b", "BATTERY"),
            (r"\b(1920x1080|2560x1440|3840x2160)\b", "DISPLAY"),
        ]

        for reg, label in regex_patterns:
            for match in re.finditer(reg, text, flags=re.IGNORECASE):
                span = (match.start(1), match.end(1))
                # Reject any span that intersects existing spans (containment or partial overlap)
                if not any(max(s[0], span[0]) < min(s[1], span[1]) for s in seen_spans):
                    seen_spans.add(span)
                    entities.append(
                        EntitySpan(
                            text=match.group(1),
                            label=label,
                            start_char=match.start(1),
                            end_char=match.end(1),
                        )
                    )

        # Sort by start_char
        entities.sort(key=lambda e: e.start_char)
        return entities

    def extract_entities_grouped(self, text: str) -> dict[str, list[str]]:
        """Extract and group entities by label category.

        Returns
        -------
        Dictionary mapping label -> list of extracted entity strings.
        """
        spans = self.extract_entities(text)
        grouped: dict[str, list[str]] = {}
        for span in spans:
            if span.label not in grouped:
                grouped[span.label] = []
            if span.text not in grouped[span.label]:
                grouped[span.label].append(span.text)
        return grouped

    def tag_text(self, text: str) -> str:
        """Annotate text with entity brackets for display: e.g. [Apple](BRAND)."""
        spans = self.extract_entities(text)
        if not spans:
            return text

        tagged = ""
        last_idx = 0
        for span in spans:
            tagged += text[last_idx:span.start_char]
            tagged += f"[{span.text}]({span.label})"
            last_idx = span.end_char
        tagged += text[last_idx:]
        return tagged


if __name__ == "__main__":
    ner = LaptopNER()
    sample = (
        "I bought this Apple MacBook with the M3 Pro chip, 16GB RAM and 512GB SSD. "
        "The 14-inch 120Hz OLED screen is stunning, much better than my older Dell with Intel i7 and RTX 3050. "
        "Battery lasts 12 hours under normal use on macOS."
    )

    print("--- Extracted Laptop Entities ---")
    for ent in ner.extract_entities(sample):
        print(f"[{ent.label}] '{ent.text}' (chars {ent.start_char}:{ent.end_char})")

    print("\n--- Grouped Specifications ---")
    grouped = ner.extract_entities_grouped(sample)
    for cat, items in grouped.items():
        print(f"{cat}: {', '.join(items)}")

    print("\n--- Tagged Review ---")
    print(ner.tag_text(sample))
