"""Matching the names sources use for vehicles to our internal vehicle IDs.

Registry labels ("I/BYD DOLPHIN MINI GS5EV", "HYUNDAI/CRETA1TA PLTINUM") and FIPE model
names ("Polo Track 1.0 Flex 12V 5p") are matched with per-family regular expressions:
the longest match wins, and a tie is sent to review instead of guessed. Free text from
users ("quero um hrv 2023") is matched the same way, with a fuzzy fallback.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, Field
from rapidfuzz import fuzz, process

IMPORT_PREFIXES = ("I/", "IMP/")
#: Marketing words some labels put before the model ("NOVO GOL", "NOVA SAVEIRO").
MODEL_PREFIXES = ("NOVO ", "NOVA ", "NEW ")
AUTO_ACCEPT = 0.9


class BrandSpec(BaseModel):
    id: str
    name: str
    labels: list[str]


class FamilySpec(BaseModel):
    id: str
    brand: str
    name: str
    segment: str | None = None
    body: str | None = None
    status: str = "unverified"
    powertrains: list[str] = Field(default_factory=list)
    patterns: list[str]
    #: Similar-looking labels that are NOT this family (e.g. 'ONIX PLUS' for Onix).
    lookalikes: list[str] = Field(default_factory=list)
    notes: str | None = None


@dataclass(frozen=True)
class Match:
    family_id: str | None
    brand_id: str | None
    confidence: float
    method: str
    matched_text: str = ""
    candidates: tuple[str, ...] = field(default_factory=tuple)

    @property
    def accepted(self) -> bool:
        return self.family_id is not None and self.confidence >= AUTO_ACCEPT


def normalize(text: str) -> str:
    """Upper case, no accents, single spaces."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", stripped.upper()).strip()


@dataclass(frozen=True)
class _CompiledFamily:
    spec: FamilySpec
    anchored: tuple[re.Pattern[str], ...]
    free: tuple[re.Pattern[str], ...]
    lookalikes: tuple[re.Pattern[str], ...]


class Resolver:
    def __init__(self, brands: list[BrandSpec], families: list[FamilySpec]) -> None:
        self.brands = {b.id: b for b in brands}
        self.families = {f.id: f for f in families}
        self._brand_labels: dict[str, str] = {}
        for brand in brands:
            for label in [brand.name, *brand.labels]:
                self._brand_labels[normalize(label)] = brand.id
        self._by_brand: dict[str, list[_CompiledFamily]] = {}
        for fam in families:
            if fam.brand not in self.brands:
                raise ValueError(f"Family {fam.id} refers to unknown brand {fam.brand}")
            compiled = _CompiledFamily(
                fam,
                tuple(re.compile(rf"^(?:{p})") for p in fam.patterns),
                tuple(re.compile(rf"(?<![A-Z0-9])(?:{p})") for p in fam.patterns),
                tuple(re.compile(rf"^(?:{p})") for p in fam.lookalikes),
            )
            self._by_brand.setdefault(fam.brand, []).append(compiled)
        self._fuzzy_choices = {
            fam.id: normalize(f"{self.brands[fam.brand].name} {fam.name}") for fam in families
        }

    @classmethod
    def from_yaml(cls, path: Path) -> Resolver:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        brands = [BrandSpec(id=k, **v) for k, v in raw["brands"].items()]
        families = [FamilySpec(**f) for f in raw["families"]]
        return cls(brands, families)

    # --- brands -----------------------------------------------------------------

    def brand_for(self, label: str) -> str | None:
        norm = normalize(label)
        for candidate in (norm, norm.split(" - ")[0], norm.split(" ")[0]):
            if candidate in self._brand_labels:
                return self._brand_labels[candidate]
        return None

    # --- structured labels ------------------------------------------------------

    @staticmethod
    def split_registry_label(label: str) -> tuple[str, str]:
        """'VW/POLO TRACK MA' -> ('VW', 'POLO TRACK MA'); 'I/BYD DOLPHIN' -> ('BYD', 'DOLPHIN')."""
        norm = normalize(label)
        for prefix in IMPORT_PREFIXES:
            if norm.startswith(prefix):
                rest = norm[len(prefix) :]
                brand, _, model = rest.partition(" ")
                return brand, model.strip()
        brand, _, model = norm.partition("/")
        return brand.strip(), model.strip()

    def resolve_registry(self, label: str) -> Match:
        """Match a SENATRAN/RENAVAM 'Marca Modelo' label."""
        brand_label, model = self.split_registry_label(label)
        brand_id = self.brand_for(brand_label)
        if brand_id is None:
            return Match(None, None, 0.0, "unknown_brand")
        return self._match_model(brand_id, model, method="registry_pattern")

    def resolve_fipe(self, brand: str, model: str) -> Match:
        """Match a FIPE brand + model name, e.g. ('VW - VolksWagen', 'Polo Track 1.0 ...')."""
        brand_id = self.brand_for(brand)
        if brand_id is None:
            return Match(None, None, 0.0, "unknown_brand")
        return self._match_model(brand_id, normalize(model), method="fipe_pattern")

    def _match_model(self, brand_id: str, model: str, *, method: str) -> Match:
        model = strip_model_prefix(model)
        hits: list[tuple[int, str]] = []
        for fam in self._by_brand.get(brand_id, []):
            best = max((len(m.group(0)) for p in fam.anchored if (m := p.match(model))), default=0)
            if best:
                hits.append((best, fam.spec.id))
        if not hits:
            return Match(None, brand_id, 0.0, "no_pattern")
        hits.sort(reverse=True)
        top_len = hits[0][0]
        tied = tuple(fid for length, fid in hits if length == top_len)
        if len(tied) > 1:
            return Match(None, brand_id, 0.5, "ambiguous", model, tied)
        return Match(hits[0][1], brand_id, 0.97, method, model[:top_len])

    # --- free text ----------------------------------------------------------------

    def find(self, text: str, limit: int = 5) -> list[Match]:
        """Families mentioned in free text, best first. Patterns first, then fuzzy."""
        norm = normalize(text)
        mentioned_brands = {
            bid for label, bid in self._brand_labels.items() if _has_word(norm, label)
        }
        pattern_hits: dict[str, Match] = {}
        for brand_id, fams in self._by_brand.items():
            for fam in fams:
                for pattern in fam.free:
                    m = pattern.search(norm)
                    if not m:
                        continue
                    # A brand named elsewhere in the text makes the match more certain.
                    conf = 0.97 if brand_id in mentioned_brands else 0.92
                    score = conf + len(m.group(0)) / 1000
                    prev = pattern_hits.get(fam.spec.id)
                    if prev is None or score > prev.confidence:
                        pattern_hits[fam.spec.id] = Match(
                            fam.spec.id, brand_id, score, "text_pattern", m.group(0)
                        )
        results = _drop_contained(sorted(pattern_hits.values(), key=lambda m: -m.confidence))
        if len(results) < limit:
            fuzzy = process.extract(
                norm, self._fuzzy_choices, scorer=fuzz.WRatio, limit=limit, score_cutoff=80
            )
            for _choice, score, family_id in fuzzy:
                if family_id in pattern_hits:
                    continue
                brand_id = self.families[family_id].brand
                results.append(Match(family_id, brand_id, round(score / 100 * 0.85, 3), "fuzzy"))
        return [
            Match(m.family_id, m.brand_id, min(m.confidence, 0.99), m.method, m.matched_text)
            for m in results[:limit]
        ]

    def suggest_for_label(self, label: str) -> tuple[str, float] | None:
        """A tracked family this unmatched label may belong to, for the review queue.

        Compares the label's first model word with each family name of the same brand
        (prefix or close spelling), ignoring labels that are a family's known lookalikes.
        """
        brand_label, model = self.split_registry_label(label)
        brand_id = self.brand_for(brand_label)
        model = strip_model_prefix(model)
        if brand_id is None or not model:
            return None
        first = model.split(" ")[0]
        best: tuple[str, float] | None = None
        for fam in self._by_brand.get(brand_id, []):
            if any(p.match(model) for p in fam.lookalikes):
                continue
            key = re.sub(r"[^A-Z0-9]", "", normalize(fam.spec.name))
            squashed = re.sub(r"[^A-Z0-9]", "", model)
            if squashed.startswith(key):
                score = 0.8
            else:
                score = fuzz.ratio(first, key) / 100
                if score < 0.85:
                    continue
            if best is None or score > best[1]:
                best = (fam.spec.id, score)
        return best


def strip_model_prefix(model: str) -> str:
    for prefix in MODEL_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![A-Z0-9]){re.escape(word)}(?![A-Z0-9])", text) is not None


def _drop_contained(matches: list[Match]) -> list[Match]:
    """If 'POLO TRACK' matched, drop a weaker 'POLO' match found inside the same words."""
    kept: list[Match] = []
    for m in matches:
        if any(m.matched_text and m.matched_text in k.matched_text for k in kept):
            continue
        kept.append(m)
    return kept
