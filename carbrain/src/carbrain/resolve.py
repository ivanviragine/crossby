"""Matching the names sources use for vehicles to our internal vehicle IDs.

Registry labels ("I/BYD DOLPHIN MINI GS5EV", "HYUNDAI/CRETA1TA PLTINUM") and FIPE model
names ("Polo Track 1.0 Flex 12V 5p") are matched with per-family regular expressions:
the longest match wins, and a tie is sent to review instead of guessed. Free text from
users ("quero um hrv 2023") is matched the same way, with a fuzzy fallback for typos in
model names ("toyota hillux").
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from rapidfuzz import fuzz

IMPORT_PREFIXES = ("I/", "IMP/")
#: Marketing words some labels put before the model ("NOVO GOL", "NOVA SAVEIRO").
MODEL_PREFIXES = ("NOVO ", "NOVA ", "NEW ")
AUTO_ACCEPT = 0.9
#: Fuzzy matching only for model names at least this long: in a short name, one changed
#: letter is often another car ("GOL" vs "GOLF").
FUZZY_MIN_LENGTH = 5
FUZZY_MIN_SCORE = 85


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
    #: Free-text guards for names that are also common words: `accent_sensitive`
    #: ("terá" is not Tera) and `proper_noun` ("polo automotivo" is not a Polo).
    text_rules: list[Literal["accent_sensitive", "proper_noun"]] = Field(default_factory=list)
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
        self._compiled: dict[str, _CompiledFamily] = {}
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
            self._compiled[fam.id] = compiled
        self._fuzzy_keys = {
            fam.id: key
            for fam in families
            if len(key := _squash(normalize(fam.name))) >= FUZZY_MIN_LENGTH
        }
        #: Changes whenever brands, families or their patterns change. Data matched with
        #: an older catalog is re-matched (see `Connector.uses_catalog`).
        self.fingerprint = hashlib.sha256(
            json.dumps(
                [[b.model_dump() for b in brands], [f.model_dump() for f in families]],
                sort_keys=True,
            ).encode()
        ).hexdigest()[:12]

    @classmethod
    def from_yaml(cls, path: Path) -> Resolver:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        brands = [BrandSpec(id=k, **v) for k, v in raw["brands"].items()]
        families = [FamilySpec(**f) for f in raw["families"]]
        return cls(brands, families)

    # --- brands -----------------------------------------------------------------

    def brands_in(self, text: str) -> set[str]:
        """Brand IDs named anywhere in free text."""
        _, norm = _aligned_upper(text)
        return {bid for label, bid in self._brand_labels.items() if _has_word(norm, label)}

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

    def find(self, text: str, limit: int = 5, *, editorial: bool = False) -> list[Match]:
        """Families mentioned in free text, best first. Patterns first, then fuzzy.

        `editorial=True` is for headlines and articles, where capitalization is reliable:
        families marked `proper_noun` then need a capital letter or their brand nearby
        ("polo automotivo" is an industrial hub, not a VW Polo). User questions are
        matched leniently. Families marked `accent_sensitive` never match accented text
        ("terá" is a verb, not the VW Tera). No fuzzy matching in editorial mode.
        """
        spaced, norm = _aligned_upper(text)
        mentioned_brands = self.brands_in(text)
        # family -> (match, span of the matched words in the text)
        pattern_hits: dict[str, tuple[Match, tuple[int, int]]] = {}
        for brand_id, fams in self._by_brand.items():
            for fam in fams:
                for pattern in fam.free:
                    m = next(
                        (
                            hit
                            for hit in pattern.finditer(norm)
                            if _acceptable(
                                fam.spec,
                                spaced[hit.start() : hit.end()],
                                editorial=editorial,
                                brand_named=brand_id in mentioned_brands,
                            )
                        ),
                        None,
                    )
                    if not m:
                        continue
                    # A brand named elsewhere in the text makes the match more certain.
                    conf = 0.97 if brand_id in mentioned_brands else 0.92
                    score = conf + len(m.group(0)) / 1000
                    prev = pattern_hits.get(fam.spec.id)
                    if prev is None or score > prev[0].confidence:
                        pattern_hits[fam.spec.id] = (
                            Match(fam.spec.id, brand_id, score, "text_pattern", m.group(0)),
                            m.span(),
                        )
        kept = _drop_contained(list(pattern_hits.values()))
        results = [m for m, _ in sorted(kept, key=lambda h: -h[0].confidence)]
        if len(results) < limit and not editorial:
            claimed = [span for _, span in pattern_hits.values()]
            for family_id, score, window in self._fuzzy(norm, claimed):
                if family_id in pattern_hits:
                    continue
                brand_id = self.families[family_id].brand
                conf = round(score / 100 * 0.85, 3)
                results.append(Match(family_id, brand_id, conf, "fuzzy", window))
        return [
            Match(m.family_id, m.brand_id, min(m.confidence, 0.99), m.method, m.matched_text)
            for m in results[:limit]
        ]

    def _fuzzy(self, norm: str, claimed: list[tuple[int, int]]) -> list[tuple[str, float, str]]:
        """(family, score, words) for model names written with a typo, best first.

        Compares runs of 1-3 words, squashed, with each family's own name. The brand is
        left out on purpose: "Toyota Corolla" must not come close to "Toyota Hilux". Runs
        inside a pattern match are skipped ("Corolla Cross" is not a misspelled T-Cross),
        and so is a run followed by one of the family's lookalikes ("COROLLA HB").
        """
        spans = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[A-Z0-9]+", norm)]
        words = [w for w, _, _ in spans]
        windows = [
            ("".join(words[i : i + n]), i)
            for n in (1, 2, 3)
            for i in range(len(words) - n + 1)
            if not any(
                start <= spans[i][1] and spans[i + n - 1][2] <= end for start, end in claimed
            )
        ]
        hits: list[tuple[str, float, str]] = []
        for family_id, key in self._fuzzy_keys.items():
            lookalikes = self._compiled[family_id].lookalikes
            best: tuple[float, str] | None = None
            for window, i in windows:
                score = fuzz.ratio(key, window)
                if score < FUZZY_MIN_SCORE or (best is not None and score <= best[0]):
                    continue
                if any(p.match(" ".join(words[i:])) for p in lookalikes):
                    continue
                best = (score, window)
            if best is not None:
                hits.append((family_id, best[0], best[1]))
        return sorted(hits, key=lambda h: -h[1])

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


def _squash(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", text)


def strip_model_prefix(model: str) -> str:
    for prefix in MODEL_PREFIXES:
        if model.startswith(prefix):
            return model[len(prefix) :]
    return model


def _fold_char(char: str) -> str:
    base = "".join(c for c in unicodedata.normalize("NFKD", char) if not unicodedata.combining(c))
    return base if len(base) == 1 else char


def _aligned_upper(text: str) -> tuple[str, str]:
    """(original with single spaces, upper-case accent-free copy) of equal length,
    so a match position in the second points at the same characters in the first."""
    spaced = re.sub(r"\s+", " ", unicodedata.normalize("NFC", text)).strip()
    upper = "".join(c.upper() if len(c.upper()) == 1 else c for c in spaced)
    return spaced, "".join(_fold_char(c) for c in upper)


def _acceptable(fam: FamilySpec, raw: str, *, editorial: bool, brand_named: bool) -> bool:
    if "accent_sensitive" in fam.text_rules and any(ord(c) > 127 for c in raw):
        return False
    return not (
        editorial and "proper_noun" in fam.text_rules and not brand_named and not raw[:1].isupper()
    )


def _has_word(text: str, word: str) -> bool:
    return re.search(rf"(?<![A-Z0-9]){re.escape(word)}(?![A-Z0-9])", text) is not None


def _drop_contained(
    hits: list[tuple[Match, tuple[int, int]]],
) -> list[tuple[Match, tuple[int, int]]]:
    """Drop a match that lies inside a longer one at the same place ('POLO' inside
    'POLO TRACK'). The same words elsewhere in the text are a separate mention."""
    kept: list[tuple[Match, tuple[int, int]]] = []
    for hit in sorted(hits, key=lambda h: (h[1][0] - h[1][1], -h[0].confidence)):
        start, end = hit[1]
        if any(k_start <= start and end <= k_end for _, (k_start, k_end) in kept):
            continue
        kept.append(hit)
    return kept
