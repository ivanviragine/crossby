from __future__ import annotations

import sqlite3

import pytest

from carbrain.catalog import open_reviews, record_match, resolve_review
from carbrain.resolve import Match, Resolver, normalize

# Real labels from the SENATRAN July 2026 file.
REGISTRY_CASES = [
    ("VW/POLO TRACK MA", "volkswagen-polo-track"),
    ("VW/POLO 1.6", "volkswagen-polo"),
    ("VW/POLO SEDAN 1.6", "volkswagen-polo"),
    ("VW/T CROSS HL TSI", "volkswagen-t-cross"),
    ("VW/TERA HIGH TSI", "volkswagen-tera"),
    ("VW/GOL 1.0", "volkswagen-gol"),
    ("I/BYD DOLPHIN MINI GS5EV", "byd-dolphin-mini"),
    ("BYD/DOLPHIN MINI GS5EV", "byd-dolphin-mini"),
    ("I/TOYOTA HILUX CD4X4 SRV", "toyota-hilux"),
    ("TOYOTA/CCROSS XRE 20", "toyota-corolla-cross"),
    ("HYUNDAI/CRETA1TA LIMITED", "hyundai-creta"),
    ("HYUNDAI/HB20 1.0M COMFOR", "hyundai-hb20"),
    ("CHEV/ONIX 10MT LT2", "chevrolet-onix"),
    ("I/CHEV TRACKER PREMIER", "chevrolet-tracker"),
    ("HONDA/HR-V EXL CVT", "honda-hr-v"),
    ("I/GWM HAVAL H6 PHEV 19", "gwm-haval-h6"),
    ("GWM/HAVAL H6 HEV2", "gwm-haval-h6"),
    ("FIAT/STRADA ENDURAN CS13", "fiat-strada"),
    ("RENAULT/KWID ZEN 10MT", "renault-kwid"),
    # Found by auditing all 40k real labels (live run, 2026-09-27):
    ("VW/NOVO GOL 1.0", "volkswagen-gol"),
    ("VW/NOVO GOL TL MCV", "volkswagen-gol"),
    ("HYUNDAI/HB2010TA LIMITE", "hyundai-hb20"),
]

# Labels that look similar but are different vehicles we do not track.
NOT_TRACKED = [
    "I/BYD DOLPHIN GS 180EV",  # Dolphin, not Dolphin Mini
    "I/TOYOTA HILUX SWSRXA4FD",  # Hilux SW4 SUV
    "I/TOYOTA HILUXSW4 SRV4X4",  # Hilux SW4 SUV
    "CHEV/ONIX PLUS 10TAT LTZ",  # Onix Plus sedan
    "HYUNDAI/HB20S 1.0M COMF",  # HB20S sedan
    "HYUNDAI/HB20S10TA LIMITE",  # HB20S sedan, 1.0 turbo automatic
    "HYUNDAI/HB20X 1.6A PREMI",  # HB20X
    "VW/GOLF GTI",  # Golf, not Gol
    "HONDA/CG 160 FAN",  # motorcycle
    "GM/CORSA WIND",
]


@pytest.mark.parametrize(("label", "family"), REGISTRY_CASES)
def test_registry_labels(resolver: Resolver, label: str, family: str) -> None:
    match = resolver.resolve_registry(label)
    assert match.family_id == family and match.accepted


@pytest.mark.parametrize("label", NOT_TRACKED)
def test_lookalikes_are_not_matched(resolver: Resolver, label: str) -> None:
    assert resolver.resolve_registry(label).family_id is None


def test_split_registry_label() -> None:
    assert Resolver.split_registry_label("I/BYD DOLPHIN MINI") == ("BYD", "DOLPHIN MINI")
    assert Resolver.split_registry_label("IMP/TOYOTA HILUX SW4") == ("TOYOTA", "HILUX SW4")
    assert Resolver.split_registry_label("vw/pólo track") == ("VW", "POLO TRACK")


@pytest.mark.parametrize(
    ("brand", "model", "family"),
    [
        ("VW - VolksWagen", "Polo Track 1.0 Flex 12V 5p", "volkswagen-polo-track"),
        ("VW - VolksWagen", "T-Cross Comfor. 200 TSI 1.0 Flex 5p Aut.", "volkswagen-t-cross"),
        ("Toyota", "COROLLA CROSS XRE 2.0 16V Flex Aut.", "toyota-corolla-cross"),
        ("GM - Chevrolet", "ONIX HATCH LT 1.0 12V Flex 5p Mec.", "chevrolet-onix"),
        ("GM - Chevrolet", "ONIX PLUS LT 1.0 12V Flex 4p Mec.", None),
        ("GM - Chevrolet", "ONIX LT 1.0 12V Flex 5p Mec.", "chevrolet-onix"),
    ],
)
def test_fipe_names(resolver: Resolver, brand: str, model: str, family: str | None) -> None:
    assert resolver.resolve_fipe(brand, model).family_id == family


@pytest.mark.parametrize(
    ("text", "family"),
    [
        ("quanto custa um polo track 2024?", "volkswagen-polo-track"),
        ("Dolphin Mini ou Kwid?", "byd-dolphin-mini"),
        ("corolla cross hibrido", "toyota-corolla-cross"),
        ("hrv 2023 vale a pena", "honda-hr-v"),
        ("toyota hillux", "toyota-hilux"),  # typo -> fuzzy
    ],
)
def test_free_text(resolver: Resolver, text: str, family: str) -> None:
    assert resolver.find(text)[0].family_id == family


def test_free_text_lists_every_mentioned_family(resolver: Resolver) -> None:
    ids = {m.family_id for m in resolver.find("Dolphin Mini ou Kwid?")}
    assert {"byd-dolphin-mini", "renault-kwid"} <= ids


def test_normalize() -> None:
    assert normalize("  Pálio   fire ") == "PALIO FIRE"


def test_human_decisions_win(seeded: sqlite3.Connection, resolver: Resolver) -> None:
    ambiguous = Match(None, "toyota", 0.5, "ambiguous", "X", ("a", "b"))
    assert record_match(seeded, "senatran_fleet", "TOYOTA/X", ambiguous) is None
    (item,) = open_reviews(seeded)
    resolve_review(seeded, item.id, family_id="toyota-hilux")
    assert open_reviews(seeded) == []
    # Automatic matching later cannot override the confirmed mapping.
    auto = resolver.resolve_registry("TOYOTA/CCROSS XRE 20")
    assert record_match(seeded, "senatran_fleet", "TOYOTA/X", auto) == "toyota-hilux"


def test_review_rejects_unknown_family(seeded: sqlite3.Connection) -> None:
    ambiguous = Match(None, "toyota", 0.5, "ambiguous", "X", ("a", "b"))
    record_match(seeded, "senatran_fleet", "TOYOTA/Y", ambiguous)
    (item,) = open_reviews(seeded)
    with pytest.raises(KeyError):
        resolve_review(seeded, item.id, family_id="toyota-nonexistent")


@pytest.mark.parametrize(
    "label",
    [
        "VW/GOLF 1.6 SPORTLINE",  # known lookalike of Gol
        "CHEV/ONIX PLUS 10TAT LTZ",  # known lookalike of Onix
        "I/TOYOTA HILUX SWSRXA4FD",  # SW4, known lookalike of Hilux
        "FIAT/DUCATO CARGO",  # 'CARGO' contains 'ARGO' but is not an Argo
        "FIAT/DOBLO CARGO 1.4",
        "HONDA/CG 160 FAN",
    ],
)
def test_no_review_suggestion_for_known_non_matches(resolver: Resolver, label: str) -> None:
    assert resolver.suggest_for_label(label) is None


@pytest.mark.parametrize(
    ("label", "family"),
    [
        ("VW/TCROS HL TSI", "volkswagen-t-cross"),  # truncated spelling
        ("HYUNDAI/CRET 1.6", "hyundai-creta"),
    ],
)
def test_review_suggestion_for_plausible_misses(
    resolver: Resolver, label: str, family: str
) -> None:
    suggestion = resolver.suggest_for_label(label)
    assert suggestion is not None and suggestion[0] == family


# Real headlines from the specialist feeds (2026-09-27 live run) and a few traps.
@pytest.mark.parametrize(
    ("headline", "families"),
    [
        (
            "Hyundai i20 terá três novas versões para ocupar de vez o lugar do HB20",
            {"hyundai-hb20"},
        ),
        ("Toyota terá carros com motorização que virou ‘moda’ entre marcas chinesas", set()),  # noqa: RUF001
        ("Flagra: Chevrolet Captiva a gasolina terá nova geração com base chinesa", set()),
        (
            "Strada abre 3.026 carros sobre Polo e alta de 11,6% amplia vantagem em setembro",
            {"fiat-strada", "volkswagen-polo"},
        ),
        (
            "Teste: BYD Atto 2 faz 22 km/l e quer tomar liderança de Creta e T-Cross",
            {"hyundai-creta", "volkswagen-t-cross"},
        ),
        ("Novo polo automotivo de Goiana recebe investimento", set()),
        ("Volkswagen Tera ganha versão mais barata", {"volkswagen-tera"}),
        ("Montadora marca um gol de placa com a nova fábrica", set()),
        # A capital at the start of a sentence proves nothing, but rejecting it would lose
        # real mentions like this one (live headline); "Gol de placa: ..." is a known miss.
        ("Compass cai a R$ 119.990 e diferença para Renegade encolhe", {"jeep-compass"}),
        ("Vale a pena comprar um Gol usado?", {"volkswagen-gol"}),
    ],
)
def test_editorial_mentions(resolver: Resolver, headline: str, families: set[str]) -> None:
    found = {m.family_id for m in resolver.find(headline, editorial=True)}
    assert found == families


def test_user_questions_stay_lenient(resolver: Resolver) -> None:
    assert resolver.find("quanto custa um polo 2024")[0].family_id == "volkswagen-polo"
    assert all(m.family_id != "volkswagen-tera" for m in resolver.find("o carro terá garantia?"))
