"""Brazilian states: names as the sources print them (upper case, no accents) to UF codes."""

from __future__ import annotations

from carbrain.resolve import normalize

UF_BY_NAME = {
    "ACRE": "AC",
    "ALAGOAS": "AL",
    "AMAPA": "AP",
    "AMAZONAS": "AM",
    "BAHIA": "BA",
    "CEARA": "CE",
    "DISTRITO FEDERAL": "DF",
    "ESPIRITO SANTO": "ES",
    "GOIAS": "GO",
    "MARANHAO": "MA",
    "MATO GROSSO": "MT",
    "MATO GROSSO DO SUL": "MS",
    "MINAS GERAIS": "MG",
    "PARA": "PA",
    "PARAIBA": "PB",
    "PARANA": "PR",
    "PERNAMBUCO": "PE",
    "PIAUI": "PI",
    "RIO DE JANEIRO": "RJ",
    "RIO GRANDE DO NORTE": "RN",
    "RIO GRANDE DO SUL": "RS",
    "RONDONIA": "RO",
    "RORAIMA": "RR",
    "SANTA CATARINA": "SC",
    "SAO PAULO": "SP",
    "SERGIPE": "SE",
    "TOCANTINS": "TO",
}
UF_CODES = frozenset(UF_BY_NAME.values())
#: Code for rows whose state is missing ("Sem Informação", "Não Identificado", ...).
UNKNOWN_UF = "XX"


def uf_code(name_or_code: str) -> str:
    """'São Paulo', 'SAO PAULO' or 'sp' -> 'SP'. Raises KeyError for anything else."""
    norm = normalize(name_or_code)
    if norm in UF_CODES:
        return norm
    return UF_BY_NAME[norm]


def uf_code_or_unknown(name_or_code: str) -> str:
    """Like `uf_code`, but unknown or missing states become `UNKNOWN_UF`."""
    try:
        return uf_code(name_or_code)
    except KeyError:
        return UNKNOWN_UF


def municipio_id(uf: str, municipio: str) -> str:
    """Stable ID for a municipality, e.g. 'PR/CURITIBA'."""
    return f"{uf_code(uf)}/{normalize(municipio)}"
