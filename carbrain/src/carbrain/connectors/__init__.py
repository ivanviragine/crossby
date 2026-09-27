"""One module per source. `REGISTRY` maps source IDs to connector factories."""

from __future__ import annotations

from collections.abc import Callable

from carbrain.connectors.base import Connector
from carbrain.connectors.official import Anfavea, AnpWeekly, BcbSgs, IbgeIpca
from carbrain.connectors.pbev import InmetroPbev
from carbrain.connectors.vehicles import SenatranFleet

CONNECTORS: dict[str, Callable[[], Connector]] = {
    "bcb_sgs": BcbSgs,
    "ibge_ipca": IbgeIpca,
    "anp_weekly": AnpWeekly,
    "anfavea": Anfavea,
    "senatran_fleet": SenatranFleet,
    "inmetro_pbev": InmetroPbev,
}
