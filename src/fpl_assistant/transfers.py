"""Équipe FPL réelle d'un joueur, lue dans l'API publique : effectif, banque, transferts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fpl_assistant import api

MAX_FREE_TRANSFERS = 5
# Ces chips ne consomment pas les transferts gratuits mis de côté (règle FPL depuis 2024-25).
CHIPS_KEEPING_TRANSFERS = {"wildcard", "freehit"}


@dataclass
class Team:
    gw: int  # Dernière deadline passée : l'effectif est celui de cette journée.
    elements: list[int]
    bank: float  # En M£.
    selling: dict[int, float]  # Prix de revente estimé de chaque joueur, en M£.
    free_transfers: int  # Estimés pour la journée suivante.

    @property
    def budget(self) -> float:
        return self.bank + sum(self.selling.values())


def free_transfers(history: dict[str, Any], started: int, gw: int) -> int:
    """Transferts gratuits pour la journée qui suit `gw`, d'après `entry/{id}/history/`.

    Un par journée à partir de celle qui suit la création de l'équipe (`started`), cumulables
    jusqu'à cinq. Un wildcard ou un free hit les garde tels quels.
    """
    chips = {c["event"]: c["name"] for c in history["chips"]}
    made = {e["event"]: e["event_transfers"] for e in history["current"]}
    ft = 1
    for event in range(started + 1, gw + 1):
        if chips.get(event) not in CHIPS_KEEPING_TRANSFERS:
            ft = min(MAX_FREE_TRANSFERS, max(ft - made.get(event, 0), 0) + 1)
    return ft


def selling_price(bought: int, now: int) -> float:
    """Prix de revente FPL en M£, à partir de prix en dixièmes de M£ comme dans l'API.

    Une baisse est subie entièrement ; une hausse n'est touchée qu'à moitié, arrondie en
    dessous (acheté 7,5, à 7,8 aujourd'hui : revendu 7,6).
    """
    return (min(now, bought) + max(now - bought, 0) // 2) / 10


def _price_at(history: list[dict[str, Any]], gw: int, default: int) -> int:
    """Prix du joueur à la journée `gw` (ou la première après), d'après `element-summary`."""
    rows = sorted((r for r in history if r["round"] >= gw), key=lambda r: r["round"])
    return rows[0]["value"] if rows else default


def purchase_prices(
    client: api.FPLClient,
    entry: int,
    elements: list[int],
    started: int,
    chips: dict[int, str],
    now: dict[int, int],
) -> dict[int, int]:
    """Prix d'achat de chaque joueur de l'effectif, en dixièmes de M£.

    Le dernier transfert entrant fait foi, sauf pendant un free hit (l'effectif revient à
    la normale la journée suivante). Un joueur présent depuis la création de l'équipe est
    compté au prix de la journée de départ : l'API publique ne donne pas son prix exact.
    """
    free_hits = {event for event, name in chips.items() if name == "freehit"}
    bought = {}
    for t in sorted(client.entry_transfers(entry), key=lambda t: t["time"]):
        if t["event"] not in free_hits:
            bought[t["element_in"]] = t["element_in_cost"]
    return {
        e: bought[e]
        if e in bought
        else _price_at(client.player_history(e)["history"], started, now[e])
        for e in elements
    }


def current_team(client: api.FPLClient, entry: int, bootstrap: dict[str, Any]) -> Team:
    """Effectif à la dernière deadline passée.

    Les transferts déjà faits pour la prochaine journée ne sont visibles qu'après sa deadline.
    """
    gw = api.current_gameweek(bootstrap)
    started = client.entry(entry)["started_event"]
    if gw is None or started > gw:
        raise ValueError(
            f"L'équipe démarre en J{started} : transferts illimités jusqu'à sa deadline, "
            "utilise `fpl recommend`."
        )
    history = client.entry_history(entry)
    chips = {c["event"]: c["name"] for c in history["chips"]}
    squad_gw = gw
    while chips.get(squad_gw) == "freehit" and squad_gw > started:
        squad_gw -= 1  # Après un free hit, l'effectif d'avant revient.
    picks = client.entry_picks(entry, squad_gw)
    elements = [p["element"] for p in picks["picks"]]
    now = {e["id"]: e["now_cost"] for e in bootstrap["elements"]}
    bought = purchase_prices(client, entry, elements, started, chips, now)
    return Team(
        gw=gw,
        elements=elements,
        bank=picks["entry_history"]["bank"] / 10,
        selling={e: selling_price(bought[e], now[e]) for e in elements},
        free_transfers=free_transfers(history, started, gw),
    )
