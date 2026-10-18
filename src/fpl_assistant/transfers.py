"""Équipe FPL réelle d'un joueur, lue dans l'API publique : effectif, banque, transferts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import requests

from fpl_assistant import api
from fpl_assistant.optimize import SQUAD_SIZE

MAX_FREE_TRANSFERS = 5
UNLIMITED = sum(SQUAD_SIZE.values())  # Avant la première deadline ou sous wildcard.
# Ces chips ne consomment pas les transferts gratuits mis de côté (règle FPL depuis 2024-25).
CHIPS_KEEPING_TRANSFERS = {"wildcard", "freehit"}


@dataclass
class Team:
    gw: int  # Dernière deadline passée.
    squad_gw: int  # Journée dont l'effectif est repris : celle d'avant après un free hit.
    elements: list[int]
    bank: float  # En M£.
    selling: dict[int, float]  # Prix de revente estimé de chaque joueur, en M£.
    free_transfers: int  # Pour la prochaine journée : estimés, ou exacts si connecté.
    paid_transfers: int = 0  # Déjà faits avec -4 pour la prochaine journée (connecté).

    @property
    def budget(self) -> float:
        return self.bank + sum(self.selling.values())


def chips_by_event(record: dict[str, Any]) -> dict[int, str]:
    """Journée → chip joué, d'après `entry/{id}/history/`."""
    return {c["event"]: c["name"] for c in record["chips"]}


def free_transfers(record: dict[str, Any], started: int, gw: int) -> int:
    """Transferts gratuits pour la journée qui suit `gw`, d'après `entry/{id}/history/`.

    Un par journée à partir de celle qui suit la création de l'équipe (`started`), cumulables
    jusqu'à cinq. Un wildcard ou un free hit les garde tels quels.
    """
    chips = chips_by_event(record)
    made = {e["event"]: e["event_transfers"] for e in record["current"]}
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


def start_prices(matches: pd.DataFrame, season: str, started: int) -> dict[int, int]:
    """Prix de chaque joueur à la journée `started` (ou la première après), en dixièmes de M£.

    Lu dans l'historique par match déjà téléchargé, plutôt qu'un appel API par joueur.
    """
    rows = matches[(matches["season"] == season) & (matches["GW"] >= started)]
    first = rows.sort_values("GW").drop_duplicates("element")
    return dict(zip(first["element"], (first["price"] * 10).round().astype(int), strict=True))


def purchase_prices(
    transfers: list[dict[str, Any]],
    elements: list[int],
    chips: dict[int, str],
    initial: dict[int, int],
) -> dict[int, int]:
    """Prix d'achat de chaque joueur de l'effectif, en dixièmes de M£.

    Le dernier transfert entrant fait foi, sauf pendant un free hit (l'effectif revient à
    la normale la journée suivante). Un joueur présent depuis la création de l'équipe est
    compté au prix `initial` de la journée de départ : l'API publique ne donne pas son prix
    exact.
    """
    free_hits = {event for event, name in chips.items() if name == "freehit"}
    bought = {}
    for t in sorted(transfers, key=lambda t: t["time"]):
        if t["event"] not in free_hits:
            bought[t["element_in"]] = t["element_in_cost"]
    return {e: bought.get(e, initial[e]) for e in elements}


def current_team(
    client: api.FPLClient, entry: int, bootstrap: dict[str, Any], matches: pd.DataFrame
) -> Team:
    """Effectif à la dernière deadline passée ; `matches` est l'historique par match.

    Les transferts déjà faits pour la prochaine journée ne sont visibles qu'après sa deadline.
    """
    try:
        started = client.entry(entry)["started_event"]
    except requests.HTTPError as exc:
        if exc.response is not None and exc.response.status_code == 404:
            raise ValueError(f"Équipe {entry} introuvable : vérifie l'ID.") from exc
        raise
    gw = api.current_gameweek(bootstrap)
    if gw is None or started > gw:
        raise ValueError(
            f"L'équipe démarre en J{started} : transferts illimités jusqu'à sa deadline, "
            "utilise `fpl recommend`."
        )
    record = client.entry_history(entry)
    chips = chips_by_event(record)
    squad_gw = gw
    while chips.get(squad_gw) == "freehit" and squad_gw > started:
        squad_gw -= 1  # Après un free hit, l'effectif d'avant revient.
    picks = client.entry_picks(entry, squad_gw)
    elements = [p["element"] for p in picks["picks"]]
    now = {e["id"]: e["now_cost"] for e in bootstrap["elements"]}
    unknown = sorted(set(elements) - set(now))
    if unknown:
        raise ValueError(f"Joueurs de l'équipe absents de l'API FPL : {unknown}")
    initial = {**now, **start_prices(matches, api.current_season(bootstrap), started)}
    bought = purchase_prices(client.entry_transfers(entry), elements, chips, initial)
    return Team(
        gw=gw,
        squad_gw=squad_gw,
        elements=elements,
        bank=picks["entry_history"]["bank"] / 10,
        selling={e: selling_price(bought[e], now[e]) for e in elements},
        free_transfers=free_transfers(record, started, gw),
    )


def team_from_my_team(my_team: dict[str, Any], bootstrap: dict[str, Any]) -> Team:
    """Équipe exacte lue dans `my-team/{id}/` (connecté) : prix de revente et transferts
    gratuits donnés par la FPL, transferts déjà faits pour la prochaine journée compris."""
    info = my_team["transfers"]
    made = info.get("made", 0)
    if info.get("status") == "unlimited" or info.get("limit") is None:
        free, paid = UNLIMITED, 0
    else:
        free, paid = max(info["limit"] - made, 0), max(made - info["limit"], 0)
    gw = api.current_gameweek(bootstrap) or 0
    return Team(
        gw=gw,
        squad_gw=api.next_gameweek(bootstrap) or gw,
        elements=[p["element"] for p in my_team["picks"]],
        bank=info["bank"] / 10,
        selling={p["element"]: p["selling_price"] / 10 for p in my_team["picks"]},
        free_transfers=free,
        paid_transfers=paid,
    )
