"""Client minimal de l'API officielle FPL (non documentée, lecture seule, sans authentification)."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://fantasy.premierleague.com/api"
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


class FPLClient:
    def __init__(self, session: requests.Session | None = None, pause: float = 0.2) -> None:
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = "fpl-assistant"
        self.pause = pause  # Politesse : l'API limite les rafales de requêtes.

    def _get(self, path: str) -> Any:
        resp = self.session.get(f"{BASE_URL}/{path}", timeout=30)
        resp.raise_for_status()
        time.sleep(self.pause)
        return resp.json()

    def bootstrap(self) -> dict[str, Any]:
        return self._get("bootstrap-static/")

    def fixtures(self) -> list[dict[str, Any]]:
        return self._get("fixtures/")

    def player_history(self, element: int) -> dict[str, Any]:
        return self._get(f"element-summary/{element}/")

    def entry_picks(self, entry: int, gw: int) -> dict[str, Any]:
        return self._get(f"entry/{entry}/event/{gw}/picks/")


def players_table(bootstrap: dict[str, Any]) -> pd.DataFrame:
    """Joueurs de la saison en cours : poste, club, prix en M£, statut.

    `team` reprend le nom long (« Man City »), comme l'historique vaastav, et `code` est
    l'identifiant stable du joueur d'une saison à l'autre.
    """
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    df = pd.DataFrame(bootstrap["elements"])
    df = df[df["element_type"].isin(POSITIONS)]
    return pd.DataFrame(
        {
            "element": df["id"],
            "code": df["code"],
            "name": df["web_name"],
            "position": df["element_type"].map(POSITIONS),
            "team": df["team"].map(teams),
            "price": df["now_cost"] / 10,
            "status": df["status"],
            # null = aucune actualité médicale, donc joueur disponible.
            "chance_of_playing": df["chance_of_playing_next_round"].fillna(100),
        }
    ).reset_index(drop=True)


def next_gameweek(bootstrap: dict[str, Any]) -> int | None:
    return next((e["id"] for e in bootstrap["events"] if e.get("is_next")), None)
