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


def current_season(bootstrap: dict[str, Any]) -> str:
    """Libellé de la saison en cours au format vaastav, ex. « 2026-27 »."""
    year = pd.Timestamp(bootstrap["events"][0]["deadline_time"]).year
    return f"{year}-{(year + 1) % 100:02d}"


def season_history(client: FPLClient, bootstrap: dict[str, Any]) -> pd.DataFrame:
    """Historique match par match de la saison en cours, au format de `ingest.fetch_season`."""
    players = players_table(bootstrap).drop(columns=["price"])
    frames = []
    for element in players["element"]:
        history = client.player_history(int(element))["history"]
        if history:
            frames.append(pd.DataFrame(history))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).rename(columns={"round": "GW"})
    # Après la deadline, l'API liste aussi les matchs de la journée pas encore joués (0 point).
    finished = {f["id"] for f in client.fixtures() if f["finished"]}
    df = df[df["fixture"].isin(finished)]
    for col in ("influence", "creativity", "threat", "ict_index"):
        df[col] = pd.to_numeric(df[col])  # L'API renvoie ces valeurs en chaînes.
    df = df.merge(players[["element", "code", "name", "position", "team"]], on="element")
    df["season"] = current_season(bootstrap)
    df["price"] = df["value"] / 10
    return df
