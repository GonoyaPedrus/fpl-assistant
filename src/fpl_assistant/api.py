"""Client minimal de l'API officielle FPL (non documentée, lecture seule, sans authentification)."""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://fantasy.premierleague.com/api"
POSITIONS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


class FPLClient:
    def __init__(self, session: requests.Session | None = None, pause: float = 0.2) -> None:
        if session is None:
            session = requests.Session()
            # L'API renvoie des 429/503 autour des deadlines : on réessaie avec un délai.
            retry = Retry(total=4, backoff_factor=2, status_forcelist=(429, 500, 502, 503, 504))
            session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session = session
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


def team_names(bootstrap: dict[str, Any]) -> dict[int, str]:
    """id → nom long du club (« Man City »), le format utilisé partout, historique compris."""
    return {t["id"]: t["name"] for t in bootstrap["teams"]}


def players_table(bootstrap: dict[str, Any]) -> pd.DataFrame:
    """Joueurs de la saison en cours : poste, club, prix en M£, statut.

    `team` reprend le nom long (« Man City »), comme l'historique vaastav, et `code` est
    l'identifiant stable du joueur d'une saison à l'autre.
    """
    teams = team_names(bootstrap)
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
            # null = pas d'actualité : 100 % pour un joueur disponible (« a »), 0 sinon
            # (parti, prêté…). Une valeur explicite de l'API prime toujours.
            "chance_of_playing": df["chance_of_playing_next_round"].fillna(
                (df["status"] == "a") * 100.0
            ),
        }
    ).reset_index(drop=True)


def next_gameweek(bootstrap: dict[str, Any]) -> int | None:
    return next((e["id"] for e in bootstrap["events"] if e.get("is_next")), None)


def current_season(bootstrap: dict[str, Any]) -> str:
    """Libellé de la saison en cours au format vaastav, ex. « 2026-27 »."""
    if not bootstrap["events"]:
        raise ValueError("Calendrier vide (remise à zéro de l'API entre deux saisons)")
    year = pd.Timestamp(bootstrap["events"][0]["deadline_time"]).year
    return f"{year}-{(year + 1) % 100:02d}"


def finished_gameweeks(bootstrap: dict[str, Any]) -> set[int]:
    """Journées terminées et dont les points sont validés (bonus compris)."""
    return {e["id"] for e in bootstrap["events"] if e["finished"] and e["data_checked"]}


def latest_label(bootstrap: dict[str, Any]) -> str:
    """« saison:journée » de la dernière journée validée, ex. « 2026-27:6 » (0 avant la J1)."""
    return f"{current_season(bootstrap)}:{max(finished_gameweeks(bootstrap), default=0)}"


def season_history(client: FPLClient, bootstrap: dict[str, Any]) -> pd.DataFrame:
    """Historique match par match de la saison en cours, au format de `ingest.fetch_season`.

    Seules les journées terminées et validées sont gardées : pendant une journée, l'API
    liste aussi les matchs pas encore joués (0 point) et des points provisoires.
    """
    gws = finished_gameweeks(bootstrap)
    teams = team_names(bootstrap)
    fixtures = {f["id"]: f for f in client.fixtures() if f["event"] in gws and f["finished"]}
    # Même format de nom que vaastav (« Bukayo Saka »), pas le web_name (« Saka »).
    full_names = {e["id"]: f"{e['first_name']} {e['second_name']}" for e in bootstrap["elements"]}
    players = players_table(bootstrap)[["element", "code", "position"]]
    players["name"] = players["element"].map(full_names)
    frames = []
    for element in players["element"]:
        history = client.player_history(int(element))["history"]
        if history:
            frames.append(pd.DataFrame(history))
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True).rename(columns={"round": "GW"})
    df = df[df["fixture"].isin(fixtures)].copy()
    # Le club vient du match, pas de bootstrap : un joueur transféré en cours de saison
    # garde ainsi son ancien club pour les matchs joués avec lui.
    home = df["fixture"].map(lambda f: teams[fixtures[f]["team_h"]])
    away = df["fixture"].map(lambda f: teams[fixtures[f]["team_a"]])
    df["team"] = home.where(df["was_home"], away)
    for col in ("influence", "creativity", "threat", "ict_index"):
        df[col] = pd.to_numeric(df[col])  # L'API renvoie ces valeurs en chaînes.
    df = df.merge(players, on="element")
    df["season"] = current_season(bootstrap)
    df["price"] = df["value"] / 10
    return df
