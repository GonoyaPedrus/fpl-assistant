"""Historique par journée : saisons passées via vaastav, saison en cours via l'API officielle."""

from __future__ import annotations

import logging
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from fpl_assistant import DEFAULT_DATA, POSITIONS, api

log = logging.getLogger(__name__)

VAASTAV_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/{file}"
)
RELEASE_URL = "https://github.com/GonoyaPedrus/fpl-assistant/releases/download/data-latest/{file}"
# Saisons terminées. Les précédentes n'ont pas la colonne `position` dans merged_gw.csv.
DEFAULT_SEASONS = (
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
)
POSITION_ALIASES = {"GKP": "GK"}  # vaastav écrit parfois GKP (ex. 2021-22 J37).

COLUMNS = [
    "season", "GW", "fixture", "element", "code", "name", "position", "team", "opponent_team",
    "was_home", "kickoff_time", "team_h_score", "team_a_score", "minutes", "total_points",
    "goals_scored", "assists", "clean_sheets", "goals_conceded", "bonus", "bps", "influence",
    "creativity", "threat", "ict_index", "price", "selected", "transfers_balance",
]  # fmt: skip


def _read_csv(season: str, file: str, timeout: int) -> pd.DataFrame:
    resp = requests.get(VAASTAV_URL.format(season=season, file=file), timeout=timeout)
    resp.raise_for_status()
    return pd.read_csv(StringIO(resp.text))


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df["position"] = df["position"].replace(POSITION_ALIASES)
    # Retire les « AM » (assistant manager, 2024-25) qui ne sont pas des joueurs.
    df = df[df["position"].isin(POSITIONS)]
    df = df.drop_duplicates(["element", "fixture"])
    return df[[c for c in COLUMNS if c in df.columns]]


def fetch_season(season: str, timeout: int = 60) -> pd.DataFrame:
    df = _read_csv(season, "gws/merged_gw.csv", timeout)
    df["season"] = season
    # `element` change à chaque saison ; `code` est l'identifiant stable du joueur.
    ids = _read_csv(season, "players_raw.csv", timeout)[["id", "code"]]
    df = df.merge(ids.rename(columns={"id": "element"}), on="element", how="left")
    df["price"] = df["value"] / 10  # `value` est en dixièmes de M£.
    return _clean(df)


def fetch_current_season(client: api.FPLClient | None = None) -> pd.DataFrame:
    """Saison en cours depuis l'API officielle (une requête par joueur, quelques minutes)."""
    client = client or api.FPLClient()
    df = api.season_history(client, client.bootstrap())
    return df if df.empty else _clean(df)  # Vide avant la J1.


def ingest(
    seasons: tuple[str, ...] = DEFAULT_SEASONS, out: Path = DEFAULT_DATA, current: bool = True
) -> Path:
    frames = []
    if current:
        try:
            live = fetch_current_season()
        except requests.RequestException as exc:
            log.warning("Saison en cours ignorée (API FPL) : %s", exc)
        else:
            if not live.empty:
                frames.append(live)
                # Entre deux saisons, l'API décrit encore la saison finie : pas de doublon.
                seasons = tuple(s for s in seasons if s != live["season"].iloc[0])
    for season in seasons:
        try:
            frames.append(fetch_season(season))
        except requests.HTTPError as exc:
            log.warning("Saison %s ignorée : %s", season, exc)
    if not frames:
        raise RuntimeError("Aucune saison téléchargée.")
    df = pd.concat(frames, ignore_index=True)
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def pull(out: Path = DEFAULT_DATA, timeout: int = 120) -> Path:
    """Télécharge le jeu de données publié par la GitHub Action (évite ~3 min d'API)."""
    resp = requests.get(RELEASE_URL.format(file="gw.parquet"), timeout=timeout)
    resp.raise_for_status()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(resp.content)
    return out
