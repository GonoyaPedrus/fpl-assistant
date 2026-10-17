"""Téléchargement des données par journée depuis le dépôt public vaastav/Fantasy-Premier-League."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests

VAASTAV_URL = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/gws/merged_gw.csv"
# Les saisons antérieures n'ont pas la colonne `position` dans merged_gw.csv.
DEFAULT_SEASONS = ("2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27")

COLUMNS = [
    "season",
    "GW",
    "element",
    "name",
    "position",
    "team",
    "opponent_team",
    "was_home",
    "kickoff_time",
    "minutes",
    "total_points",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "bonus",
    "bps",
    "influence",
    "creativity",
    "threat",
    "ict_index",
    "value",
    "selected",
    "transfers_balance",
]


def fetch_season(season: str, timeout: int = 60) -> pd.DataFrame:
    resp = requests.get(VAASTAV_URL.format(season=season), timeout=timeout)
    resp.raise_for_status()
    df = pd.read_csv(StringIO(resp.text))
    df["season"] = season
    # Les postes « AM » (assistant manager, 2024-25) ne sont pas des joueurs.
    df = df[df["position"].isin(["GK", "DEF", "MID", "FWD"])]
    return df[[c for c in COLUMNS if c in df.columns]]


def ingest(seasons: tuple[str, ...] = DEFAULT_SEASONS, out: Path = Path("data/gw.parquet")) -> Path:
    frames = [fetch_season(s) for s in seasons]
    df = pd.concat(frames, ignore_index=True)
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out
