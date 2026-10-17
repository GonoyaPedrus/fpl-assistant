"""Historique par journée : saisons passées via vaastav, saison en cours via l'API officielle."""

from __future__ import annotations

import logging
import os
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

from fpl_assistant import DEFAULT_DATA, POSITIONS, api

log = logging.getLogger(__name__)

VAASTAV_URL = (
    "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/{season}/{file}"
)
RELEASE_URL = os.environ.get(
    "FPL_DATA_URL",
    "https://github.com/GonoyaPedrus/fpl-assistant/releases/download/data-latest/gw.parquet",
)
# Les saisons avant 2020-21 n'ont pas la colonne `position` dans merged_gw.csv.
FIRST_SEASON = 2020
# À incrémenter quand le format du parquet change : la GitHub Action republie alors les données.
SCHEMA_VERSION = 1
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


def seasons_before(season: str) -> tuple[str, ...]:
    """Saisons terminées à prendre chez vaastav avant `season`, ex. « 2026-27 »."""
    start = int(season[:4])
    return tuple(f"{y}-{(y + 1) % 100:02d}" for y in range(FIRST_SEASON, start))


def marker(bootstrap: dict) -> str:
    """Identifie un jeu de données : dernière journée validée et version du format."""
    return f"{api.latest_label(bootstrap)}:v{SCHEMA_VERSION}"


def ingest(out: Path = DEFAULT_DATA, strict: bool = False) -> Path:
    """Écrit le parquet complet et, à côté, `latest.txt` avec son `marker`.

    `strict=True` (utilisé par la GitHub Action) échoue au lieu d'ignorer une saison qui ne
    se télécharge pas, pour ne jamais publier un historique incomplet.
    """
    client = api.FPLClient()
    bootstrap = client.bootstrap()
    season = api.current_season(bootstrap)
    frames = []
    try:
        live = api.season_history(client, bootstrap)
        if not live.empty:  # Vide avant la J1.
            frames.append(_clean(live))
    except requests.RequestException as exc:
        if strict:
            raise
        log.warning("Saison en cours ignorée (API FPL) : %s", exc)
    for past in seasons_before(season):
        try:
            frames.append(fetch_season(past))
        except requests.RequestException as exc:
            if strict:
                raise
            log.warning("Saison %s ignorée : %s", past, exc)
    if not frames:
        raise RuntimeError("Aucune saison téléchargée.")
    df = pd.concat(frames, ignore_index=True)
    df["kickoff_time"] = pd.to_datetime(df["kickoff_time"], utc=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    out.with_name("latest.txt").write_text(marker(bootstrap) + "\n")
    return out


def pull(out: Path = DEFAULT_DATA, url: str = RELEASE_URL, timeout: int = 120) -> Path:
    """Télécharge le jeu de données publié par la GitHub Action (évite ~3 min d'API)."""
    resp = requests.get(url, timeout=timeout, stream=True)
    if resp.status_code == 404:
        raise RuntimeError("Aucune donnée publiée pour l'instant : lance `fpl ingest`.")
    resp.raise_for_status()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".part")
    with tmp.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 20):
            f.write(chunk)
    tmp.replace(out)  # Atomique : jamais de parquet tronqué en cas d'interruption.
    return out
