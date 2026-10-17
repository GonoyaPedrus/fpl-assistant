"""Équipe recommandée pour la prochaine journée : prédictions LightGBM puis optimiseur."""

from __future__ import annotations

from typing import Any

import pandas as pd

from fpl_assistant import api, features, model
from fpl_assistant.optimize import Squad, pick_squad

UNAVAILABLE = {"i", "s", "u", "n"}  # Blessé, suspendu, parti, non disponible.


def upcoming_matches(
    bootstrap: dict[str, Any], fixtures: list[dict[str, Any]], gw: int
) -> pd.DataFrame:
    """Une ligne par joueur et par match de la journée `gw`, au format de l'historique.

    Les colonnes de résultat (points, minutes, scores…) restent vides : seules les features
    connues avant la deadline sont remplies. Un joueur dont le club ne joue pas n'a pas de
    ligne ; en double journée, il en a deux.
    """
    teams = {t["id"]: t["name"] for t in bootstrap["teams"]}
    games = []
    for f in fixtures:
        if f["event"] != gw:
            continue
        for side, home in (("team_h", True), ("team_a", False)):
            games.append(
                {
                    "fixture": f["id"],
                    "team": teams[f[side]],
                    "was_home": home,
                    "kickoff_time": pd.Timestamp(f["kickoff_time"]),
                }
            )
    if not games:
        return pd.DataFrame()
    players = api.players_table(bootstrap)
    rows = players.merge(pd.DataFrame(games), on="team")
    rows["season"] = api.current_season(bootstrap)
    rows["GW"] = gw
    return rows


def availability(players: pd.DataFrame) -> pd.Series:
    """Probabilité de jouer : `chance_of_playing` en %, 0 pour un joueur indisponible."""
    chance = players["chance_of_playing"] / 100
    return chance.where(~players["status"].isin(UNAVAILABLE), 0.0)


def predict_next(history: pd.DataFrame, upcoming: pd.DataFrame) -> pd.DataFrame:
    """Points attendus (`xp`) par joueur pour la journée de `upcoming`.

    Le modèle est entraîné sur tout l'historique, puis appliqué aux lignes à venir ; leurs
    features de forme et d'adversaire ne voient que les journées déjà jouées.
    """
    matches = pd.concat([history, upcoming], ignore_index=True)
    gw = features.add_form_features(features.to_gameweeks(features.add_fixture_features(matches)))
    season, next_gw = upcoming["season"].iloc[0], upcoming["GW"].iloc[0]
    is_next = (gw["season"] == season) & (gw["GW"] == next_gw)
    trained = model.train(gw[~is_next])
    target = gw[is_next].copy()
    target["raw_xp"] = model.predict(trained, target)
    # `to_gameweeks` ne garde pas le statut médical : il vient de `upcoming`.
    current = upcoming.drop_duplicates("element").set_index("element")
    for col in ("status", "chance_of_playing"):
        target[col] = target["element"].map(current[col])
    target["xp"] = target["raw_xp"] * availability(target)
    return target


def recommend(
    history: pd.DataFrame, client: api.FPLClient | None = None, budget: float = 100.0
) -> tuple[int, Squad]:
    client = client or api.FPLClient()
    bootstrap = client.bootstrap()
    gw = api.next_gameweek(bootstrap)
    if gw is None:
        raise ValueError("Pas de prochaine journée dans le calendrier FPL")
    upcoming = upcoming_matches(bootstrap, client.fixtures(), gw)
    if upcoming.empty:
        raise ValueError(f"Aucun match prévu en J{gw}")
    preds = predict_next(history, upcoming)
    return gw, pick_squad(preds.set_index("element"), budget=budget)
