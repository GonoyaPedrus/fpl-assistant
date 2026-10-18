"""Équipe et transferts pour la prochaine journée : prédictions LightGBM puis optimiseur."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import pandas as pd

from fpl_assistant import POSITIONS, api, features, model, transfers
from fpl_assistant.optimize import Squad, pick_squad


def upcoming_matches(
    bootstrap: dict[str, Any], fixtures: list[dict[str, Any]], gw: int
) -> pd.DataFrame:
    """Une ligne par joueur et par match de la journée `gw`, au format de l'historique.

    Les colonnes de résultat (points, minutes, scores…) restent vides : seules les features
    connues avant la deadline sont remplies. Un joueur dont le club ne joue pas n'a pas de
    ligne ; en double journée, il en a deux.
    """
    teams = api.team_names(bootstrap)
    games = [
        {
            "fixture": f["id"],
            "team": teams[f[side]],
            "was_home": side == "team_h",
            "kickoff_time": pd.Timestamp(f["kickoff_time"]),
        }
        for f in fixtures
        if f["event"] == gw
        for side in ("team_h", "team_a")
    ]
    if not games:
        return pd.DataFrame()
    rows = api.players_table(bootstrap).merge(pd.DataFrame(games), on="team")
    rows["season"] = api.current_season(bootstrap)
    rows["GW"] = gw
    return rows


def availability(players: pd.DataFrame) -> pd.Series:
    """Probabilité de jouer, d'après `chance_of_playing` (en %, cf. `api.players_table`)."""
    return players["chance_of_playing"] / 100


def predict_next(history: pd.DataFrame, upcoming: pd.DataFrame) -> pd.DataFrame:
    """Points attendus (`xp`) par joueur pour la journée de `upcoming`.

    Le modèle est entraîné sur tout l'historique, puis appliqué aux lignes à venir ; leurs
    features de forme et d'adversaire ne voient que les journées déjà jouées.
    """
    gw = features.prepare(pd.concat([history, upcoming], ignore_index=True))
    season, next_gw = upcoming["season"].iloc[0], upcoming["GW"].iloc[0]
    is_next = (gw["season"] == season) & (gw["GW"] == next_gw)
    trained = model.train(gw[~is_next])
    target = gw[is_next].copy()
    # Une prédiction légèrement négative ne doit pas classer un titulaire sous un blessé à 0.
    target["raw_xp"] = model.predict(trained, target).clip(lower=0)
    target["xp"] = target["raw_xp"] * availability(target)
    return target


def check_fresh(history: pd.DataFrame, bootstrap: dict[str, Any]) -> None:
    """Refuse un historique qui n'a pas la dernière journée validée par la FPL."""
    done = api.finished_gameweeks(bootstrap)
    if not done:
        return  # Avant la J1 : rien à attendre de la saison en cours.
    season = api.current_season(bootstrap)
    have = history.loc[history["season"] == season, "GW"].max()
    if pd.isna(have) or have < max(done):
        raise ValueError(
            f"Données périmées : J{max(done)} est validée mais l'historique s'arrête "
            f"{'avant la saison' if pd.isna(have) else f'à la J{have}'}. "
            "Lance `fpl pull` (ou `fpl ingest`)."
        )


def candidate_pool(
    bootstrap: dict[str, Any], preds: pd.DataFrame, keep: Collection[int] = ()
) -> pd.DataFrame:
    """Tous les joueurs disponibles, indexés par `element`, avec leur `xp`.

    Un joueur dont le club ne joue pas reste candidat à 0 point (utile au banc) ; un joueur
    indisponible (blessé, suspendu, parti) est retiré pour ne jamais être choisi, sauf s'il
    est dans `keep` (l'effectif actuel) : on peut le garder sur le banc plutôt que payer -4.
    """
    players = api.players_table(bootstrap)
    players = players[(availability(players) > 0) | players["element"].isin(list(keep))]
    cols = ["element", "raw_xp", "xp", "n_fixtures"]
    pool = players.merge(preds[cols], on="element", how="left")
    pool[["raw_xp", "xp", "n_fixtures"]] = pool[["raw_xp", "xp", "n_fixtures"]].fillna(0)
    return pool.set_index("element")


def _next_predictions(
    history: pd.DataFrame, client: api.FPLClient, bootstrap: dict[str, Any]
) -> tuple[int, pd.DataFrame]:
    gw = api.next_gameweek(bootstrap)
    if gw is None:
        raise ValueError("Pas de prochaine journée dans le calendrier FPL")
    check_fresh(history, bootstrap)
    upcoming = upcoming_matches(bootstrap, client.fixtures(), gw)
    if upcoming.empty:
        raise ValueError(f"Aucun match prévu en J{gw}")
    return gw, predict_next(history, upcoming)


def recommend(
    history: pd.DataFrame, client: api.FPLClient | None = None, budget: float = 100.0
) -> tuple[int, Squad]:
    client = client or api.FPLClient()
    bootstrap = client.bootstrap()
    gw, preds = _next_predictions(history, client, bootstrap)
    return gw, pick_squad(candidate_pool(bootstrap, preds), budget=budget)


@dataclass
class TransferPlan:
    gw: int
    team: transfers.Team
    pool: pd.DataFrame  # Candidats, prix de revente pour l'effectif actuel.
    best: Squad  # Avec les transferts conseillés.
    hold: Squad  # Meilleur XI sans aucun transfert, pour comparer.

    @property
    def sold(self) -> list[int]:
        return [e for e in self.team.elements if e not in self.best.squad.index]

    @property
    def bought(self) -> list[int]:
        return [e for e in self.best.squad.index if e not in self.team.elements]

    def pairs(self) -> list[tuple[int, int]]:
        """(sortant, entrant) : autant de départs que d'arrivées par poste, appariés poste par
        poste et du plus au moins prometteur."""
        order = {pos: i for i, pos in enumerate(POSITIONS)}

        def ranked(elements: list[int]) -> list[int]:
            rows = self.pool.loc[elements]
            rows = rows.assign(_order=rows["position"].map(order))
            return list(rows.sort_values(["_order", "xp"], ascending=[True, False]).index)

        return list(zip(ranked(self.sold), ranked(self.bought), strict=True))


def plan_transfers(
    history: pd.DataFrame,
    entry: int,
    client: api.FPLClient | None = None,
    free_transfers: int | None = None,
    max_transfers: int | None = None,
    logged_in: bool = False,
    max_hits: int | None = None,
) -> TransferPlan:
    """Transferts conseillés pour la prochaine journée, depuis l'équipe FPL `entry`.

    `logged_in` : `client` est connecté au compte, l'équipe est lue exactement dans
    `my-team` ; sinon elle est reconstituée depuis l'API publique, transferts gratuits et
    prix de revente estimés. `free_transfers` remplace ce nombre. `max_hits` plafonne les
    transferts payants (-4 chacun).
    """
    client = client or api.FPLClient()
    bootstrap = client.bootstrap()
    check_fresh(history, bootstrap)  # Avant les appels API de l'équipe et l'entraînement.
    if logged_in:
        team = transfers.team_from_my_team(client.my_team(entry), bootstrap)
    else:
        team = transfers.current_team(client, entry, bootstrap, history)
    if free_transfers is not None:
        team.free_transfers = free_transfers
    if max_hits is not None and team.free_transfers < transfers.UNLIMITED:
        # Les -4 déjà pris pour cette journée (un passage précédent) comptent dans le plafond.
        cap = team.free_transfers + max(max_hits - team.paid_transfers, 0)
        max_transfers = cap if max_transfers is None else min(max_transfers, cap)
    gw, preds = _next_predictions(history, client, bootstrap)
    pool = candidate_pool(bootstrap, preds, keep=team.elements)
    # Un joueur gardé « coûte » son prix de revente, celui qui compte dans le budget.
    pool["price"] = pd.Series(team.selling).reindex(pool.index).fillna(pool["price"])
    best = pick_squad(
        pool,
        budget=team.budget,
        current=team.elements,
        free_transfers=team.free_transfers,
        max_transfers=max_transfers,
    )
    # Sans transfert, seuls le XI et le capitaine restent à choisir parmi les 15.
    hold = pick_squad(pool.loc[team.elements], budget=team.budget, current=team.elements)
    return TransferPlan(gw, team, pool, best, hold)
