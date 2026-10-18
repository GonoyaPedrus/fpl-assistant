"""Sélection d'un effectif FPL de 15 joueurs par programmation linéaire (PuLP)."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from typing import Any

import pandas as pd
import pulp

from fpl_assistant import POSITIONS

SQUAD_SIZE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_TEAM = 3
BENCH_WEIGHT = 0.1  # Le banc compte un peu, pour éviter des remplaçants à 0 point.
HIT_COST = 4  # Points retirés par transfert au-delà des transferts gratuits.
# Départage les ex aequo en faveur de l'effectif actuel : pas de transfert pour 0 point.
TRANSFER_EPSILON = 1e-3


@dataclass
class Squad:
    squad: pd.DataFrame
    xi: pd.DataFrame
    captain: Any  # Label d'index du capitaine dans `players`.
    cost: float
    expected_points: float  # XI + capitaine, avant pénalité.
    bench_points: float = 0.0  # Points attendus du banc, comptés à `bench_weight` dans le choix.
    transfers: int = 0  # Joueurs achetés par rapport à l'effectif actuel.
    hit_points: float = 0.0  # Pénalité des transferts payants.


def _validate(players: pd.DataFrame) -> None:
    unknown = set(players["position"]) - set(POSITIONS)
    if unknown:
        raise ValueError(f"Postes inconnus : {sorted(unknown)}")
    missing = players[["price", "xp"]].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} joueurs sans prix ou sans xp")


def pick_squad(
    players: pd.DataFrame,
    budget: float = 100.0,
    bench_weight: float = BENCH_WEIGHT,
    current: Collection[Any] | None = None,
    free_transfers: int = 1,
    hit_cost: float = HIT_COST,
    max_transfers: int | None = None,
) -> Squad:
    """`players` : colonnes position, team, price (en M£), xp (points attendus).

    Sans `current`, l'effectif est choisi librement. Avec `current` (labels d'index de
    l'effectif actuel), chaque joueur acheté est un transfert : au-delà de `free_transfers`,
    chacun coûte `hit_cost` points, et `max_transfers` plafonne leur nombre. Le prix d'un
    joueur déjà dans l'effectif doit alors être son prix de revente.
    """
    _validate(players)
    df = players.reset_index(drop=True)  # Index positionnel : noms de variables uniques.
    rows = range(len(df))
    xp, price = df["xp"].to_numpy(), df["price"].to_numpy()
    missing = set(current) - set(players.index) if current is not None else set()
    if missing:
        raise ValueError(f"Joueurs de l'effectif absents des candidats : {sorted(missing)}")
    owned = players.index.isin([] if current is None else list(current))

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    in_squad = [pulp.LpVariable(f"squad_{i}", cat="Binary") for i in rows]
    in_xi = [pulp.LpVariable(f"xi_{i}", cat="Binary") for i in rows]
    captain = [pulp.LpVariable(f"cap_{i}", cat="Binary") for i in rows]

    points = pulp.lpSum(
        xp[i] * (in_xi[i] + captain[i] + bench_weight * (in_squad[i] - in_xi[i])) for i in rows
    )
    if current is None:
        prob += points
    else:
        bought = pulp.lpSum(in_squad[i] for i in rows if not owned[i])
        hits = pulp.LpVariable("hits", lowBound=0)
        prob += points - hit_cost * hits - TRANSFER_EPSILON * bought
        prob += hits >= bought - free_transfers
        if max_transfers is not None:
            prob += bought <= max_transfers
    # Tolérance : les prix sont des multiples de 0,1 M£, mais leur somme en flottants non.
    prob += pulp.lpSum(price[i] * in_squad[i] for i in rows) <= budget + 1e-6
    prob += pulp.lpSum(in_squad) == sum(SQUAD_SIZE.values())
    prob += pulp.lpSum(in_xi) == 11
    prob += pulp.lpSum(captain) == 1
    for i in rows:
        prob += in_xi[i] <= in_squad[i]
        prob += captain[i] <= in_xi[i]
    for pos, members in df.groupby("position").groups.items():
        prob += pulp.lpSum(in_squad[i] for i in members) == SQUAD_SIZE[pos]
        prob += pulp.lpSum(in_xi[i] for i in members) >= XI_MIN[pos]
        prob += pulp.lpSum(in_xi[i] for i in members) <= XI_MAX[pos]
    for members in df.groupby("team").groups.values():
        # Un joueur transféré d'un club à l'autre peut en laisser 4 du même club dans
        # l'effectif : on peut les garder, pas en ajouter.
        limit = max(MAX_PER_TEAM, int(owned[members].sum()))
        prob += pulp.lpSum(in_squad[i] for i in members) <= limit

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise ValueError(f"Pas de solution : {pulp.LpStatus[status]}")

    chosen = [i for i in rows if in_squad[i].value() > 0.5]
    xi = [i for i in rows if in_xi[i].value() > 0.5]
    bench = [i for i in chosen if i not in xi]
    cap = next(i for i in rows if captain[i].value() > 0.5)
    n_bought = sum(not owned[i] for i in chosen) if current is not None else 0
    return Squad(
        squad=players.iloc[chosen],
        xi=players.iloc[xi],
        captain=players.index[cap],
        cost=float(price[chosen].sum()),
        expected_points=float(xp[xi].sum() + xp[cap]),
        bench_points=float(xp[bench].sum()),
        transfers=n_bought,
        hit_points=hit_cost * max(n_bought - free_transfers, 0) if current is not None else 0.0,
    )


def lineup(squad: Squad) -> pd.DataFrame:
    """Effectif dans l'ordre FPL, avec `slot` (1 à 15), `bench`, `is_captain`, `is_vice_captain`.

    Le XI par poste, puis le banc dans l'ordre des remplacements : le gardien en 12, puis les
    joueurs de champ du plus au moins prometteur. Le vice-capitaine est le meilleur xp du XI
    après le capitaine.
    """
    order = {pos: i for i, pos in enumerate(POSITIONS)}
    table = squad.squad.assign(
        bench=~squad.squad.index.isin(squad.xi.index),
        _order=squad.squad["position"].map(order),
    )
    table["_rank"] = table["_order"].where(~table["bench"], (table["position"] != "GK") * 1)
    table = table.sort_values(["bench", "_rank", "xp"], ascending=[True, True, False])
    vice = squad.xi[squad.xi.index != squad.captain]["xp"].idxmax()
    return table.drop(columns=["_order", "_rank"]).assign(
        slot=range(1, len(table) + 1),
        is_captain=table.index == squad.captain,
        is_vice_captain=table.index == vice,
    )
