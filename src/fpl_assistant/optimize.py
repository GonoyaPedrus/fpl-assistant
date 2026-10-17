"""Sélection d'un effectif FPL de 15 joueurs par programmation linéaire (PuLP)."""

from __future__ import annotations

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


@dataclass
class Squad:
    squad: pd.DataFrame
    xi: pd.DataFrame
    captain: Any  # Label d'index du capitaine dans `players`.
    cost: float
    expected_points: float


def _validate(players: pd.DataFrame) -> None:
    unknown = set(players["position"]) - set(POSITIONS)
    if unknown:
        raise ValueError(f"Postes inconnus : {sorted(unknown)}")
    missing = players[["price", "xp"]].isna().any(axis=1)
    if missing.any():
        raise ValueError(f"{int(missing.sum())} joueurs sans prix ou sans xp")


def pick_squad(
    players: pd.DataFrame, budget: float = 100.0, bench_weight: float = BENCH_WEIGHT
) -> Squad:
    """`players` : colonnes position, team, price (en M£), xp (points attendus)."""
    _validate(players)
    df = players.reset_index(drop=True)  # Index positionnel : noms de variables uniques.
    rows = range(len(df))
    xp, price = df["xp"].to_numpy(), df["price"].to_numpy()

    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    in_squad = [pulp.LpVariable(f"squad_{i}", cat="Binary") for i in rows]
    in_xi = [pulp.LpVariable(f"xi_{i}", cat="Binary") for i in rows]
    captain = [pulp.LpVariable(f"cap_{i}", cat="Binary") for i in rows]

    prob += pulp.lpSum(
        xp[i] * (in_xi[i] + captain[i] + bench_weight * (in_squad[i] - in_xi[i])) for i in rows
    )
    prob += pulp.lpSum(price[i] * in_squad[i] for i in rows) <= budget
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
        prob += pulp.lpSum(in_squad[i] for i in members) <= MAX_PER_TEAM

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise ValueError(f"Pas de solution : {pulp.LpStatus[status]}")

    chosen = [i for i in rows if in_squad[i].value() > 0.5]
    xi = [i for i in rows if in_xi[i].value() > 0.5]
    cap = next(i for i in rows if captain[i].value() > 0.5)
    return Squad(
        squad=players.iloc[chosen],
        xi=players.iloc[xi],
        captain=players.index[cap],
        cost=float(price[chosen].sum()),
        expected_points=float(xp[xi].sum() + xp[cap]),
    )
