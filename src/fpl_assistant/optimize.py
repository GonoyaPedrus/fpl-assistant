"""Sélection d'un effectif FPL de 15 joueurs par programmation linéaire (PuLP)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pulp

SQUAD_SIZE = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
MAX_PER_TEAM = 3
BENCH_WEIGHT = 0.1  # Le banc compte un peu, pour éviter des remplaçants à 0 point.


@dataclass
class Squad:
    squad: pd.DataFrame
    xi: pd.DataFrame
    captain: str
    cost: float
    expected_points: float


def pick_squad(
    players: pd.DataFrame, budget: float = 100.0, bench_weight: float = BENCH_WEIGHT
) -> Squad:
    """`players` : colonnes name, position, team, price (en M£), xp (points attendus)."""
    idx = list(players.index)
    prob = pulp.LpProblem("fpl_squad", pulp.LpMaximize)
    in_squad = {i: pulp.LpVariable(f"squad_{i}", cat="Binary") for i in idx}
    in_xi = {i: pulp.LpVariable(f"xi_{i}", cat="Binary") for i in idx}
    captain = {i: pulp.LpVariable(f"cap_{i}", cat="Binary") for i in idx}
    xp = players["xp"]

    prob += pulp.lpSum(
        xp[i] * (in_xi[i] + captain[i] + bench_weight * (in_squad[i] - in_xi[i])) for i in idx
    )
    prob += pulp.lpSum(players.at[i, "price"] * in_squad[i] for i in idx) <= budget
    prob += pulp.lpSum(in_xi[i] for i in idx) == 11
    prob += pulp.lpSum(captain[i] for i in idx) == 1
    for i in idx:
        prob += in_xi[i] <= in_squad[i]
        prob += captain[i] <= in_xi[i]
    for pos, n in SQUAD_SIZE.items():
        members = [i for i in idx if players.at[i, "position"] == pos]
        prob += pulp.lpSum(in_squad[i] for i in members) == n
        prob += pulp.lpSum(in_xi[i] for i in members) >= XI_MIN[pos]
        prob += pulp.lpSum(in_xi[i] for i in members) <= XI_MAX[pos]
    for team in players["team"].unique():
        members = [i for i in idx if players.at[i, "team"] == team]
        prob += pulp.lpSum(in_squad[i] for i in members) <= MAX_PER_TEAM

    status = prob.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[status] != "Optimal":
        raise ValueError(f"Pas de solution : {pulp.LpStatus[status]}")

    chosen = [i for i in idx if in_squad[i].value() > 0.5]
    xi = [i for i in idx if in_xi[i].value() > 0.5]
    cap = next(i for i in idx if captain[i].value() > 0.5)
    squad = players.loc[chosen]
    return Squad(
        squad=squad,
        xi=players.loc[xi],
        captain=players.at[cap, "name"],
        cost=float(squad["price"].sum()),
        expected_points=float(players.loc[xi, "xp"].sum() + players.at[cap, "xp"]),
    )
