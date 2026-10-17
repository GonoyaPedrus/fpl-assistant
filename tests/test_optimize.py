import itertools

import pandas as pd

from fpl_assistant.optimize import MAX_PER_TEAM, SQUAD_SIZE, pick_squad


def make_players() -> pd.DataFrame:
    rows = []
    teams = [f"T{i}" for i in range(10)]
    counter = itertools.count()
    for pos, n in {"GK": 10, "DEF": 30, "MID": 30, "FWD": 20}.items():
        for k in range(n):
            i = next(counter)
            rows.append(
                {
                    "name": f"{pos}{k}",
                    "position": pos,
                    "team": teams[i % len(teams)],
                    "price": 4.0 + (i * 7 % 90) / 10,
                    "xp": 1.0 + (i * 13 % 70) / 10,
                }
            )
    return pd.DataFrame(rows)


def test_squad_respects_rules():
    result = pick_squad(make_players(), budget=100.0)
    squad = result.squad
    assert len(squad) == 15
    assert squad["position"].value_counts().to_dict() == SQUAD_SIZE
    assert squad["team"].value_counts().max() <= MAX_PER_TEAM
    assert result.cost <= 100.0 + 1e-9
    assert len(result.xi) == 11
    assert (result.xi["position"] == "GK").sum() == 1
    assert result.captain in set(result.xi["name"])


def test_captain_is_best_xi_player():
    result = pick_squad(make_players())
    best = result.xi.sort_values("xp").iloc[-1]["name"]
    assert result.captain == best
