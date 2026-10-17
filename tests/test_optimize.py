import itertools

import pandas as pd
import pytest

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
    assert result.captain in result.xi.index


def test_captain_is_best_xi_player():
    result = pick_squad(make_players())
    assert result.captain == result.xi["xp"].idxmax()


def test_rejects_unknown_position_and_missing_xp():
    players = make_players()
    with pytest.raises(ValueError, match="Postes inconnus"):
        pick_squad(pd.concat([players, players.head(1).assign(position="AM")]))
    with pytest.raises(ValueError, match="sans prix ou sans xp"):
        pick_squad(players.assign(xp=players["xp"].where(players.index != 0)))


def test_duplicate_index_labels_are_handled():
    players = make_players()
    players.index = [i // 2 for i in range(len(players))]
    assert len(pick_squad(players).squad) == 15
