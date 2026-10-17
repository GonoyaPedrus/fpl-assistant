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


def test_no_pointless_transfer_from_an_optimal_squad():
    players = make_players()
    current = pick_squad(players).squad.index
    result = pick_squad(players, current=current, free_transfers=1)
    assert result.transfers == 0 and result.hit_points == 0
    assert set(result.squad.index) == set(current)


def test_transfers_pay_hits_beyond_free_transfers():
    players = make_players()
    current = pick_squad(players).squad.index
    star = players[~players.index.isin(current) & (players["position"] == "MID")]["price"].idxmin()
    players.loc[star, "xp"] = 50.0

    free = pick_squad(players, current=current, free_transfers=5)
    assert star in free.squad.index and free.hit_points == 0

    paid = pick_squad(players, current=current, free_transfers=0)
    assert star in paid.squad.index and paid.hit_points == 4 * paid.transfers

    assert pick_squad(players, current=current, free_transfers=0, hit_cost=1000).transfers == 0
    held = pick_squad(players, current=current, max_transfers=0)
    assert set(held.squad.index) == set(current)


def test_rejects_current_players_missing_from_candidates():
    players = make_players()
    with pytest.raises(ValueError, match="absents"):
        pick_squad(players, current=[*players.index[:14], 999])


def test_keeps_four_players_from_one_club_but_never_adds_one():
    players = make_players()
    current = pick_squad(players).squad.index
    # Un joueur de l'effectif rejoint un club qui en compte déjà trois.
    club = players.loc[current, "team"].value_counts().idxmax()
    mover = players.loc[current][players.loc[current, "team"] != club].index[0]
    players.loc[mover, "team"] = club
    assert (players.loc[current, "team"] == club).sum() == MAX_PER_TEAM + 1

    held = pick_squad(players, current=current, max_transfers=0)
    assert set(held.squad.index) == set(current)
    moved = pick_squad(players, current=current, free_transfers=5)
    assert (moved.squad["team"] == club).sum() <= MAX_PER_TEAM + 1


def test_bench_points_are_reported():
    result = pick_squad(make_players())
    bench = result.squad.loc[~result.squad.index.isin(result.xi.index), "xp"].sum()
    assert result.bench_points == pytest.approx(bench)
