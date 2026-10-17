import numpy as np
import pandas as pd
import pytest

from fpl_assistant.recommend import availability, predict_next, upcoming_matches

BOOTSTRAP = {
    "teams": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}, {"id": 3, "name": "C"}],
    "events": [{"id": 1, "deadline_time": "2026-08-21T17:30:00Z"}],
    "elements": [
        {"id": i, "code": 100 + i, "web_name": f"P{i}", "element_type": 1 + i % 4,
         "team": team, "now_cost": 50 + i, "status": status,
         "chance_of_playing_next_round": chance}
        for i, team, status, chance in [
            (1, 1, "a", None), (2, 1, "i", 0), (3, 2, "d", 50), (4, 3, "a", None),
        ]
    ],
}  # fmt: skip
# J5 : A reçoit B deux fois (double journée), C ne joue pas.
FIXTURES = [
    {"id": 50, "event": 5, "team_h": 1, "team_a": 2, "kickoff_time": "2026-09-20T14:00:00Z"},
    {"id": 51, "event": 5, "team_h": 2, "team_a": 1, "kickoff_time": "2026-09-23T19:00:00Z"},
    {"id": 60, "event": 6, "team_h": 3, "team_a": 1, "kickoff_time": "2026-09-27T14:00:00Z"},
]


def test_upcoming_matches_handles_double_and_blank_gameweeks():
    up = upcoming_matches(BOOTSTRAP, FIXTURES, 5)
    assert up.groupby("element").size().to_dict() == {1: 2, 2: 2, 3: 2}  # P4 (C) ne joue pas.
    assert set(up["season"]) == {"2026-27"} and set(up["GW"]) == {5}
    p1 = up[up["element"] == 1].sort_values("fixture")
    assert p1["was_home"].tolist() == [True, False]


def test_availability():
    players = pd.DataFrame(
        {"status": ["a", "d", "i", "u"], "chance_of_playing": [100.0, 75.0, 0.0, 100.0]}
    )
    assert availability(players).tolist() == [1.0, 0.75, 0.0, 0.0]


def test_predict_next_scores_every_playing_player():
    pytest.importorskip("lightgbm")
    rng = np.random.default_rng(0)
    rows = []
    # Quatre journées jouées : A-B (J1, J3) et B-C / C-A en alternance.
    games = {1: [(1, "A", "B")], 2: [(2, "B", "C")], 3: [(3, "A", "B")], 4: [(4, "C", "A")]}
    club = {1: "A", 2: "A", 3: "B", 4: "C"}
    for gw, matches in games.items():
        for fixture, home, away in matches:
            hs, as_ = rng.integers(0, 4, 2)
            for element, team in club.items():
                if team not in (home, away):
                    continue
                rows.append({
                    "season": "2026-27", "GW": gw, "fixture": fixture, "element": element,
                    "code": 100 + element, "name": f"P{element}", "position": "MID",
                    "team": team, "was_home": team == home, "team_h_score": hs,
                    "team_a_score": as_, "total_points": int(rng.integers(0, 10)),
                    "minutes": 90, "bps": 20, "ict_index": 5.0, "goals_scored": 0,
                    "assists": 0, "price": 5.0,
                    "kickoff_time": pd.Timestamp("2026-08-14", tz="UTC") + pd.Timedelta(weeks=gw),
                })  # fmt: skip
    history = pd.DataFrame(rows)
    up = upcoming_matches(BOOTSTRAP, FIXTURES, 5)
    preds = predict_next(history, up)
    assert sorted(preds["element"]) == [1, 2, 3]
    assert preds.loc[preds["element"] == 2, "xp"].item() == 0.0  # Blessé.
    p3 = preds[preds["element"] == 3].iloc[0]
    assert p3["xp"] == pytest.approx(0.5 * p3["raw_xp"])
    assert preds["n_fixtures"].tolist() == [2, 2, 2]
