import numpy as np
import pandas as pd
import pytest

from fpl_assistant import recommend
from fpl_assistant.api import players_table
from fpl_assistant.optimize import BENCH_WEIGHT, pick_squad
from fpl_assistant.recommend import (
    availability,
    candidate_pool,
    check_fresh,
    plan_transfers,
    predict_next,
    upcoming_matches,
)

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


def test_availability_uses_chance_and_status():
    # P1 dispo sans actualité, P2 blessé, P3 incertain à 50 %, P5 parti sans actualité.
    bootstrap = {
        **BOOTSTRAP,
        "elements": [
            *BOOTSTRAP["elements"],
            {**BOOTSTRAP["elements"][3], "id": 5, "status": "u"},
        ],
    }
    assert availability(players_table(bootstrap)).tolist() == [1.0, 0.0, 0.5, 1.0, 0.0]


def test_candidate_pool_keeps_blank_players_and_drops_unavailable():
    preds = pd.DataFrame(
        {"element": [1, 2, 3], "raw_xp": [4.0, 3.0, 2.0], "xp": [4.0, 0.0, 1.0],
         "n_fixtures": [2, 2, 2]}
    )  # fmt: skip
    pool = candidate_pool(BOOTSTRAP, preds)
    assert sorted(pool.index) == [1, 3, 4]  # P2 blessé retiré, P4 (club au repos) gardé.
    assert pool.loc[4, "xp"] == 0.0 and pool.loc[4, "n_fixtures"] == 0
    # Dans l'effectif actuel, le blessé reste candidat : on peut le garder sur le banc.
    assert 2 in candidate_pool(BOOTSTRAP, preds, keep=[2]).index


def test_check_fresh_rejects_stale_history():
    events = [
        {"id": gw, "deadline_time": f"2026-08-{14 + 7 * gw}T17:30:00Z", "finished": True,
         "data_checked": True}
        for gw in (1, 2)
    ]  # fmt: skip
    bootstrap = {**BOOTSTRAP, "events": events}
    check_fresh(pd.DataFrame({"season": ["2026-27"], "GW": [2]}), bootstrap)
    with pytest.raises(ValueError, match="périmées"):
        check_fresh(pd.DataFrame({"season": ["2026-27"], "GW": [1]}), bootstrap)
    with pytest.raises(ValueError, match="avant la saison"):
        check_fresh(pd.DataFrame({"season": ["2025-26"], "GW": [38]}), bootstrap)


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
    # Les matchs à venir reçoivent bien la force des équipes, calculée sur J1 à J4.
    assert preds[["team_att", "team_def", "opp_att", "opp_def"]].notna().all().all()


def league(n_teams=10):
    """Bootstrap de 90 joueurs, en J1 terminée, et leurs prédictions pour la J2."""
    sizes = {1: 10, 2: 30, 3: 30, 4: 20}
    elements = [
        {"id": i, "code": i, "web_name": f"P{i}", "element_type": t, "team": 1 + i % n_teams,
         "now_cost": 40 + i * 7 % 90, "status": "a", "chance_of_playing_next_round": None}
        for i, t in enumerate((t for t, n in sizes.items() for _ in range(n)), start=1)
    ]  # fmt: skip
    bootstrap = {
        "teams": [{"id": k, "name": f"T{k}"} for k in range(1, n_teams + 1)],
        "events": [
            {"id": 1, "is_current": True, "finished": True, "data_checked": True,
             "deadline_time": "2026-08-14T17:30:00Z"},
            {"id": 2, "is_next": True, "finished": False, "data_checked": False,
             "deadline_time": "2026-08-21T17:30:00Z"},
        ],
        "elements": elements,
    }  # fmt: skip
    preds = pd.DataFrame(
        {"element": [e["id"] for e in elements],
         "xp": [1.0 + e["id"] * 13 % 70 / 10 for e in elements]}
    ).assign(raw_xp=lambda d: d["xp"], n_fixtures=1)  # fmt: skip
    return bootstrap, preds


class LeagueClient:
    def __init__(self, bootstrap, squad):
        self._bootstrap, self.squad = bootstrap, squad

    def bootstrap(self):
        return self._bootstrap

    def entry(self, entry):
        return {"started_event": 1}

    def entry_history(self, entry):
        return {"current": [{"event": 1, "event_transfers": 0}], "chips": []}

    def entry_picks(self, entry, gw):
        return {"picks": [{"element": e} for e in self.squad], "entry_history": {"bank": 5}}

    def entry_transfers(self, entry):
        return []


def test_plan_transfers_uses_selling_prices_and_compares_with_holding(monkeypatch):
    bootstrap, preds = league()
    # L'effectif actuel est le meilleur d'il y a une journée ; depuis, les xp ont changé.
    squad = list(pick_squad(candidate_pool(bootstrap, preds)).squad.index)
    preds["xp"] = preds["xp"].to_numpy()[::-1]
    monkeypatch.setattr(recommend, "_next_predictions", lambda *_: (2, preds))

    now = {e["id"]: e["now_cost"] / 10 for e in bootstrap["elements"]}
    # Chaque joueur a pris 0,5 M£ depuis la J1.
    matches = pd.DataFrame(
        {
            "season": "2026-27",
            "GW": 1,
            "element": list(now),
            "price": [p - 0.5 for p in now.values()],
        }
    )
    plan = plan_transfers(matches, 1, client=LeagueClient(bootstrap, squad))
    # Revente : achat à now - 0,5, puis la moitié de la hausse (0,2) : now - 0,3.
    assert plan.pool.loc[squad, "price"].to_numpy() == pytest.approx([now[e] - 0.3 for e in squad])
    assert plan.team.budget == pytest.approx(0.5 + sum(now[e] - 0.3 for e in squad))
    assert set(plan.hold.squad.index) == set(squad)
    assert plan.best.cost <= plan.team.budget + 1e-6
    assert len(plan.sold) == len(plan.bought) == plan.best.transfers > 0
    assert plan.best.hit_points == 4 * max(plan.best.transfers - 1, 0)

    # Le choix compte le banc à 10 % : c'est ce score qui ne peut que s'améliorer.
    def score(s):
        return s.expected_points + BENCH_WEIGHT * s.bench_points - s.hit_points

    assert score(plan.best) >= score(plan.hold) - 1e-6
