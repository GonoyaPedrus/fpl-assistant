from fpl_assistant.api import next_gameweek, players_table, season_history

BOOTSTRAP = {
    "teams": [{"id": 1, "name": "Arsenal"}, {"id": 2, "name": "Liverpool"}],
    "events": [{"id": 8, "is_next": False}, {"id": 9, "is_next": True}],
    "elements": [
        {
            "id": 10,
            "code": 223340,
            "web_name": "Saka",
            "element_type": 3,
            "team": 1,
            "now_cost": 101,
            "status": "a",
            "chance_of_playing_next_round": None,
        },
        {
            "id": 11,
            "code": 116535,
            "web_name": "Alisson",
            "element_type": 1,
            "team": 2,
            "now_cost": 55,
            "status": "d",
            "chance_of_playing_next_round": 75,
        },
        {
            "id": 12,
            "code": 1,
            "web_name": "Coach",
            "element_type": 5,
            "team": 2,
            "now_cost": 15,
            "status": "a",
            "chance_of_playing_next_round": None,
        },
    ],
}


def test_players_table_maps_positions_and_prices():
    df = players_table(BOOTSTRAP)
    assert df["name"].tolist() == ["Saka", "Alisson"]  # le type 5 (manager) est exclu
    assert df["position"].tolist() == ["MID", "GK"]
    assert df["team"].tolist() == ["Arsenal", "Liverpool"]
    assert df["code"].tolist() == [223340, 116535]
    assert df["chance_of_playing"].tolist() == [100, 75]
    assert df["price"].tolist() == [10.1, 5.5]


def test_next_gameweek():
    assert next_gameweek(BOOTSTRAP) == 9


class FakeClient:
    def fixtures(self):
        return [{"id": 1, "finished": True}, {"id": 2, "finished": False}]

    def player_history(self, element):
        if element != 10:
            return {"history": []}
        base = {"element": 10, "influence": "12.4", "creativity": "3.0", "threat": "8.0",
                "ict_index": "2.3", "value": 101, "total_points": 6}  # fmt: skip
        return {
            "history": [
                {**base, "fixture": 1, "round": 1},
                {**base, "fixture": 2, "round": 2, "total_points": 0},
            ]
        }


def test_season_history_keeps_finished_fixtures_only():
    bootstrap = {**BOOTSTRAP, "events": [{"id": 1, "deadline_time": "2026-08-21T17:30:00Z"}]}
    df = season_history(FakeClient(), bootstrap)
    assert df["fixture"].tolist() == [1]
    assert df["GW"].tolist() == [1]
    assert df["season"].tolist() == ["2026-27"]
    assert df["price"].tolist() == [10.1]
    assert df["influence"].tolist() == [12.4]
    assert df["team"].tolist() == ["Arsenal"]
