from fpl_assistant.api import next_gameweek, players_table

BOOTSTRAP = {
    "teams": [{"id": 1, "short_name": "ARS"}, {"id": 2, "short_name": "LIV"}],
    "events": [{"id": 8, "is_next": False}, {"id": 9, "is_next": True}],
    "elements": [
        {
            "id": 10,
            "web_name": "Saka",
            "element_type": 3,
            "team": 1,
            "now_cost": 101,
            "status": "a",
            "chance_of_playing_next_round": None,
        },
        {
            "id": 11,
            "web_name": "Alisson",
            "element_type": 1,
            "team": 2,
            "now_cost": 55,
            "status": "d",
            "chance_of_playing_next_round": 75,
        },
        {
            "id": 12,
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
    assert df["team"].tolist() == ["ARS", "LIV"]
    assert df["price"].tolist() == [10.1, 5.5]


def test_next_gameweek():
    assert next_gameweek(BOOTSTRAP) == 9
