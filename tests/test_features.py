import pandas as pd

from fpl_assistant.features import ROLLING_COLS, add_form_features, to_gameweeks


def make_matches(gws: list[int], points: list[int]) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "season": "2024-25",
            "element": 1,
            "GW": gws,
            "fixture": range(len(gws)),
            "kickoff_time": pd.date_range("2024-08-17", periods=len(gws), freq="3D", tz="UTC"),
            "total_points": points,
        }
    )
    for col in ROLLING_COLS[1:]:
        df[col] = 0
    return df


def test_form_uses_only_past_gameweeks():
    out = add_form_features(to_gameweeks(make_matches([1, 2, 3, 4], [2, 10, 6, 1])))
    assert pd.isna(out["total_points_avg3"].iloc[0])
    assert out["total_points_avg3"].tolist()[1:] == [2.0, 6.0, 6.0]
    assert out["n_past_gws"].tolist() == [0, 1, 2, 3]


def test_double_gameweek_is_one_row_and_does_not_leak():
    # J2 est une double journée : 15 puis 3 points.
    gw = to_gameweeks(make_matches([1, 2, 2, 3], [4, 15, 3, 2]))
    assert gw["total_points"].tolist() == [4, 18, 2]
    assert gw["n_fixtures"].tolist() == [1, 2, 1]
    out = add_form_features(gw)
    # La forme de J2 ne voit que J1, pas le premier match de J2.
    assert out.loc[out["GW"] == 2, "total_points_avg3"].item() == 4.0
