import pandas as pd

from fpl_assistant.features import add_form_features


def test_form_uses_only_past_matches():
    df = pd.DataFrame(
        {
            "season": ["2024-25"] * 4,
            "element": [1, 1, 1, 1],
            "GW": [1, 2, 3, 4],
            "kickoff_time": pd.date_range("2024-08-17", periods=4, freq="7D", tz="UTC"),
            "total_points": [2, 10, 6, 1],
        }
    )
    for col in ["minutes", "bps", "ict_index", "goals_scored", "assists"]:
        df[col] = 0
    out = add_form_features(df).sort_values("GW")
    assert pd.isna(out["total_points_avg3"].iloc[0])
    assert out["total_points_avg3"].tolist()[1:] == [2.0, 6.0, 6.0]
    assert out["n_past_matches"].tolist() == [0, 1, 2, 3]
