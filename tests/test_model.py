import numpy as np
import pandas as pd
import pytest

from fpl_assistant.model import FEATURES, predict, train

pytest.importorskip("lightgbm")


def test_model_learns_from_recent_form():
    rng = np.random.default_rng(0)
    n = 2000
    gw = pd.DataFrame({col: rng.uniform(0, 5, n) for col in FEATURES})
    gw["position"] = rng.choice(["GK", "DEF", "MID", "FWD"], n)
    gw["was_home"] = rng.integers(0, 2, n).astype(bool)
    gw["total_points"] = 2 * gw["total_points_avg3"] + rng.normal(0, 0.5, n)

    model = train(gw.iloc[:1500])
    pred = predict(model, gw.iloc[1500:])
    assert pred.index.equals(gw.index[1500:])
    assert np.corrcoef(pred, gw["total_points"].iloc[1500:])[0, 1] > 0.9
