"""Modèle LightGBM de points par joueur et par journée."""

from __future__ import annotations

import pandas as pd

from fpl_assistant import POSITIONS
from fpl_assistant.features import FIXTURE_COLS, ROLLING_COLS, WINDOWS

FEATURES = [
    *(f"{col}_avg{w}" for col in ROLLING_COLS for w in WINDOWS),
    "n_past_gws",
    "n_fixtures",
    "price",
    "position",
    *FIXTURE_COLS,
]
PARAMS = {
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 50,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "verbose": -1,
}


def design_matrix(gw: pd.DataFrame) -> pd.DataFrame:
    X = gw[FEATURES].copy()
    X["position"] = pd.Categorical(X["position"], categories=POSITIONS)
    return X.astype({"was_home": float})


def train(gw: pd.DataFrame, seed: int = 0):
    from lightgbm import LGBMRegressor  # Dépendance optionnelle (extra `ml`).

    model = LGBMRegressor(**PARAMS, random_state=seed)
    model.fit(design_matrix(gw), gw["total_points"])
    return model


def predict(model, gw: pd.DataFrame) -> pd.Series:
    return pd.Series(model.predict(design_matrix(gw)), index=gw.index)
