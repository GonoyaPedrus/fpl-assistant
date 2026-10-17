"""Évaluation honnête : split temporel par journée et baseline « forme récente »."""

from __future__ import annotations

import numpy as np
import pandas as pd


def temporal_split(df: pd.DataFrame, test_season: str, test_from_gw: int):
    """Train = tout ce qui précède (season, GW) ; test = la suite de la saison de test."""
    is_test = (df["season"] == test_season) & (df["GW"] >= test_from_gw)
    is_train = (df["season"] < test_season) | (
        (df["season"] == test_season) & (df["GW"] < test_from_gw)
    )
    return df[is_train], df[is_test]


def metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    err = np.asarray(y_pred, dtype=float) - np.asarray(y_true, dtype=float)
    return {"rmse": float(np.sqrt(np.mean(err**2))), "mae": float(np.mean(np.abs(err)))}


def baseline_predict(df: pd.DataFrame, window: int = 5) -> pd.Series:
    """Prédit la moyenne des `window` derniers matchs (0 si aucun historique)."""
    return df[f"total_points_avg{window}"].fillna(0.0)
