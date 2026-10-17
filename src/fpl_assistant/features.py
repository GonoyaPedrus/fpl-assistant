"""Features de forme récente, calculées uniquement sur les journées passées (pas de fuite)."""

from __future__ import annotations

import pandas as pd

ROLLING_COLS = ["total_points", "minutes", "bps", "ict_index", "goals_scored", "assists"]
WINDOWS = (3, 5)


def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des moyennes glissantes décalées d'une journée, par joueur et par saison.

    Une ligne = un match d'un joueur. Les doubles journées comptent comme deux lignes.
    """
    df = df.sort_values(["season", "element", "kickoff_time"]).copy()
    grouped = df.groupby(["season", "element"], sort=False)
    for col in ROLLING_COLS:
        past = grouped[col].shift(1)
        for w in WINDOWS:
            df[f"{col}_avg{w}"] = (
                past.groupby([df["season"], df["element"]])
                .rolling(w, min_periods=1)
                .mean()
                .reset_index(level=[0, 1], drop=True)
            )
    df["n_past_matches"] = grouped.cumcount()
    return df
