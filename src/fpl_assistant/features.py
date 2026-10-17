"""Features de forme récente, calculées uniquement sur les journées passées (pas de fuite)."""

from __future__ import annotations

import pandas as pd

ROLLING_COLS = ["total_points", "minutes", "bps", "ict_index", "goals_scored", "assists"]
WINDOWS = (3, 5)
KEYS = ["season", "element"]


def to_gameweeks(df: pd.DataFrame) -> pd.DataFrame:
    """Agrège les matchs par joueur et journée : une double journée devient une seule ligne."""
    agg = {col: "sum" for col in ROLLING_COLS}
    agg |= {"kickoff_time": "min", "fixture": "count"}
    agg |= {c: "first" for c in ("code", "name", "position", "team", "price") if c in df}
    out = df.groupby([*KEYS, "GW"], as_index=False).agg(agg)
    return out.rename(columns={"fixture": "n_fixtures"})


def add_form_features(gw: pd.DataFrame) -> pd.DataFrame:
    """Moyennes glissantes sur les journées précédentes, par joueur et par saison.

    `gw` doit avoir une ligne par (season, element, GW), cf. `to_gameweeks`. Le décalage se
    fait donc par journée : les deux matchs d'une double journée ne se voient pas entre eux.
    """
    gw = gw.sort_values([*KEYS, "GW"]).reset_index(drop=True)
    grouped = gw.groupby(KEYS, sort=False)
    past = grouped[ROLLING_COLS].shift(1)
    past_grouped = past.groupby([gw["season"], gw["element"]], sort=False)
    for w in WINDOWS:
        rolled = past_grouped.rolling(w, min_periods=1).mean().reset_index(level=[0, 1], drop=True)
        gw = gw.join(rolled.add_suffix(f"_avg{w}"))
    gw["n_past_gws"] = grouped.cumcount()
    return gw
