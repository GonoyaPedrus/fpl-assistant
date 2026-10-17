"""Features de forme récente, calculées uniquement sur les journées passées (pas de fuite)."""

from __future__ import annotations

import pandas as pd

ROLLING_COLS = ["total_points", "minutes", "bps", "ict_index", "goals_scored", "assists"]
WINDOWS = (3, 5)
KEYS = ["season", "element"]
TEAM_WINDOW = 6
FIXTURE_COLS = ["was_home", "team_att", "team_def", "opp_att", "opp_def"]


def add_fixture_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la force récente de l'équipe et de l'adversaire à chaque match d'un joueur.

    `*_att` = buts marqués et `*_def` = buts encaissés par match, en moyenne sur les
    `TEAM_WINDOW` journées précédentes de la saison. Le décalage se fait par journée.
    """
    played = matches.dropna(subset=["team_h_score", "team_a_score"])
    teams = played.groupby(["season", "fixture", "team"], as_index=False).agg(
        GW=("GW", "first"),
        was_home=("was_home", "first"),
        h=("team_h_score", "first"),
        a=("team_a_score", "first"),
    )
    home = teams["was_home"].astype(bool)
    teams["gf"] = teams["h"].where(home, teams["a"])
    teams["ga"] = teams["a"].where(home, teams["h"])

    per_gw = teams.groupby(["season", "team", "GW"], as_index=False)[["gf", "ga"]].mean()
    per_gw = per_gw.sort_values(["season", "team", "GW"])
    past = per_gw.groupby(["season", "team"])[["gf", "ga"]].shift(1)
    rolled = past.groupby([per_gw["season"], per_gw["team"]]).rolling(TEAM_WINDOW, min_periods=1)
    per_gw[["team_att", "team_def"]] = (
        rolled.mean().reset_index(level=[0, 1], drop=True)[["gf", "ga"]].to_numpy()
    )

    strength = teams[["season", "fixture", "team", "GW"]].merge(
        per_gw[["season", "team", "GW", "team_att", "team_def"]], on=["season", "team", "GW"]
    )
    # Chaque match a deux équipes : l'adversaire est l'autre ligne du même match.
    opp = strength.rename(
        columns={"team": "opp", "team_att": "opp_att", "team_def": "opp_def"}
    ).drop(columns="GW")
    pairs = strength.merge(opp, on=["season", "fixture"])
    pairs = pairs[pairs["team"] != pairs["opp"]]
    cols = ["season", "fixture", "team", "team_att", "team_def", "opp_att", "opp_def"]
    return matches.merge(pairs[cols], on=["season", "fixture", "team"], how="left")


def to_gameweeks(df: pd.DataFrame) -> pd.DataFrame:
    """Agrège les matchs par joueur et journée : une double journée devient une seule ligne."""
    agg = {col: "sum" for col in ROLLING_COLS}
    agg |= {"kickoff_time": "min", "fixture": "count"}
    agg |= {c: "first" for c in ("code", "name", "position", "team", "price") if c in df}
    agg |= {c: "mean" for c in FIXTURE_COLS if c in df}
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
