"""Features de forme récente, calculées uniquement sur les journées passées (pas de fuite)."""

from __future__ import annotations

import pandas as pd

ROLLING_COLS = ["total_points", "minutes", "bps", "ict_index", "goals_scored", "assists"]
WINDOWS = (3, 5)
KEYS = ["season", "element"]
TEAM_WINDOW = 6
FIXTURE_COLS = ["was_home", "team_att", "team_def", "opp_att", "opp_def"]


def _lagged_mean(df: pd.DataFrame, keys: list[str], cols: list[str], window: int) -> pd.DataFrame:
    """Moyenne glissante des `window` lignes précédentes par groupe (la ligne courante exclue).

    `df` doit être trié par groupe puis par journée.
    """
    past = df.groupby(keys, sort=False)[cols].shift(1)
    rolled = past.groupby([df[k] for k in keys], sort=False).rolling(window, min_periods=1)
    return rolled.mean().reset_index(level=list(range(len(keys))), drop=True)


def add_fixture_features(matches: pd.DataFrame) -> pd.DataFrame:
    """Ajoute la force récente de l'équipe et de l'adversaire à chaque match d'un joueur.

    `*_att` = buts marqués et `*_def` = buts encaissés par match, en moyenne sur les
    `TEAM_WINDOW` journées précédentes de la saison. Le décalage se fait par journée.
    Les matchs pas encore joués (scores manquants) reçoivent aussi ces features, calculées
    sur les journées jouées avant eux : c'est ce qui sert à prédire la prochaine journée.
    """
    n_teams = matches.groupby(["season", "fixture"])["team"].nunique()
    if (n_teams > 2).any():
        bad = n_teams[n_teams > 2].index.tolist()[:5]
        raise ValueError(f"Matchs avec plus de deux clubs (étiquettes incohérentes) : {bad}")
    teams = matches.groupby(["season", "fixture", "team"], as_index=False).agg(
        GW=("GW", "first"),
        was_home=("was_home", "first"),
        h=("team_h_score", "first"),
        a=("team_a_score", "first"),
    )
    home = teams["was_home"].astype(bool)
    teams["gf"] = teams["h"].where(home, teams["a"])
    teams["ga"] = teams["a"].where(home, teams["h"])

    per_gw = teams.groupby(["season", "team", "GW"], as_index=False)[["gf", "ga"]].mean()
    lagged = _lagged_mean(per_gw, ["season", "team"], ["gf", "ga"], TEAM_WINDOW)
    per_gw = per_gw.join(lagged.rename(columns={"gf": "team_att", "ga": "team_def"}))

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
    return matches.merge(
        pairs[cols], on=["season", "fixture", "team"], how="left", validate="many_to_one"
    )


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
    for w in WINDOWS:
        gw = gw.join(_lagged_mean(gw, KEYS, ROLLING_COLS, w).add_suffix(f"_avg{w}"))
    gw["n_past_gws"] = gw.groupby(KEYS, sort=False).cumcount()
    return gw
