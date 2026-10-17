"""FPL assistant: ingestion, prédiction de points et optimisation d'équipe."""

from pathlib import Path

POSITIONS = ("GK", "DEF", "MID", "FWD")
DEFAULT_DATA = Path("data/gw.parquet")
