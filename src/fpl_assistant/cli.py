"""Point d'entrée en ligne de commande : `fpl ingest`, `fpl baseline`."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from fpl_assistant import evaluate, features, ingest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="fpl")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Télécharge les données par journée")
    p_ingest.add_argument("--out", type=Path, default=Path("data/gw.parquet"))

    p_base = sub.add_parser("baseline", help="Évalue la baseline forme récente")
    p_base.add_argument("--data", type=Path, default=Path("data/gw.parquet"))
    p_base.add_argument("--test-season", default="2025-26")
    p_base.add_argument("--test-from-gw", type=int, default=20)

    args = parser.parse_args(argv)
    if args.cmd == "ingest":
        print(f"Écrit : {ingest.ingest(out=args.out)}")
    elif args.cmd == "baseline":
        df = features.add_form_features(pd.read_parquet(args.data))
        _, test = evaluate.temporal_split(df, args.test_season, args.test_from_gw)
        for w in features.WINDOWS:
            m = evaluate.metrics(test["total_points"], evaluate.baseline_predict(test, w))
            print(f"forme {w} matchs : RMSE {m['rmse']:.3f}  MAE {m['mae']:.3f}  (n={len(test)})")


if __name__ == "__main__":
    main()
