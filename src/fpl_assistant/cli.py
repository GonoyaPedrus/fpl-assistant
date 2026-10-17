"""Point d'entrée en ligne de commande : `fpl ingest`, `fpl baseline`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from fpl_assistant import DEFAULT_DATA, evaluate, features, ingest


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="fpl")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Télécharge l'historique par journée")
    p_ingest.add_argument("--out", type=Path, default=DEFAULT_DATA)

    p_base = sub.add_parser("baseline", help="Évalue la baseline forme récente")
    p_base.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p_base.add_argument("--test-season", default="2025-26")
    p_base.add_argument("--test-from-gw", type=int, default=20)

    args = parser.parse_args(argv)
    if args.cmd == "ingest":
        print(f"Écrit : {ingest.ingest(out=args.out)}")
    elif args.cmd == "baseline":
        gw = features.add_form_features(features.to_gameweeks(pd.read_parquet(args.data)))
        _, test = evaluate.temporal_split(gw, args.test_season, args.test_from_gw)
        # La majorité des lignes sont des joueurs qui ne jouent pas : on sépare les deux cas.
        for label, subset in [("tous", test), ("ont joué", test[test["minutes"] > 0])]:
            for w in features.WINDOWS:
                pred = evaluate.baseline_predict(subset, w)
                m = evaluate.metrics(subset["total_points"], pred)
                print(
                    f"{label:9} forme {w} J : RMSE {m['rmse']:.3f}  MAE {m['mae']:.3f}"
                    f"  (n={len(subset)})"
                )


if __name__ == "__main__":
    main()
