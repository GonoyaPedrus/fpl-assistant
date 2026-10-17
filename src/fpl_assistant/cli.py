"""Point d'entrée en ligne de commande : `fpl ingest`, `fpl baseline`, `fpl train`."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from fpl_assistant import DEFAULT_DATA, evaluate, features, ingest


def _prepare(path: Path) -> pd.DataFrame:
    matches = features.add_fixture_features(pd.read_parquet(path))
    return features.add_form_features(features.to_gameweeks(matches))


def _report(label: str, test: pd.DataFrame, pred: pd.Series) -> None:
    # La majorité des lignes sont des joueurs qui ne jouent pas : on sépare les deux cas.
    played = test["minutes"] > 0
    m_all = evaluate.metrics(test["total_points"], pred)
    m_played = evaluate.metrics(test.loc[played, "total_points"], pred[played])
    top = evaluate.top_k_points(test, pred)
    print(
        f"{label:14} RMSE {m_all['rmse']:.3f}  MAE {m_all['mae']:.3f}  |  ont joué : "
        f"RMSE {m_played['rmse']:.3f}  MAE {m_played['mae']:.3f}  |  top 20 : {top:.2f} pts"
    )


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="fpl")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Télécharge l'historique par journée")
    p_ingest.add_argument("--out", type=Path, default=DEFAULT_DATA)
    p_ingest.add_argument(
        "--no-current", action="store_true", help="Sans la saison en cours (API FPL)"
    )

    for name, help_ in [
        ("baseline", "Évalue la baseline forme récente"),
        ("train", "Entraîne LightGBM et le compare à la baseline"),
    ]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--data", type=Path, default=DEFAULT_DATA)
        p.add_argument("--test-season", default="2025-26")
        p.add_argument("--test-from-gw", type=int, default=20)

    args = parser.parse_args(argv)
    if args.cmd == "ingest":
        print(f"Écrit : {ingest.ingest(out=args.out, current=not args.no_current)}")
        return

    gw = _prepare(args.data)
    train, test = evaluate.temporal_split(gw, args.test_season, args.test_from_gw)
    print(f"Test : {args.test_season} à partir de la J{args.test_from_gw}, {len(test)} lignes")
    for w in features.WINDOWS:
        _report(f"forme {w} J", test, evaluate.baseline_predict(test, w))
    if args.cmd == "train":
        from fpl_assistant import model

        _report("LightGBM", test, model.predict(model.train(train), test))


if __name__ == "__main__":
    main()
