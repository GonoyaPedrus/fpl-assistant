"""Ligne de commande : `fpl ingest`, `pull`, `latest`, `baseline`, `train`, `recommend`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from fpl_assistant import (
    DEFAULT_DATA,
    POSITIONS,
    api,
    evaluate,
    features,
    ingest,
    model,
    recommend,
)


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


def _print_squad(gw: int, squad) -> None:
    order = {pos: i for i, pos in enumerate(POSITIONS)}
    table = squad.squad.assign(
        _order=squad.squad["position"].map(order),
        role=lambda d: d.index.map(lambda i: "XI" if i in squad.xi.index else "banc"),
        _bench=lambda d: d["role"] == "banc",
    ).sort_values(["_bench", "_order", "xp"], ascending=[True, True, False])
    print(f"Équipe recommandée pour la J{gw} ({squad.cost:.1f} M£)")
    for element, row in table.iterrows():
        cap = " (C)" if element == squad.captain else ""
        flag = (
            "" if row["status"] == "a" else f"  [{row['status']}, {row['chance_of_playing']:.0f} %]"
        )
        print(
            f"  {row['role']:4} {row['position']:3} {row['name'] + cap:24} {row['team']:15}"
            f" {row['price']:5.1f} M£  {row['xp']:4.1f} pts{flag}"
        )
    print(f"Points attendus (XI + capitaine) : {squad.expected_points:.1f}")


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(prog="fpl")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ingest = sub.add_parser("ingest", help="Télécharge l'historique par journée")
    p_ingest.add_argument("--out", type=Path, default=DEFAULT_DATA)
    p_ingest.add_argument(
        "--strict", action="store_true", help="Échoue si une saison ne se télécharge pas"
    )

    p_pull = sub.add_parser("pull", help="Télécharge les données publiées par la GitHub Action")
    p_pull.add_argument("--out", type=Path, default=DEFAULT_DATA)
    sub.add_parser("latest", help="Dernière journée validée et version du format, ex. 2026-27:6:v1")
    p_rec = sub.add_parser("recommend", help="Équipe recommandée pour la prochaine journée")
    p_rec.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p_rec.add_argument("--budget", type=float, default=100.0, help="En M£ (défaut 100)")

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
        print(f"Écrit : {ingest.ingest(out=args.out, strict=args.strict)}")
        return
    if args.cmd == "pull":
        try:
            print(f"Écrit : {ingest.pull(out=args.out)}")
        except RuntimeError as exc:
            parser.exit(1, f"{exc}\n")
        return
    if args.cmd == "latest":
        try:
            print(ingest.marker(api.FPLClient().bootstrap()))
        except ValueError as exc:  # Rien à signaler : la sortie standard reste vide.
            print(exc, file=sys.stderr)
        return

    if args.cmd == "recommend":
        try:
            next_gw, squad = recommend.recommend(pd.read_parquet(args.data), budget=args.budget)
        except ValueError as exc:
            parser.exit(1, f"{exc}\n")
        _print_squad(next_gw, squad)
        return

    gw = _prepare(args.data)
    train, test = evaluate.temporal_split(gw, args.test_season, args.test_from_gw)
    if test.empty or train.empty:
        parser.error(f"Split vide pour {args.test_season} à partir de la J{args.test_from_gw}")
    print(f"Test : {args.test_season} à partir de la J{args.test_from_gw}, {len(test)} lignes")
    for w in features.WINDOWS:
        _report(f"forme {w} J", test, evaluate.baseline_predict(test, w))
    if args.cmd == "train":
        _report("LightGBM", test, model.predict(model.train(train), test))


if __name__ == "__main__":
    main()
