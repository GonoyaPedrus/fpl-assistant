"""Ligne de commande : `fpl ingest`, `pull`, `latest`, `baseline`, `train`, `recommend`,
`transfers`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

from fpl_assistant import (
    DEFAULT_DATA,
    POSITIONS,
    api,
    evaluate,
    features,
    ingest,
    model,
    optimize,
    recommend,
    transfers,
)

POSITION_ORDER = {pos: i for i, pos in enumerate(POSITIONS)}


def _prepare(path: Path) -> pd.DataFrame:
    return features.prepare(pd.read_parquet(path))


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


def _print_squad(title: str, squad) -> None:
    table = squad.squad.assign(
        bench=~squad.squad.index.isin(squad.xi.index),
        _order=squad.squad["position"].map(POSITION_ORDER),
    )
    # Titulaires par poste ; banc dans l'ordre des remplacements FPL : gardien d'abord,
    # puis les joueurs de champ du plus au moins prometteur.
    table["_rank"] = table["_order"].where(~table["bench"], (table["position"] != "GK") * 1)
    table = table.sort_values(["bench", "_rank", "xp"], ascending=[True, True, False])
    print(f"{title} ({squad.cost:.1f} M£)")
    for element, row in table.iterrows():
        cap = " (C)" if element == squad.captain else ""
        notes = []
        if row["status"] != "a":
            notes.append(f"{row['status']}, {row['chance_of_playing']:.0f} %")
        if row["n_fixtures"] != 1:
            notes.append(
                "ne joue pas" if row["n_fixtures"] == 0 else f"{row['n_fixtures']:.0f} matchs"
            )
        flag = f"  [{'; '.join(notes)}]" if notes else ""
        print(
            f"  {'banc' if row['bench'] else 'XI':4} {row['position']:3} {row['name'] + cap:24}"
            f" {row['team']:15} {row['price']:5.1f} M£  {row['xp']:4.1f} pts{flag}"
        )
    print(f"Points attendus (XI + capitaine) : {squad.expected_points:.1f}")


def _print_transfers(plan, estimated: bool) -> None:
    team, best, hold = plan.team, plan.best, plan.hold
    print(
        f"Ton équipe (effectif de la J{team.squad_gw}) : banque {team.bank:.1f} M£, "
        f"{team.free_transfers} transfert(s) gratuit(s) pour la J{plan.gw}"
        + (" (estimation, corrige avec --free-transfers)" if estimated else "")
    )

    def ranked(elements):
        rows = plan.pool.loc[elements].assign(_order=lambda d: d["position"].map(POSITION_ORDER))
        return rows.sort_values(["_order", "xp"], ascending=[True, False])

    if not plan.bought:
        print("Aucun transfert conseillé.")
    else:
        # Même nombre de départs et d'arrivées par poste : on les apparie poste par poste.
        for (_, out), (_, in_) in zip(
            ranked(plan.sold).iterrows(), ranked(plan.bought).iterrows(), strict=True
        ):
            print(
                f"  {out['position']:3} {out['name']:16} {out['price']:4.1f} M£ {out['xp']:4.1f}"
                f" pts  ->  {in_['name']:16} {in_['price']:4.1f} M£ {in_['xp']:4.1f} pts"
            )
        if best.hit_points:
            print(f"Pénalité : -{best.hit_points:.0f} pts")
        net = best.expected_points - best.hit_points
        print(
            f"Points attendus (XI + capitaine) : {net:.1f} avec ces transferts, pénalité "
            f"déduite, contre {hold.expected_points:.1f} sans transfert"
        )
        if abs(best.bench_points - hold.bench_points) > 0.05:
            print(
                f"Banc : {best.bench_points:.1f} pts contre {hold.bench_points:.1f}, "
                f"compté à {optimize.BENCH_WEIGHT * 100:.0f} % dans le choix"
            )
    print(
        "Prix de revente estimés : l'API publique ne donne pas le prix d'achat exact "
        "des joueurs pris à la création de l'équipe."
    )
    print()
    _print_squad(f"Effectif pour la J{plan.gw}", best)


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
    p_tr = sub.add_parser("transfers", help="Transferts conseillés depuis ton équipe FPL")
    p_tr.add_argument("--entry", type=int, required=True, help="ID d'équipe FPL (dans son URL)")
    p_tr.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p_tr.add_argument(
        "--free-transfers",
        type=int,
        choices=range(transfers.MAX_FREE_TRANSFERS + 1),
        metavar=f"0-{transfers.MAX_FREE_TRANSFERS}",
        help="Corrige l'estimation des transferts gratuits",
    )
    p_tr.add_argument(
        "--max-transfers",
        type=int,
        choices=range(16),
        metavar="0-15",
        help="Nombre maximum de transferts",
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

    if args.cmd in ("recommend", "transfers"):
        try:
            history = pd.read_parquet(args.data)
            if args.cmd == "recommend":
                next_gw, squad = recommend.recommend(history, budget=args.budget)
            else:
                plan = recommend.plan_transfers(
                    history,
                    args.entry,
                    free_transfers=args.free_transfers,
                    max_transfers=args.max_transfers,
                )
        except FileNotFoundError:
            parser.exit(1, f"Pas de données dans {args.data} : lance `fpl pull`.\n")
        except ModuleNotFoundError as exc:
            parser.exit(1, f"{exc} : installe l'extra ML (pip install -e '.[ml]').\n")
        except requests.RequestException as exc:
            parser.exit(1, f"API FPL injoignable : {exc}\n")
        except ValueError as exc:
            parser.exit(1, f"{exc}\n")
        if args.cmd == "recommend":
            _print_squad(f"Équipe recommandée pour la J{next_gw}", squad)
        else:
            _print_transfers(plan, estimated=args.free_transfers is None)
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
