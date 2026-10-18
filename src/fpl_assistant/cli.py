"""Ligne de commande : `fpl ingest`, `pull`, `latest`, `baseline`, `train`, `recommend`,
`transfers`, `login`, `token`, `autopilot`."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import requests

from fpl_assistant import (
    DEFAULT_DATA,
    api,
    auth,
    autopilot,
    evaluate,
    features,
    ingest,
    model,
    recommend,
    report,
    transfers,
)


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


def _login() -> None:
    login = auth.start_login()
    print("1. Ouvre cette adresse et connecte-toi à ton compte FPL :\n")
    print(f"   {login.url}\n")
    print("2. Tu arrives sur fantasy.premierleague.com. Copie tout de suite l'adresse complète")
    print("   de la page (elle contient « code= », valable une minute) et colle-la ici.\n")
    tokens = auth.finish_login(login, input("Adresse : "))
    # Un premier rafraîchissement dit si le serveur change le jeton à chaque usage.
    renewed = auth.refresh(tokens["refresh_token"])
    token = renewed.get("refresh_token", tokens["refresh_token"])
    auth.save_refresh_token(token)
    rotates = token != tokens["refresh_token"]
    entry = api.FPLClient(access_token=renewed["access_token"]).me()["player"]["entry"]
    print(f"\nConnecté : équipe {entry}. Jeton enregistré dans {auth.TOKEN_FILE}.")
    print("Pour la GitHub Action :")
    print('  gh secret set FPL_REFRESH_TOKEN --body "$(fpl token)"')
    print(f"  gh variable set FPL_ENTRY --body {entry}")
    if rotates:
        print(
            "Le jeton change à chaque usage : ajoute aussi un jeton GitHub (secret "
            "SECRETS_PAT, droit « Secrets : read and write » sur le dépôt) pour que l'Action "
            "enregistre le nouveau après chaque passage."
        )


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

    sub.add_parser("login", help="Connecte le dépôt à ton compte FPL (une fois)")
    sub.add_parser("token", help="Affiche le jeton de connexion, pour le secret GitHub")
    p_auto = sub.add_parser(
        "autopilot", help="Règle ton équipe pour la prochaine journée (connexion requise)"
    )
    p_auto.add_argument("--entry", type=int, help="ID d'équipe FPL (défaut : celle du compte)")
    p_auto.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p_auto.add_argument("--apply", action="store_true", help="Envoie vraiment à la FPL")
    p_auto.add_argument(
        "--max-hits",
        type=int,
        choices=range(16),
        metavar="0-15",
        default=autopilot.MAX_HITS,
        help=f"Transferts payants (-4) au plus (défaut {autopilot.MAX_HITS})",
    )
    p_auto.add_argument("--report", type=Path, help="Écrit le compte rendu en Markdown")

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

    if args.cmd == "login":
        _login()
        return
    if args.cmd == "token":
        try:
            print(auth.load_refresh_token())
        except RuntimeError as exc:
            parser.exit(1, f"{exc}\n")
        return

    if args.cmd in ("recommend", "transfers", "autopilot"):
        try:
            history = pd.read_parquet(args.data)
            if args.cmd == "recommend":
                next_gw, squad = recommend.recommend(history, budget=args.budget)
            elif args.cmd == "transfers":
                plan = recommend.plan_transfers(
                    history,
                    args.entry,
                    free_transfers=args.free_transfers,
                    max_transfers=args.max_transfers,
                )
            else:
                client = api.FPLClient(access_token=auth.access_token())
                entry = args.entry or client.me()["player"]["entry"]
                result = autopilot.run(
                    history, entry, client, apply=args.apply, max_hits=args.max_hits
                )
        except FileNotFoundError:
            parser.exit(1, f"Pas de données dans {args.data} : lance `fpl pull`.\n")
        except ModuleNotFoundError as exc:
            parser.exit(1, f"{exc} : installe l'extra ML (pip install -e '.[ml]').\n")
        except requests.HTTPError as exc:
            parser.exit(1, f"Refus de l'API FPL : {exc}\n")
        except requests.RequestException as exc:
            parser.exit(1, f"API FPL injoignable : {exc}\n")
        except (ValueError, RuntimeError) as exc:
            parser.exit(1, f"{exc}\n")
        if args.cmd == "recommend":
            print("\n".join(report.squad_lines(f"Équipe recommandée pour la J{next_gw}", squad)))
        elif args.cmd == "transfers":
            lines = report.transfer_lines(
                plan, estimated_free=args.free_transfers is None, estimated_prices=True
            )
            print("\n".join(lines))
        else:
            print(result.markdown())
            if args.report:
                args.report.write_text(result.markdown())
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
