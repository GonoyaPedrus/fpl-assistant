"""Pilote automatique : règle l'équipe FPL avant la deadline (transferts, XI, capitaine).

Les chips (wildcard, free hit, triple capitaine, bench boost) ne sont jamais joués ici.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from fpl_assistant import api, recommend, report
from fpl_assistant.optimize import lineup

# Garde-fou : le modèle ne voit qu'une journée, un -8 sur sa seule foi coûte cher s'il se
# trompe. Au-delà, il faut le demander explicitement (`--max-hits`).
MAX_HITS = 1


@dataclass
class Run:
    plan: recommend.TransferPlan
    transfers: list[dict[str, Any]]  # Corps de `transfers/`, vide sans transfert.
    picks: list[dict[str, Any]]  # Corps de `my-team/{id}/`.
    applied: bool = False

    def markdown(self) -> str:
        done = "envoyés à la FPL" if self.applied else "simulation, rien n'a été envoyé"
        lines = report.transfer_lines(self.plan, estimated_free=False, estimated_prices=False)
        body = "\n".join(lines)
        return (
            f"Transferts, XI et capitaine pour la J{self.plan.gw} : {done}.\n\n```\n{body}\n```\n"
        )


def transfers_payload(plan: recommend.TransferPlan) -> list[dict[str, Any]]:
    """Prix en dixièmes de M£ : achat au prix du jour, revente au prix donné par `my-team`."""
    return [
        {
            "element_in": int(in_),
            "element_out": int(out),
            "purchase_price": round(plan.pool.loc[in_, "price"] * 10),
            "selling_price": round(plan.team.selling[out] * 10),
        }
        for out, in_ in plan.pairs()
    ]


def picks_payload(plan: recommend.TransferPlan) -> list[dict[str, Any]]:
    return [
        {
            "element": int(element),
            "position": int(row["slot"]),
            "is_captain": bool(row["is_captain"]),
            "is_vice_captain": bool(row["is_vice_captain"]),
        }
        for element, row in lineup(plan.best).iterrows()
    ]


def run(
    history: pd.DataFrame,
    entry: int,
    client: api.FPLClient,
    apply: bool = False,
    max_hits: int = MAX_HITS,
) -> Run:
    """Calcule l'équipe de la prochaine journée et, si `apply`, l'envoie à la FPL.

    `client` doit être connecté au compte (cf. `auth.access_token`).
    """
    plan = recommend.plan_transfers(history, entry, client, max_hits=max_hits, logged_in=True)
    result = Run(plan, transfers_payload(plan), picks_payload(plan))
    if apply:
        if result.transfers:
            client.make_transfers(entry, plan.gw, result.transfers)
        client.set_lineup(entry, result.picks)
        result.applied = True
    return result
