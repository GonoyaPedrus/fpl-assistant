"""Mise en texte des équipes et des transferts, pour la CLI et les issues du pilote."""

from __future__ import annotations

from fpl_assistant.optimize import BENCH_WEIGHT, Squad, lineup
from fpl_assistant.transfers import UNLIMITED


def squad_lines(title: str, squad: Squad) -> list[str]:
    lines = [f"{title} ({squad.cost:.1f} M£)"]
    for _, row in lineup(squad).iterrows():
        role = " (C)" if row["is_captain"] else " (V)" if row["is_vice_captain"] else ""
        notes = []
        if row["status"] != "a":
            notes.append(f"{row['status']}, {row['chance_of_playing']:.0f} %")
        if row["n_fixtures"] != 1:
            notes.append(
                "ne joue pas" if row["n_fixtures"] == 0 else f"{row['n_fixtures']:.0f} matchs"
            )
        flag = f"  [{'; '.join(notes)}]" if notes else ""
        lines.append(
            f"  {'banc' if row['bench'] else 'XI':4} {row['position']:3} {row['name'] + role:24}"
            f" {row['team']:15} {row['price']:5.1f} M£  {row['xp']:4.1f} pts{flag}"
        )
    lines.append(f"Points attendus (XI + capitaine) : {squad.expected_points:.1f}")
    return lines


def transfer_lines(plan, estimated_free: bool, estimated_prices: bool) -> list[str]:
    """`plan` : un `recommend.TransferPlan`. Sans connexion au compte, transferts gratuits et
    prix de revente sont estimés : `estimated_*` l'indiquent dans le texte."""
    team, best, hold = plan.team, plan.best, plan.hold
    free = "illimités" if team.free_transfers >= UNLIMITED else str(team.free_transfers)
    lines = [
        f"Ton équipe : banque {team.bank:.1f} M£, transferts gratuits pour la J{plan.gw} : {free}"
        + (" (estimation, corrige avec --free-transfers)" if estimated_free else "")
    ]
    if not plan.bought:
        lines.append("Aucun transfert conseillé.")
    for out, in_ in plan.pairs():
        o, i = plan.pool.loc[out], plan.pool.loc[in_]
        lines.append(
            f"  {o['position']:3} {o['name']:16} {o['price']:4.1f} M£ {o['xp']:4.1f}"
            f" pts  ->  {i['name']:16} {i['price']:4.1f} M£ {i['xp']:4.1f} pts"
        )
    if plan.bought:
        if best.hit_points:
            lines.append(f"Pénalité : -{best.hit_points:.0f} pts")
        net = best.expected_points - best.hit_points
        lines.append(
            f"Points attendus (XI + capitaine) : {net:.1f} avec ces transferts, pénalité "
            f"déduite, contre {hold.expected_points:.1f} sans transfert"
        )
        if abs(best.bench_points - hold.bench_points) > 0.05:
            lines.append(
                f"Banc : {best.bench_points:.1f} pts contre {hold.bench_points:.1f}, "
                f"compté à {BENCH_WEIGHT * 100:.0f} % dans le choix"
            )
    if estimated_prices:
        lines.append(
            "Prix de revente estimés : l'API publique ne donne pas le prix d'achat exact "
            "des joueurs pris à la création de l'équipe."
        )
    return [*lines, "", *squad_lines(f"Effectif pour la J{plan.gw}", best)]
