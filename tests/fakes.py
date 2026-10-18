"""Ligue factice partagée par les tests : 90 joueurs, une journée jouée, des xp fixes."""

import pandas as pd


def league(n_teams=10):
    """Bootstrap de 90 joueurs, en J1 terminée, et leurs prédictions pour la J2."""
    sizes = {1: 10, 2: 30, 3: 30, 4: 20}
    elements = [
        {"id": i, "code": i, "web_name": f"P{i}", "element_type": t, "team": 1 + i % n_teams,
         "now_cost": 40 + i * 7 % 90, "status": "a", "chance_of_playing_next_round": None}
        for i, t in enumerate((t for t, n in sizes.items() for _ in range(n)), start=1)
    ]  # fmt: skip
    bootstrap = {
        "teams": [{"id": k, "name": f"T{k}"} for k in range(1, n_teams + 1)],
        "events": [
            {"id": 1, "is_current": True, "finished": True, "data_checked": True,
             "deadline_time": "2026-08-14T17:30:00Z"},
            {"id": 2, "is_next": True, "finished": False, "data_checked": False,
             "deadline_time": "2026-08-21T17:30:00Z"},
        ],
        "elements": elements,
    }  # fmt: skip
    preds = pd.DataFrame(
        {"element": [e["id"] for e in elements],
         "xp": [1.0 + e["id"] * 13 % 70 / 10 for e in elements]}
    ).assign(raw_xp=lambda d: d["xp"], n_fixtures=1)  # fmt: skip
    return bootstrap, preds


class LeagueClient:
    def __init__(self, bootstrap, squad, transfer_info=None):
        self._bootstrap, self.squad = bootstrap, squad
        # Prix de revente : 0,3 M£ sous le prix du jour, comme après une hausse de 0,5.
        self.selling = {e["id"]: e["now_cost"] - 3 for e in bootstrap["elements"]}
        self.transfer_info = transfer_info or {"status": "cost", "limit": 1, "made": 0, "bank": 5}
        self.sent = []

    def bootstrap(self):
        return self._bootstrap

    def entry(self, entry):
        return {"started_event": 1}

    def entry_history(self, entry):
        return {"current": [{"event": 1, "event_transfers": 0}], "chips": []}

    def entry_picks(self, entry, gw):
        return {"picks": [{"element": e} for e in self.squad], "entry_history": {"bank": 5}}

    def entry_transfers(self, entry):
        return []

    # Connecté : équipe exacte, et les envois sont enregistrés au lieu d'être faits.
    def my_team(self, entry):
        return {
            "picks": [{"element": e, "selling_price": self.selling.get(e, 0)} for e in self.squad],
            "transfers": self.transfer_info,
        }

    def make_transfers(self, entry, gw, transfers, chip=None):
        self.sent.append(("transfers", gw, transfers))

    def set_lineup(self, entry, picks, chip=None):
        self.sent.append(("lineup", picks))
