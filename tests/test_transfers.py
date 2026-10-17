import pandas as pd
import pytest
import requests

from fpl_assistant.transfers import current_team, free_transfers, selling_price


def history(transfers, chips=()):
    return {
        "current": [{"event": gw, "event_transfers": n} for gw, n in transfers.items()],
        "chips": [{"event": gw, "name": name} for gw, name in chips],
    }


def test_free_transfers_bank_up_to_five_and_survive_chips():
    # Comme l'équipe publique n°1 : bboost J2, wildcard J3, free hit J5, un transfert en J6.
    h = history({1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 1, 7: 0}, [(2, "bboost"), (3, "wildcard"),
                                                            (5, "freehit")])  # fmt: skip
    assert free_transfers(h, started=1, gw=7) == 4
    assert free_transfers(history({gw: 0 for gw in range(1, 11)}), started=1, gw=10) == 5
    assert free_transfers(history({1: 0, 2: 0, 3: 3}), started=1, gw=3) == 1  # Avec -4.
    assert free_transfers(history({8: 0}), started=8, gw=8) == 1


@pytest.mark.parametrize(
    ("bought", "now", "sell"), [(75, 78, 7.6), (75, 79, 7.7), (75, 72, 7.2), (75, 75, 7.5)]
)
def test_selling_price_keeps_half_of_the_rise(bought, now, sell):
    assert selling_price(bought, now) == pytest.approx(sell)


class FakeClient:
    """Équipe créée en J1, transfert en J2, free hit en J3 (journée en cours)."""

    def __init__(self, started=1):
        self.started = started

    def entry(self, entry):
        if entry == 404:
            resp = requests.Response()
            resp.status_code = 404
            raise requests.HTTPError("404 Client Error", response=resp)
        return {"started_event": self.started}

    def entry_history(self, entry):
        return history({1: 0, 2: 1, 3: 0}, [(3, "freehit")])

    def entry_picks(self, entry, gw):
        squads = {2: [1, 2, 3], 3: [4, 5, 6]}
        return {"picks": [{"element": e} for e in squads[gw]], "entry_history": {"bank": 12 + gw}}

    def entry_transfers(self, entry):
        return [
            {"element_in": 2, "element_in_cost": 50, "event": 2, "time": "2026-08-28T10:00Z"},
            # Transfert du free hit, annulé ensuite : le prix d'achat de 1 reste celui de J1.
            {"element_in": 1, "element_in_cost": 90, "event": 3, "time": "2026-09-04T10:00Z"},
        ]


BOOTSTRAP = {
    "events": [
        {"id": 3, "is_current": True, "deadline_time": "2026-08-15T17:30:00Z"},
        {"id": 4, "is_next": True, "deadline_time": "2026-08-22T17:30:00Z"},
    ],
    "elements": [{"id": e, "now_cost": 64} for e in range(1, 7)],
}
# Historique par match : 6,0 M£ en J1, 6,1 en J2 (et une ligne de la saison passée).
MATCHES = pd.DataFrame(
    {"season": ["2026-27"] * 4 + ["2025-26"], "GW": [2, 1, 2, 1, 1],
     "element": [1, 1, 3, 3, 3], "price": [6.1, 6.0, 6.1, 6.0, 9.9]}
)  # fmt: skip


def test_current_team_reverts_free_hit_and_estimates_selling_prices():
    team = current_team(FakeClient(), 1, BOOTSTRAP, MATCHES)
    assert team.gw == 3 and team.squad_gw == 2  # L'effectif d'avant le free hit.
    assert team.elements == [1, 2, 3]
    assert team.bank == pytest.approx(1.4)
    # 1 et 3 : achetés 6,0 en J1 ; 2 : acheté 5,0 en J2. Tous valent 6,4 aujourd'hui.
    assert team.selling == pytest.approx({1: 6.2, 2: 5.7, 3: 6.2})
    assert team.budget == pytest.approx(1.4 + 6.2 + 5.7 + 6.2)
    assert team.free_transfers == 1  # Le free hit garde celui de la J3.


def test_team_not_started_yet_or_unknown():
    with pytest.raises(ValueError, match="démarre en J4"):
        current_team(FakeClient(started=4), 1, BOOTSTRAP, MATCHES)
    with pytest.raises(ValueError, match="introuvable"):
        current_team(FakeClient(), 404, BOOTSTRAP, MATCHES)
