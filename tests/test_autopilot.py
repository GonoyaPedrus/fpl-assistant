import pandas as pd
import pytest
from fakes import LeagueClient, league

from fpl_assistant import autopilot, recommend
from fpl_assistant.optimize import pick_squad
from fpl_assistant.recommend import candidate_pool

MATCHES = pd.DataFrame({"season": ["2026-27"], "GW": [1], "element": [1], "price": [5.0]})


def setup(monkeypatch, transfer_info=None):
    bootstrap, preds = league()
    squad = list(pick_squad(candidate_pool(bootstrap, preds)).squad.index)
    preds["xp"] = preds["xp"].to_numpy()[::-1]  # Les xp ont changé : des transferts valent.
    monkeypatch.setattr(recommend, "_next_predictions", lambda *_: (2, preds))
    return bootstrap, LeagueClient(bootstrap, squad, transfer_info)


def test_simulation_sends_nothing(monkeypatch):
    _, client = setup(monkeypatch)
    result = autopilot.run(MATCHES, 1, client)
    assert not result.applied and client.sent == []
    assert "simulation" in result.markdown()


def test_apply_sends_transfers_then_lineup(monkeypatch):
    bootstrap, client = setup(monkeypatch)
    result = autopilot.run(MATCHES, 1, client, apply=True)
    assert result.applied and [call[0] for call in client.sent] == ["transfers", "lineup"]
    _, gw, transfers, chip = client.sent[0]
    assert chip is None
    assert gw == 2 and 0 < len(transfers) <= 2  # 1 gratuit + 1 payant au plus.
    elements = {e["id"]: e for e in bootstrap["elements"]}
    for t in transfers:
        assert t["element_out"] in client.squad and t["element_in"] not in client.squad
        assert (
            elements[t["element_in"]]["element_type"] == elements[t["element_out"]]["element_type"]
        )
        assert t["purchase_price"] == elements[t["element_in"]]["now_cost"]
        assert t["selling_price"] == client.selling[t["element_out"]]
    picks = client.sent[1][1]
    assert [p["position"] for p in picks] == list(range(1, 16))
    assert sum(p["is_captain"] for p in picks) == sum(p["is_vice_captain"] for p in picks) == 1
    new_squad = set(client.squad) - {t["element_out"] for t in transfers}
    assert {p["element"] for p in picks} == new_squad | {t["element_in"] for t in transfers}


def test_hits_already_taken_count_against_the_cap(monkeypatch):
    # Un passage précédent a déjà fait 2 transferts pour 1 gratuit : le -4 du jour est pris.
    info = {"status": "cost", "limit": 1, "made": 2, "bank": 5}
    _, client = setup(monkeypatch, info)
    assert autopilot.run(MATCHES, 1, client, apply=True).transfers == []


def test_chip_activated_by_hand_is_sent_back(monkeypatch):
    _, client = setup(monkeypatch)
    client.chips = [
        {"name": "wildcard", "status_for_entry": "available"},
        {"name": "bboost", "status_for_entry": "active"},
    ]
    result = autopilot.run(MATCHES, 1, client, apply=True)
    assert {call[-1] for call in client.sent} == {"bboost"}  # Un envoi sans chip l'annulerait.
    assert "bboost" in result.markdown()


def test_nothing_is_sent_once_the_deadline_has_passed(monkeypatch):
    _, client = setup(monkeypatch)  # La prochaine journée est la J2.
    with pytest.raises(ValueError, match="J1 est passée"):
        autopilot.run(MATCHES, 1, client, apply=True, gw=1)
    assert client.sent == []
