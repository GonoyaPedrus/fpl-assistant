import pytest
import requests

from fpl_assistant import ingest


class FakeResponse:
    def __init__(self, status_code: int, body: bytes = b"") -> None:
        self.status_code = status_code
        self.body = body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size: int):
        yield self.body


def test_seasons_before():
    assert ingest.seasons_before("2022-23") == ("2020-21", "2021-22")
    assert ingest.seasons_before("2020-21") == ()


def test_pull_writes_file(tmp_path, monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(200, b"PAR1"))
    out = ingest.pull(tmp_path / "data" / "gw.parquet")
    assert out.read_bytes() == b"PAR1"
    assert not out.with_suffix(".part").exists()


def test_pull_explains_missing_release(tmp_path, monkeypatch):
    monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse(404))
    with pytest.raises(RuntimeError, match="fpl ingest"):
        ingest.pull(tmp_path / "gw.parquet")


def test_strict_ingest_fails_instead_of_dropping_a_season(tmp_path, monkeypatch):
    events = [{"id": 1, "deadline_time": "2021-08-13T17:30:00Z", "finished": True,
               "data_checked": True}]  # fmt: skip

    class FakeClient:
        def bootstrap(self):
            return {"events": events}

    def broken(*args, **kwargs):
        raise requests.ConnectionError("API indisponible")

    monkeypatch.setattr(ingest.api, "FPLClient", FakeClient)
    monkeypatch.setattr(ingest.api, "season_history", broken)
    monkeypatch.setattr(ingest, "fetch_season", broken)
    with pytest.raises(requests.ConnectionError):
        ingest.ingest(tmp_path / "gw.parquet", strict=True)
    with pytest.raises(RuntimeError, match="Aucune saison"):
        ingest.ingest(tmp_path / "gw.parquet")  # Non strict : tout est ignoré, rien à écrire.
