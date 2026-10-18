import base64
import hashlib
import json
from urllib.parse import parse_qs, urlparse

import pytest

from fpl_assistant import auth
from fpl_assistant.api import FPLClient

REDIRECT = auth.REDIRECT_URI


def test_login_url_uses_pkce_and_the_site_client():
    login = auth.start_login()
    query = parse_qs(urlparse(login.url).query)
    assert login.url.startswith("https://account.premierleague.com/as/authorize?")
    assert query["client_id"] == [auth.CLIENT_ID] and query["state"] == [login.state]
    assert "offline_access" in query["scope"][0]
    digest = hashlib.sha256(login.verifier.encode()).digest()
    assert query["code_challenge"] == [base64.urlsafe_b64encode(digest).rstrip(b"=").decode()]


def test_finish_login_checks_state_and_sends_the_verifier(monkeypatch):
    login = auth.start_login()
    sent = {}
    monkeypatch.setattr(auth, "_token", lambda data: sent.update(data) or {"ok": True})
    url = f"https://fantasy.premierleague.com/?code=abc&state={login.state}"
    assert auth.finish_login(login, f"  {url}\n") == {"ok": True}
    assert sent["code"] == "abc" and sent["code_verifier"] == login.verifier
    with pytest.raises(RuntimeError, match="invalide"):
        auth.finish_login(login, "https://fantasy.premierleague.com/?code=abc&state=autre")
    with pytest.raises(RuntimeError, match="access_denied"):
        auth.finish_login(login, f"{REDIRECT}?error=access_denied&state={login.state}")


def test_access_token_keeps_a_rotated_refresh_token(tmp_path, monkeypatch):
    path = tmp_path / "token.json"
    monkeypatch.setenv("FPL_REFRESH_TOKEN", "old")
    monkeypatch.setattr(auth, "refresh", lambda t: {"access_token": "a", "refresh_token": "new"})
    assert auth.access_token(path) == "a"
    assert json.loads(path.read_text()) == {"refresh_token": "new"}
    assert auth.load_refresh_token(path) == "new"  # Le fichier prime sur l'environnement.


def test_access_token_writes_nothing_without_rotation(tmp_path, monkeypatch):
    path = tmp_path / "token.json"
    monkeypatch.setenv("FPL_REFRESH_TOKEN", "same")
    monkeypatch.setattr(auth, "refresh", lambda t: {"access_token": "a", "refresh_token": "same"})
    assert auth.access_token(path) == "a" and not path.exists()


def test_not_logged_in(tmp_path, monkeypatch):
    monkeypatch.delenv("FPL_REFRESH_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="fpl login"):
        auth.load_refresh_token(tmp_path / "absent.json")


class FakeSession:
    def __init__(self):
        self.headers, self.cookies, self.calls = {}, {}, []

    def get(self, url, timeout):
        self.calls.append(("GET", url))
        self.cookies["csrftoken"] = "tok"  # Posé par l'API au premier passage.

    def post(self, url, json, headers, timeout):
        self.calls.append(("POST", url, json, headers))
        return type("Resp", (), {"status_code": 200, "content": b"", "json": dict})()


def test_client_sends_the_token_and_csrf_headers():
    session = FakeSession()
    client = FPLClient(session=session, pause=0, access_token="xyz")
    assert session.headers["X-API-Authorization"] == "Bearer xyz"
    client.set_lineup(42, [{"element": 1, "position": 1}])
    method, url, body, headers = session.calls[-1]
    assert url.endswith("/api/my-team/42/") and body["picks"][0]["element"] == 1
    assert headers["X-CSRFToken"] == "tok" and headers["Origin"].startswith("https://fantasy")


def test_saved_token_is_private(tmp_path):
    path = tmp_path / "dir" / "token.json"
    auth.save_refresh_token("t1", path)
    auth.save_refresh_token("t2", path)
    assert auth.load_refresh_token(path) == "t2"
    assert path.stat().st_mode & 0o777 == 0o600
    assert list(path.parent.iterdir()) == [path]  # Pas de fichier temporaire laissé.
