"""Script de .github/actions/fpl-token : le jeton que la FPL change à chaque usage passe d'un
passage de l'Action au suivant, chiffré avec le secret."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / ".github" / "actions" / "fpl-token" / "token.sh"

pytestmark = pytest.mark.skipif(
    not (shutil.which("openssl") and shutil.which("jq")), reason="openssl et jq requis"
)


class Job:
    """Un passage de l'Action : dossier temporaire neuf, comme `runner.temp`."""

    def __init__(self, root: Path, name: str, secret: str):
        self.temp = root / name
        self.temp.mkdir()
        self.token_file = root / f"{name}-token.json"
        self.secret = secret

    def run(self, step: str) -> subprocess.CompletedProcess:
        output = self.temp / "output"
        env = {
            **os.environ,
            "FPL_REFRESH_TOKEN": self.secret,
            "FPL_TOKEN_FILE": str(self.token_file),
            "RUNNER_TEMP": str(self.temp),
            "GITHUB_OUTPUT": str(output),
        }
        output.write_text("")
        result = subprocess.run(
            ["bash", str(SCRIPT), step], env=env, capture_output=True, text=True
        )
        result.outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
        return result

    def token(self) -> str:
        return json.loads(self.token_file.read_text())["refresh_token"]

    def rotate(self, new: str) -> None:
        """Ce que fait `fpl` quand la FPL donne un nouveau jeton."""
        self.token_file.write_text(json.dumps({"refresh_token": new}))

    @property
    def sealed(self) -> Path:
        return self.temp / "fpl-token.enc"


def test_first_run_starts_from_the_secret_without_spaces(tmp_path):
    job = Job(tmp_path, "un", "eyJ.seed\n")
    assert job.run("open").returncode == 0
    assert job.token() == "eyJ.seed"
    # Sans nouveau jeton, rien à garder : le cache reste au dernier jeton valable.
    assert job.run("seal").outputs == {} and not job.sealed.exists()


def test_the_new_token_reaches_the_next_run(tmp_path):
    first = Job(tmp_path, "un", "seed")
    first.run("open")
    first.rotate("R1")
    assert first.run("seal").outputs == {"sealed": "true"}
    assert b"R1" not in first.sealed.read_bytes()

    second = Job(tmp_path, "deux", "seed")
    shutil.copy(first.sealed, second.sealed)  # Restauré depuis le cache.
    second.run("open")
    assert second.token() == "R1"


def test_a_new_secret_gets_its_own_cache_key(tmp_path):
    def key(name: str, secret: str) -> str:
        return Job(tmp_path, name, secret).run("key").outputs["key"]

    assert key("un", "seed") == key("deux", "seed")
    assert key("trois", "seed").startswith("fpl-token-")
    assert key("quatre", "seed") != key("cinq", "autre")


def test_empty_secret_says_what_to_do(tmp_path):
    result = Job(tmp_path, "un", "").run("open")
    assert result.returncode == 1 and "FPL_REFRESH_TOKEN vide" in result.stdout


def test_a_cache_that_does_not_decrypt_leaves_no_token(tmp_path):
    first = Job(tmp_path, "un", "seed")
    first.run("open")
    first.rotate("R1")
    first.run("seal")
    other = Job(tmp_path, "deux", "autre")
    shutil.copy(first.sealed, other.sealed)
    assert other.run("open").returncode != 0
    assert not other.token_file.exists()
