""".github/scripts/window.sh : quand le pilote règle l'équipe, alors que GitHub ne lance son
cron que toutes les 3 à 6 heures."""

import json
import os
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / ".github" / "scripts" / "window.sh"
NOW = datetime(2026, 10, 23, 12, 0, tzinfo=UTC)
TITLE = "2026-27 J8 : équipe réglée par le pilote automatique"

pytestmark = pytest.mark.skipif(not shutil.which("jq"), reason="jq requis")


def window(tmp_path, minutes_left, titles=(), force=False, next_gw=True):
    deadline = NOW + timedelta(minutes=minutes_left)
    events = [
        {"id": 1, "deadline_time": "2026-08-14T17:30:00Z", "is_next": False},
        {"id": 8, "deadline_time": deadline.strftime("%Y-%m-%dT%H:%M:%SZ"), "is_next": next_gw},
    ]
    (tmp_path / "bootstrap.json").write_text(json.dumps({"events": events}))
    (tmp_path / "titles.json").write_text(json.dumps([{"title": t} for t in titles]))
    output = tmp_path / "output"
    output.write_text("")
    env = {
        **os.environ,
        "BOOTSTRAP_FILE": str(tmp_path / "bootstrap.json"),
        "TITLES_FILE": str(tmp_path / "titles.json"),
        "GITHUB_OUTPUT": str(output),
        "NOW": str(int(NOW.timestamp())),
        "FORCE": "true" if force else "false",
    }
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    out = dict(line.split("=", 1) for line in output.read_text().splitlines())
    if "wake" in out:
        out["wait"] = (int(out.pop("wake")) - int(NOW.timestamp())) // 60
    return out


@pytest.mark.parametrize("left", [391, 14, -30])
def test_too_early_or_too_late_does_nothing(tmp_path, left):
    assert window(tmp_path, left) == {}


def test_the_first_run_of_the_window_waits_until_one_hour_before(tmp_path):
    # 6 h 30 couvre l'écart le plus long observé entre deux passages (5 h 40).
    assert window(tmp_path, 390) == {
        "run": "true",
        "gw": "8",
        "season": "2026-27",
        "title": TITLE,
        "wait": 330,
        "check": "true",
    }


@pytest.mark.parametrize(
    ("left", "wait", "check"),
    [(76, 16, "true"), (70, 10, "false"), (60, 0, "false"), (15, 0, "false")],
)
def test_a_short_wait_skips_the_connection_check(tmp_path, left, wait, check):
    out = window(tmp_path, left)
    assert (out["run"], out["wait"], out["check"]) == ("true", wait, check)


def test_a_team_already_set_is_not_sent_again(tmp_path):
    assert window(tmp_path, 120, titles=[TITLE, "2026-27 J7 : autre"]) == {}


def test_force_sets_the_team_now_under_its_own_title(tmp_path):
    out = window(tmp_path, 3000, titles=[TITLE], force=True)
    assert out["title"] == "2026-27 J8 : réglage manuel"
    assert (out["run"], out["wait"], out["check"]) == ("true", 0, "false")


def test_no_next_gameweek_does_nothing(tmp_path):
    assert window(tmp_path, 120, next_gw=False) == {}
