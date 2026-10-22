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

GNU_DATE = subprocess.run(["date", "-d", "@0"], capture_output=True).returncode == 0
pytestmark = pytest.mark.skipif(
    not (shutil.which("jq") and GNU_DATE), reason="jq et date GNU requis"
)


def run_window(tmp_path, minutes_left, titles=(), force=False, next_gw=True, **env):
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
        "WAITING": "false",
        **env,
    }
    result = subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True)
    out = dict(line.split("=", 1) for line in output.read_text().splitlines())
    if "wake" in out:
        out["wait"] = (int(out.pop("wake")) - int(NOW.timestamp())) // 60
    return result, out


def window(tmp_path, minutes_left, **kwargs):
    result, out = run_window(tmp_path, minutes_left, **kwargs)
    assert result.returncode == 0, result.stderr
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


def test_only_one_run_waits_for_a_deadline(tmp_path):
    assert window(tmp_path, 200, WAITING="true") == {}


def test_after_the_wait_the_team_is_sent_now(tmp_path):
    out = window(tmp_path, 58, EXPECT_GW="8", WAITING="true")
    assert (out["run"], out["title"], out["wait"]) == ("true", TITLE, 0)
    # Deadline reculée de moins de 6 h 30 : mieux vaut régler maintenant qu'attendre.
    assert window(tmp_path, 200, EXPECT_GW="8")["wait"] == 0


def test_after_the_wait_a_team_set_meanwhile_is_left_alone(tmp_path):
    assert window(tmp_path, 58, titles=[TITLE], EXPECT_GW="8") == {}
    # Même une fois la deadline passée, quand la J8 n'est plus la prochaine journée.
    j7 = "2026-27 J7 : équipe réglée par le pilote automatique"
    assert window(tmp_path, 7 * 24 * 60, titles=[j7], EXPECT_GW="7") == {}


def test_after_the_wait_a_deadline_moved_far_away_is_left_to_a_later_run(tmp_path):
    assert window(tmp_path, 1440, EXPECT_GW="8") == {}


@pytest.mark.parametrize(("left", "expect"), [(10, "8"), (58, "7")])
def test_after_the_wait_too_late_is_an_error_for_the_alert(tmp_path, left, expect):
    result, out = run_window(tmp_path, left, EXPECT_GW=expect)
    assert result.returncode == 1 and out == {}
    assert "Trop tard" in result.stdout


@pytest.mark.parametrize("waiting", [True, False])
def test_waiting_runs_are_found_through_the_github_api(tmp_path, waiting):
    # Faux `gh` : le passage 1 est celui-ci ; le passage 2 attend, ou non, la deadline.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "gh"
    found = "wait" if waiting else ""
    fake.write_text(
        "#!/usr/bin/env bash\n"
        'case "$1 $2" in\n'
        '  "run list") printf "1\\n2\\n" ;;\n'
        f'  "run view") [ "$3" = 2 ] && printf %s "{found}" ;;\n'
        "esac\n"
    )
    fake.chmod(0o755)
    env = {"PATH": f"{bin_dir}:{os.environ['PATH']}", "GITHUB_RUN_ID": "1", "WAITING": ""}
    out = window(tmp_path, 200, **env)
    assert out == {} if waiting else out["wait"] == 140
