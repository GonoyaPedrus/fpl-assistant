#!/usr/bin/env bash
# Ce passage règle-t-il l'équipe ? Écrit run, gw, season, title, wake et check dans
# $GITHUB_OUTPUT, ou rien si ce n'est pas le moment. GitHub ne lance le cron que toutes les
# 3 à 6 heures : un passage des WINDOW minutes avant la deadline attend jusqu'à LEAD minutes
# avant (wake), et rien ne part à moins de FLOOR minutes. FORCE=true : tout de suite.
# BOOTSTRAP_FILE et TITLES_FILE remplacent les téléchargements (tests).
set -euo pipefail
WINDOW=390 LEAD=60 FLOOR=15

bootstrap="${BOOTSTRAP_FILE:-$RUNNER_TEMP/bootstrap.json}"
titles="${TITLES_FILE:-$RUNNER_TEMP/titles.json}"
if [ -z "${BOOTSTRAP_FILE:-}" ]; then
  curl -fsS --max-time 60 --retry 3 --retry-all-errors -A fpl-assistant \
    https://fantasy.premierleague.com/api/bootstrap-static/ > "$bootstrap"
fi
if [ -z "${TITLES_FILE:-}" ]; then
  # Liste par label plutôt que par recherche : l'index de recherche a du retard.
  gh issue list --label autopilot --state all --limit 200 --json title > "$titles"
fi

now="${NOW:-$(date +%s)}"
next="$(jq -c '[.events[] | select(.is_next)][0] // empty' "$bootstrap")"
if [ -z "$next" ]; then
  echo "::notice::Pas de prochaine journée dans le calendrier FPL."
  exit 0
fi
# La saison (« 2026-27 ») distingue la J1 d'une saison de celle de la suivante.
year="$(jq -r '.events[0].deadline_time[:4]' "$bootstrap")"
season="$year-$(printf %02d $(( (year + 1) % 100 )))"
gw="$(jq -r .id <<<"$next")"
deadline="$(date -d "$(jq -r .deadline_time <<<"$next")" +%s)"
left=$(( (deadline - now) / 60 ))
title="$season J$gw : équipe réglée par le pilote automatique"
done="$(jq --arg t "$title" '[.[] | select(.title == $t)] | length' "$titles")"
echo "$season J$gw : deadline dans $left min, déjà réglée : $done"

wake="$now"
if [ "${FORCE:-false}" = "true" ]; then
  title="$season J$gw : réglage manuel" # Laisse la place au passage avant la deadline.
elif [ "$done" != "0" ] || [ "$left" -gt "$WINDOW" ] || [ "$left" -lt "$FLOOR" ]; then
  exit 0
elif [ "$left" -gt "$LEAD" ]; then
  wake=$(( deadline - LEAD * 60 ))
fi
{
  echo "run=true"
  echo "gw=$gw"
  echo "season=$season"
  echo "title=$title"
  echo "wake=$wake"
  # Une longue attente : vérifier d'abord la connexion, pour qu'un jeton annulé se voie
  # des heures avant la deadline.
  echo "check=$([ $(( wake - now )) -gt 900 ] && echo true || echo false)"
} >> "$GITHUB_OUTPUT"
