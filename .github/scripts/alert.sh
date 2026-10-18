#!/usr/bin/env bash
# Ouvre une issue d'alerte « $1 » (corps « $2 »), sauf si la même est déjà ouverte.
set -euo pipefail
title="$1"
gh label create autopilot --color 1D76DB --description "Pilote automatique FPL" --force
open="$(gh issue list --label autopilot --state open --limit 200 --json title \
  | jq --arg t "$title" '[.[] | select(.title == $t)] | length')"
[ "$open" = "0" ] || exit 0
gh issue create --title "$title" --label autopilot --body "$2"
