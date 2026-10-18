#!/usr/bin/env bash
# `fpl` n'écrit $FPL_TOKEN_FILE que si la FPL a remplacé le jeton : celui du secret ne vaut
# alors plus rien, et ce workflow ne réécrit pas ses secrets. On échoue pour alerter.
set -euo pipefail
if [ -f "$FPL_TOKEN_FILE" ]; then
  echo "::error::La FPL a remplacé le jeton de connexion : le secret FPL_REFRESH_TOKEN est périmé."
  exit 1
fi
