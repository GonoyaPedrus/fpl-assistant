#!/usr/bin/env bash
# Chiffre (`seal`) ou déchiffre (`open`) le dernier jeton FPL gardé dans le cache, avec le
# secret FPL_REFRESH_TOKEN pour clé. `key` donne la clé de cache propre à ce secret : après
# un nouveau `fpl login --print`, le pilote repart du secret au lieu d'un ancien cache.
set -euo pipefail
: "${FPL_TOKEN_FILE:?FPL_TOKEN_FILE doit indiquer le fichier de jeton lu par fpl}"
if [ -z "${FPL_REFRESH_TOKEN:-}" ]; then
  echo "::error::Secret FPL_REFRESH_TOKEN vide : voir « Pilote automatique » dans le README."
  exit 1
fi
enc="$RUNNER_TEMP/fpl-token.enc"
opened="$RUNNER_TEMP/fpl-token.opened" # Empreinte du jeton au départ du passage.
crypt() { openssl enc -aes-256-cbc -pbkdf2 -salt -pass env:FPL_REFRESH_TOKEN "$@"; }
umask 077
case "${1:-}" in
  key)
    echo "key=fpl-token-$(printf %s "$FPL_REFRESH_TOKEN" | sha256sum | cut -c1-16)" >> "$GITHUB_OUTPUT"
    ;;
  open)
    if [ -f "$enc" ]; then
      # Via un fichier temporaire : un déchiffrement raté ne laisse pas de jeton tronqué.
      crypt -d -in "$enc" -out "$FPL_TOKEN_FILE.tmp"
      mv "$FPL_TOKEN_FILE.tmp" "$FPL_TOKEN_FILE"
    else
      # Premier passage avec ce secret. Un jeton ne contient jamais d'espace : on retire
      # ceux d'un copier-coller (retour à la ligne, espace final).
      jq -n --arg t "$(printf %s "$FPL_REFRESH_TOKEN" | tr -d '[:space:]')" \
        '{refresh_token: $t}' > "$FPL_TOKEN_FILE"
    fi
    sha256sum < "$FPL_TOKEN_FILE" > "$opened"
    ;;
  seal)
    # `fpl` ne réécrit le fichier que si la FPL a donné un nouveau jeton : sinon rien à garder.
    if [ -f "$FPL_TOKEN_FILE" ] && ! sha256sum < "$FPL_TOKEN_FILE" | cmp -s - "$opened"; then
      crypt -in "$FPL_TOKEN_FILE" -out "$enc"
      echo "sealed=true" >> "$GITHUB_OUTPUT"
    fi
    ;;
  *)
    echo "Usage : $0 key|open|seal" >&2
    exit 2
    ;;
esac
