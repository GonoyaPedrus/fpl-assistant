#!/usr/bin/env bash
# Chiffre (`seal`) ou déchiffre (`open`) le dernier jeton FPL gardé dans le cache, avec le
# secret FPL_REFRESH_TOKEN pour clé. `key` donne la clé de cache propre à ce secret : après
# un nouveau `fpl login --print`, le pilote repart du secret au lieu d'un ancien cache.
set -euo pipefail
if [ -z "${FPL_REFRESH_TOKEN:-}" ]; then
  echo "::error::Secret FPL_REFRESH_TOKEN vide : lance \`fpl login --print\` et colle le jeton dans Settings > Secrets and variables > Actions."
  exit 1
fi
enc="$RUNNER_TEMP/fpl-token.enc"
crypt() { openssl enc -aes-256-cbc -pbkdf2 -salt -pass env:FPL_REFRESH_TOKEN "$@"; }
umask 077
case "$1" in
  key)
    echo "key=fpl-token-$(printf %s "$FPL_REFRESH_TOKEN" | sha256sum | cut -c1-16)" >> "$GITHUB_OUTPUT"
    ;;
  open)
    if [ -f "$enc" ]; then
      # Via un fichier temporaire : un déchiffrement raté ne laisse rien que `seal` resauverait.
      crypt -d -in "$enc" -out "$FPL_TOKEN_FILE.tmp"
      mv "$FPL_TOKEN_FILE.tmp" "$FPL_TOKEN_FILE"
    fi
    ;;
  seal)
    # `fpl` n'écrit $FPL_TOKEN_FILE que si la FPL a donné un nouveau jeton.
    if [ -f "$FPL_TOKEN_FILE" ]; then
      crypt -in "$FPL_TOKEN_FILE" -out "$enc"
      echo "sealed=true" >> "$GITHUB_OUTPUT"
    fi
    ;;
  *)
    echo "Usage : $0 key|open|seal" >&2
    exit 2
    ;;
esac
