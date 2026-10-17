# fpl-assistant

Prédire les points Fantasy Premier League de chaque joueur, puis choisir l'effectif de 15 qui maximise les points attendus sous les règles du jeu (budget 100 M£, 2/5/5/3 par poste, 3 joueurs max par club, XI valide et capitaine).

Suite de [RSPL](https://github.com/GonoyaPedrus/RSPL), projet de fin d'études, réécrit proprement.

## Démarrage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
fpl ingest      # télécharge 2020-21 → saison en cours dans data/gw.parquet
fpl baseline    # évalue la baseline « forme récente » sur un split temporel
pytest
```

## Données

Historique par journée issu de [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League) ; saison en cours (joueurs, prix, calendrier, ton équipe) via l'API officielle `fantasy.premierleague.com/api` (`fpl_assistant.api`). Les données ne sont pas versionnées : `data/` est ignoré par git.

## Feuille de route

1. Ingestion + baseline forme récente évaluée sur split temporel
2. Modèle GBM (LightGBM) qui doit battre la baseline : difficulté des adversaires, minutes attendues, xG/xA
3. Optimiseur PuLP branché sur les prédictions de la prochaine journée
4. Transferts : partir de mon équipe actuelle, coût des transferts (-4), horizon de plusieurs journées
5. Routine hebdomadaire avant la deadline, puis une interface
