# World Model v3.1 — Pass HP Fix (spec rapide)

Objectif
- Faire passer le benchmark final en réduisant la dérive HP sur action `Pass`.
- Cible prioritaire: `HP divergence on Pass` (actuel ~0.1287) et `Cumul drift/game (HP)` (actuel ~5.7962).

Contexte
- Les métriques critiques sont déjà bonnes: `HP divergence MAE`, `KO miss rate`, `Done F1`, `Reward MAE`.
- Le point bloquant restant est la dérive sur actions neutres (`Pass`).

Changement demandé
- Ne modifier que le poids de `loss_pass` dans le training.
- Baseline actuelle: `2.0 * loss_pass`.
- Tester en priorité:
  - variante A: `5.0 * loss_pass`
  - variante B: `10.0 * loss_pass`

Contraintes (pour isoler l'effet)
- Aucun changement architecture (heads, dimensions, backbone).
- Aucun changement dataset.
- Aucun changement des autres poids de loss.
- Aucun post-processing C#.
- Même protocole de benchmark (mêmes seeds, mêmes seuils).

Implémentation
- Fichier cible: `training/train.py`.
- Rendre le poids `loss_pass` configurable en argument CLI (si pas déjà fait), ex: `--w-pass`.
- Valeur par défaut recommandée: conserver `2.0` pour compatibilité.

Plan d'exécution
1. Run R1: `w_pass=5.0`.
2. Run R2: `w_pass=10.0`.
3. Export ONNX du meilleur run (selon critères ci-dessous).
4. Benchmark dual backend complet.

Critères de décision (priorité ordre)
1. `Cumul drift/game (HP) < 5.0` (bloquant)
2. `HP divergence on Pass` en baisse nette vs 0.1287
3. Pas de régression critique sur:
   - `HP divergence MAE`
   - `KO miss rate`
   - `Done F1`
   - `Reward MAE`

Règle de sélection
- Si R1 passe le seuil drift sans régression: retenir R1.
- Sinon tester R2.
- Si ni R1 ni R2 ne passent: passer à v3.2 (ajustements supplémentaires, hors scope de cette spec).

Livrables attendus
- 2 runs comparatifs (R1, R2) avec configs enregistrées.
- 1 tableau comparatif baseline v3.1 actuelle vs R1 vs R2.
- 1 benchmark final du run retenu.
- Décision explicite: "GO production" ou "GO v3.2".
