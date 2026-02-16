# World Model v3.4 — Tuning rapide et robuste (M3 Max 128 GB)

Objectif
- Réduire fortement le temps d'itération training tout en préservant la qualité rollout.
- Stabiliser les métriques critiques: `pass_hp`, `attack_hp_mae`, `ko_miss_rate`, `cumul_drift_game_hp`.

Contexte
- Hardware cible: MacBook Pro M3 Max, 128 GB RAM.
- Dataset principal: `data/transitions_v2.1.jsonl` (~2.14M transitions).
- Baseline de comparaison: dernier run v3.1 validé en config standard.

## 1) Règles d'isolation

Toujours fixer:
- même code modèle (architecture inchangée)
- même dataset source
- même split par game_id
- mêmes seeds d'évaluation benchmark

Changer un seul bloc à la fois:
- bloc perf (batch, dataloader)
- bloc loss-weights
- bloc scheduler/epochs

## 2) Plan en 3 phases

Phase A — Accélération des itérations (sans changer la loss)
1. Préchargement complet en RAM des tenseurs (pas de lazy I/O pendant train).
2. Tuning batch size par paliers: 1024 -> 2048 -> 4096.
3. Ajustement learning rate: `lr_new = lr_base * sqrt(batch_new / batch_base)`.
4. Early stopping sur `val_total` avec patience 15.
5. Runs courts 40 à 60 epochs pour screening.

Critère de succès Phase A:
- temps/epoch réduit d'au moins 2x
- pas de dégradation majeure des métriques val (`pass_hp`, `attack_hp_mae`, `ko_miss_rate`)

Phase B — Tuning des poids de loss (focus pass vs attaque)
- Garder meilleur setup perf de Phase A.
- Grille minimale:
  - `w_pass`: [2, 5, 10]
  - `w_attack_hp`: [10, 12, 15]
- Prioriser combinaisons équilibrées (ex: 5/12, 10/12, 10/15).

Critère de succès Phase B:
- baisse nette de `pass_hp` sans remontée significative de `attack_hp_mae`
- `ko_miss_rate` reste proche de 0

Phase C — Validation finale longue
- 1 run complet 150 à 200 epochs avec meilleure config Phase B.
- Sauvegarde best-checkpoint sur `val_total`.
- Export ONNX.
- Benchmark dual backend complet (mêmes seeds standard).

## 3) Métriques de décision

Métriques training/val prioritaires:
- `val/pass_hp_mae` (plus bas = mieux)
- `val/attack_hp_mae` (plus bas = mieux)
- `val/ko_miss_rate` (cible ~0)
- `val/done_acc` et `val/done_loss`

Métriques benchmark bloquantes:
- `HP divergence MAE < 1.0`
- `HP divergence on Pass`: baisse significative vs baseline
- `KO miss rate < 1%` (cible 0)
- `Done F1 > 0.95`
- `Reward MAE < 0.05`
- `Cumul drift/game (HP) < 5.0`

## 4) Stratégie de sélection

Ordre de priorité:
1. passe le seuil `cumul drift/game`
2. minimise `HP divergence on Pass`
3. maintient `attack` stable (pas de régression forte)

Règle:
- une config plus rapide n'est retenue que si elle ne casse pas les métriques rollout.

## 5) Implémentation recommandée

`training/train.py`
- ajouter arguments CLI:
  - `--batch-size`
  - `--lr`
  - `--epochs`
  - `--early-stop-patience`
  - `--w-pass`
  - `--w-attack-hp`
- journaliser temps/epoch et samples/sec.

Dataset loader
- charger et conserver `input`, `delta`, `reward`, `done`, `action`, `obs`, `next_obs` en RAM.
- vérifier absence de parsing JSON dans `__getitem__`.

## 6) Livrables v3.4

- tableau comparatif runs (config + temps + métriques clés)
- meilleure config "fast-tuning"
- meilleur checkpoint final + ONNX
- benchmark final et verdict GO/NO-GO

## 7) GO / NO-GO

GO si:
- `cumul drift/game (HP) < 5.0`
- pas de régression KO/done/reward
- temps d'itération amélioré de façon mesurable

Sinon:
- NO-GO et ouverture v3.5 (nouveau levier architecture ou loss)

