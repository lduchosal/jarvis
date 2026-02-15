# World Model v2.1 — Extension dataset uniquement (aucun changement modèle)

Objectif: améliorer la prédiction des dégâts en élargissant uniquement la diversité des transitions d'entraînement.

Principe de v2.1:
- aucune modification d'architecture réseau
- aucune modification de loss, pondérations, scheduler, optimizer
- aucune modification du backend C# (pas de post-processing)
- seul le dataset change

## 1) Hypothèse

Le biais de sous-estimation des dégâts vient en partie d'une couverture trop faible des gros dégâts/HP dans le dataset actuel (Basic-only).

Hypothèse testée:
- en élargissant le CardRegistry aux cartes Base Set complètes (Base + Stage 1 + Stage 2),
- tout en gardant le protocole POC "1 Pokémon actif, pas de règle d'évolution",
- le modèle v2 devrait mieux apprendre les amplitudes de dégâts et réduire la dérive HP en rollout.

## 2) Changement unique autorisé

Changement autorisé:
- élargissement du pool de cartes utilisé pour la collecte des transitions

Changements interdits (pour isoler l'effet dataset):
- code modèle
- fonctions de loss
- pondérations de sampling
- hyperparamètres de training
- logique de step backend

## 3) Scope dataset v2.1

Pool cartes:
- Base Set complet: cartes Pokémon Basic + Stage 1 + Stage 2

Convention POC conservée:
- toutes les cartes du pool sont jouables directement comme Pokémon actif
- aucune mécanique d'évolution introduite dans les règles du POC

Note:
- ce choix est volontairement "non compétitif" mais valide pour apprentissage de dynamique

## 4) Pipeline

1. Mettre à jour le CardRegistry de collecte avec le pool v2.1
2. Recollecter un nouveau dataset transitions (même format JSONL)
3. Exécuter les checks qualité dataset existants
4. Entraîner exactement le même modèle v2 avec les mêmes hyperparamètres
5. Export ONNX
6. Benchmark dual backend avec protocole identique à v2

## 5) Contrôle d'isolation

Pour garantir que seul le dataset change:
- taguer le run: `wm_v2.1_data_only`
- stocker hash/commit du code training identique à v2
- stocker hash du nouveau dataset v2.1
- comparer config training v2 et v2.1 (doit être strictement identique)

## 6) Métriques d'évaluation

Offline (mêmes métriques que v2):
- `val_delta`, `val_reward`, `val_done`, `done_acc`
- `pass_hp`, `ko_miss`

Rollout dual backend (mêmes seuils):
- HP divergence MAE
- HP divergence on Pass
- KO miss rate
- Done F1
- Reward MAE
- Cumul drift/game (HP)

## 7) Critère de succès v2.1

Succès attendu minimum:
- amélioration mesurable de `HP divergence MAE`
- amélioration de `Cumul drift/game (HP)`
- maintien de `KO miss rate` à 0 (ou <= seuil)
- pas de dégradation critique de `Done F1`

Critère d'arrêt:
- si v2.1 n'améliore pas les métriques rollout de dégâts, passer à v3 (architecture/loss)

## 8) Livrables

- dataset v2.1 (transitions)
- rapport qualité dataset v2.1
- checkpoint modèle v2 entraîné sur dataset v2.1
- benchmark comparatif v2 vs v2.1 (mêmes seeds)
- conclusion "data-only suffisant" ou "passage v3 requis"

## 9) Décision suivante

Si v2.1 atteint les seuils rollout:
- promouvoir v2.1 comme baseline opérationnelle

Sinon:
- lancer v3 (tête HP dédiée + repondération attaque) sur le dataset élargi

