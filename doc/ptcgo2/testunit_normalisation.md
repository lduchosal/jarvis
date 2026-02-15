# Tests Unitaires — Pipeline Normalisation/Dénormalisation (C#)

Objectif: valider ou invalider rapidement l'hypothèse "bug pipeline" côté `WorldModelBackend` (et non bug d'apprentissage du réseau).

Ce document définit:
- les tests unitaires C# prioritaires
- les invariants attendus
- les critères PASS/FAIL
- une proposition d'implémentation pragmatique

## Portée

Composants visés:
- `Observe()` (projection `GameState -> obs[122]`)
- application du `delta_obs` prédit
- dénormalisation et mapping `obs -> GameState` (si utilisé)
- logique d'application post-modèle dans `WorldModelBackend.Step()`

Hors portée:
- qualité intrinsèque du modèle PyTorch
- tuning des hyperparamètres

## Pré-requis

- framework test C#: xUnit ou NUnit (au choix du repo)
- fixtures déterministes de `GameState` (au moins 2 cas: early-game, near-KO)
- implémentation d'un "model runner" mockable (retour delta/reward/done injecté)

## Test 1 — Delta nul (test coupe-circuit)

But:
- prouver qu'un `delta_obs = 0` n'introduit aucune dérive d'état.

Setup:
- backend WM configuré avec un runner mock qui retourne:
  - `delta_obs = zeros(122)`
  - `reward = 0`
  - `done_logit` cohérent avec `done=false`
- état initial déterministe non terminal

Procédure:
1. exécuter une action valide (ex: `Pass`) pendant N steps (N=10 recommandé)
2. après chaque step, comparer l'état C# attendu à l'état obtenu

Invariants attendus:
- HP identiques à chaque step
- énergies identiques (hors changements strictement dus aux règles si action non neutre)
- aucun drift sur les champs dynamiques
- aucune variation sur les champs statiques

Critère PASS:
- drift exact = 0 sur toutes les dims monitorées

Critère FAIL:
- toute variation non expliquée par la règle du moteur

Note:
- c'est le test le plus décisif. Si FAIL, le bug est pipeline C#.

## Test 2 — Round-trip avec delta synthétique connu

But:
- vérifier la réversibilité normalisation/dénormalisation et l'application correcte d'un delta contrôlé.

Setup:
- état initial déterministe
- delta synthétique simple et interprétable, par exemple:
  - `P1.HP -= 10` (normalisé selon convention)
  - `P0.energy[type] += 1`
  - le reste à 0

Procédure:
1. produire `obs0` via `Observe()`
2. appliquer le delta synthétique via le même chemin que le WM
3. reconstruire/appliquer l'état final côté C#
4. comparer aux valeurs physiques attendues

Invariants attendus:
- HP final exactement égal à la valeur cible (avec clamp bornes jeu)
- énergie finale exactement égale à la valeur cible
- aucune modification hors champs ciblés

Critère PASS:
- erreur absolue <= 1e-6 sur valeurs normalisées
- erreur absolue = 0 sur valeurs discrètes reconstruites (int/flags)

Critère FAIL:
- toute erreur de mapping (off-by-one, mauvaise échelle, mauvais clamp)

## Test 3 — Oracle dataset (transition réelle)

But:
- vérifier l'alignement collecte/inférence sur des transitions réelles du dataset.

Setup:
- sélectionner K transitions (K=100 minimum) depuis `data/transitions.jsonl`
- pour chaque transition: `obs`, `action`, `next_obs`, `reward`, `done`

Procédure:
1. injecter `delta_oracle = next_obs - obs` (au lieu du delta modèle)
2. exécuter le chemin d'application C# pour obtenir `obs_pred`
3. comparer `obs_pred` à `next_obs`

Invariants attendus:
- `obs_pred` ~= `next_obs` à tolérance faible
- cohérence reward/done propagés

Critère PASS:
- MAE globale < 1e-6 sur dims continues
- mismatch = 0 sur champs bool/int

Critère FAIL:
- écart systématique sur une famille de features (HP, énergie, phase, joueur)

## Test 4 — Action neutre stricte (Pass)

But:
- verrouiller explicitement le bug observé: dérive HP sur `Pass`.

Setup:
- état non terminal, aucun effet latent
- runner mock renvoyant delta nul

Procédure:
1. appeler `Step(action=Pass)`
2. vérifier post-state

Invariants attendus:
- aucune perte/gain HP
- pas d'altération d'énergie inattendue

Critère PASS:
- HP avant == HP après (strict)

Critère FAIL:
- toute variation HP sur `Pass`

## Test 5 — Terminal KO et done

But:
- vérifier la cohérence de la réconciliation terminale (`done = done_wm OR HP<=0`).

Setup:
- état proche KO
- delta synthétique qui met HP cible à 0
- cas A: `done_wm=false`
- cas B: `done_wm=true`

Procédure:
1. appliquer le delta terminal
2. vérifier `done`, winner et état final

Invariants attendus:
- si HP<=0 alors `done=true` dans tous les cas
- état final figé selon règles moteur

Critère PASS:
- règle de réconciliation respectée à 100%

Critère FAIL:
- partie continue avec HP<=0

## Instrumentation recommandée

Pour accélérer le debug en cas de FAIL:
- loguer `obs_in`, `delta_in`, `obs_out` (top 10 dims les plus divergentes)
- loguer conversion normalisée <-> physique pour HP/énergie
- loguer les clamps appliqués

## Convention de tolérance

- dims continues normalisées: tolérance `1e-6`
- reconstructions discrètes (int, bool, phase, current_player): égalité stricte

## Plan d'implémentation proposé

Fichiers recommandés:
- `tests/WorldModelBackendNormalizationTests.cs`
- `tests/Fixtures/GameStateFixtureFactory.cs`
- `tests/Mocks/FakeModelRunner.cs`

Cas de test minimaux:
- `ZeroDelta_NoStateDrift_Across10Steps`
- `SyntheticDelta_RoundTrip_IsExact`
- `DatasetOracleDelta_ReproducesNextObs`
- `PassAction_DoesNotChangeHp`
- `KoReconciliation_ForcesDoneWhenHpZero`

## Gating CI

Règle proposée:
- ces tests sont bloquants avant tout run "dual backend" ou benchmark divergence
- en cas d'échec: bloquer export/release backend WM côté C#

## Critère de sortie (done)

Hypothèse "pipeline bug" validée/invalidée quand:
- les 5 tests passent en local et CI
- aucun drift sur delta nul
- aucune dérive HP sur `Pass`
- alignement oracle dataset confirmé
