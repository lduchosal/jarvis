# Checkpoint — World Model POC (2026-02-16)

Panel : Q (humain), Claude Opus 4.6, Codex 5.3

## Objectif global

Entraîner un MLP (~130K params) qui approxime le backend C# du jeu Pokémon TCG POC.
Le world model est un BACKEND (prédit les conséquences d'une action), pas un joueur.

## Specs produites

| Fichier                                | Contenu |
|----------------------------------------|---------|
| `doc/ptcgo2/board.md`                  | Encodage board v2.1 (49 dims/slot) |
| `doc/ptcgo2/poc.md`                    | Règles POC simplifiées (1 Pokémon, pas d'évolution, 6 actions) |
| `doc/ptcgo2/impl.md`                   | Implémentation C# Milestone 1 (ITransitionBackend, IActionProvider, CardRegistry 69 cartes) |
| `doc/ptcgo2/worldmodel.md`             | Spec world model (pipeline, convention perspective, statiques/dynamiques, réconciliation done, current_player) |
| `doc/ptcgo2/worldmodel_v2.md`          | Correction v2 (loss pondérée, loss auxiliaires pass/ko, rééquilibrage sampling) |
| `doc/ptcgo2/worldmodel_v3.md`          | Correction v3 (tête HP dédiée, piste B rejetée) |
| `doc/ptcgo2/v3/tuning.md`              | Spec tuning v3.4 (3 phases, accélération, tuning, validation) |
| `doc/ptcgo2/dataset_quality.md`        | Checks qualité dataset (7 critères, seuils PASS/FAIL) |
| `doc/ptcgo2/testunit_normalisation.md` | 5 tests unitaires C# pour valider le pipeline normalisation |
| `doc/ptcgo2/faq/training_loss.md`      | FAQ loss (MSE, BCE, Huber, tableau des termes) |
| `doc/ptcgo2/faq/mlp_head.md`           | FAQ têtes dédiées (quoi, pourquoi, coût, exemple concret) |
| `doc/ptcgo2/faq/hyperparameters.md`    | FAQ hyperparamètres (paramètres vs hyperparamètres, tableau complet) |

## Décisions clés

- One-hot pour catégoriels (pokemonType, evoStage, weaknessType, resistanceType)
- evoFamilyId en scalaire (embedding learnable plus tard)
- Delta prediction (pas reconstruction état complet)
- Convention perspective : obs/next_obs toujours du point de vue du joueur qui agit
- Séparation statiques (94 dims) / dynamiques (28 dims)
- current_player = déterministe (pas prédit par le réseau)
- done = réconcilié (WM prediction OR HP <= 0)
- Piste B (classification discrète des dégâts) rejetée (ne scale pas)
- Pas de post-processing C# — le modèle doit porter la qualité seul
- JSON (pas YAML) pour le protocole
- Option A pour infractions (action rejetée, état inchangé, reward -0.01)

## Historique des runs

| Run | Dataset | Config | HP MAE | Pass HP | KO miss | Drift | Notes |
|-----|---------|--------|--------|---------|---------|-------|-------|
| v1 | 1.9M, 23 cartes | baseline | - | - | - | - | Loss excellente offline, divergence forte en jeu |
| v2 | 1.9M, 23 cartes | +pass/ko loss, sampling pondéré | 0.70 | 0.018 | 0.000 | 26.6 | Pass corrigé, drift élevé |
| v2.1 | 212K, 69 cartes | v2 config, data-only | 0.88 | 0.057 | 0.000 | 35.1 | Data-only insuffisant (volume trop faible) |
| v3.0 | 2.14M, 69 cartes | +tête HP, w_attack_hp=10 | 0.45 | 0.125 | 0.000 | 19.2 | Avant clamp fix C# |
| v3.1 | 2.14M, 69 cartes (pre-clamp) | v3.0, bench post-clamp | 0.14 | 0.129 | 0.000 | 5.80 | Biaisé (train pre-clamp, bench post-clamp) |
| v3.2 | pre-clamp | w_pass augmenté | 0.16 | 0.117 | 0.000 | 7.00 | Régression drift |
| v3.3 | post-clamp | batch=4096, lr=4e-3, w_pass=10, w_attack_hp=15 | 0.47 | 0.304 | 0.000 | 20.3 | Config trop agressive, régression massive |
| **v3.4** | **2.14M post-clamp** | **batch=1024, lr=2e-3, w_pass=2, w_hp=5, w_atk=10** | **0.18** | **0.189** | **0.000** | **7.87** | **Vraie baseline post-clamp** |

## État actuel

- **v3.4 est la baseline de référence** (seul run train+bench aligné post-clamp)
- Drift à 7.87, seuil à 5.0 — il manque ~37% de réduction
- KO miss = 0 depuis la v2 (résolu définitivement)
- Pass HP reste le contributeur principal au drift résiduel
- Pipeline normalization C# validé par 5 tests unitaires (pas de bug pipeline)

## Bug important découvert et corrigé

Le moteur C# retournait des HP négatifs sur KO (ex: -90). Le WM clampait à 0. Les runs v3.0 à v3.2 ont été entraînés sur données pre-clamp puis benchmarkés post-clamp — les résultats étaient biaisés. Corrigé dans v3.4 : tout le pipeline est aligné post-clamp.

## Prochaines étapes

1. **Option A** : run avec w_pass=5 (batch=1024, lr=2e-3), ~90 min — tester si ça réduit le drift sous 5.0
2. **Option B** : si A insuffisant, run batch=256, lr=1e-3 pour vérifier si le batch 1024 dégrade la convergence fine
3. Si ni A ni B ne passent → v3.5 (nouveau levier architecture ou loss)

## Contraintes techniques

- Hardware : MacBook Pro M3 Max, 128 Go RAM
- Training ~90 min (batch=1024) ou ~3h (batch=256)
- 69 cartes Base Set (Basic + Stage 1 + Stage 2, jouées sans règles d'évolution)
- Dual backend pour benchmark (C# et WM en parallèle sur mêmes actions)
