# World Model v3 — Réduction de la dérive HP sans post-processing C#

Objectif: corriger la sous-estimation systématique des dégâts d'attaque observée en v2, sans ajouter de garde-fous dans `WorldModelBackend.Step()`.

Contrainte non négociable:
- aucun post-processing C# spécifique modèle (pas de forçage delta HP sur Pass, pas d'arrondi imposé dans le backend)
- la qualité doit venir du modèle et du training

## 1) Diagnostic v2

Symptômes observés:
- divergences concentrées sur les steps d'attaque
- sous-estimation typique des dégâts (écarts +3 à +6 HP)
- actions neutres (Pass/AttachEnergy) largement corrigées

Lecture:
- biais de régression vers la moyenne sur la composante HP en contexte attaque
- signal de supervision HP trop diffus dans la tête delta globale

## 2) Pistes v3

### Piste A — Tête HP dédiée (multi-tête)

Principe:
- conserver la tête principale `delta_obs[122]`
- ajouter une tête auxiliaire dédiée HP, par exemple:
  - `delta_hp_self`
  - `delta_hp_opp`

Loss:
- `L_total = L_delta + w_hp_aux * L_hp_aux + L_reward + L_done + L_constraints`
- `L_hp_aux`: MSE ou Huber sur les deltas HP cibles

Intérêt:
- supervision plus forte et plus stable sur la variable critique
- réduit la dilution du signal HP dans les 122 dims

### Piste B — Dégâts en classification discrète

Principe:
- ajouter une tête `damage_class` qui prédit une classe de dégâts discrets
- classes POC proposées: `{0, 10, 20, 30, 40, 50, 60}`
- mapping classe -> dégâts appliqués dans la cible d'entraînement

Loss:
- cross-entropy sur `damage_class`
- combinable avec tête HP dédiée (Piste A)

Intérêt:
- évite les sorties intermédiaires biaisées (ex: 24.7 au lieu de 20/30)
- colle à la nature discrète des dégâts dans le POC

### Piste C — Repondération attaque plus agressive

Principe:
- augmenter le poids des transitions d'attaque (`Attack0..Attack3`)
- baseline v2: ~x3
- cible v3 expérimentale: x10 à x20

Loss auxiliaire métier:
- `attack_hp_loss`: pénalité explicite sur erreur HP en steps d'attaque
- `pass_hp_loss` conservée (déjà efficace en v2)

Intérêt:
- force l'optimiseur à prioriser les erreurs gameplay critiques

## 3) Design v3 recommandé

Version v3.1 (recommandée):
- Piste A + Piste C
- garder régression continue pour limiter la complexité d'intégration

Version v3.2 (si v3.1 insuffisante):
- Piste A + Piste B + Piste C
- architecture hybride (delta continu + dégâts discrets)

## 4) Datasets et split

Invariants:
- dataset brut inchangé
- split train/val/test par `game_id` (jamais par transition)
- sampler pondéré conservé

Ajouts:
- audit de couverture des classes de dégâts pour la Piste B
- contrôle des transitions d'attaque rares

## 5) Plan d'expérimentation

Runs minimum:
- R0: v2 baseline (référence)
- R1: v3.1 (tête HP + poids attaque x10)
- R2: v3.1 (tête HP + poids attaque x20)
- R3: v3.2 (ajout damage_class)

Comparaison:
- mêmes seeds de benchmark
- mêmes protocoles de dual backend

## 6) Métriques de décision

Offline:
- `ko_miss`
- `pass_hp`
- `attack_hp_mae` (nouvelle métrique obligatoire)
- `done_acc`, `done_f1`

Rollout dual backend (critères bloquants):
- `HP divergence MAE < 1.0`
- `HP divergence on Pass = 0` (ou quasi nul à tolérance stricte)
- `Cumul drift/game (HP) < 5.0`
- `KO miss rate < 1%` (cible 0)

## 7) Promotion modèle

Un modèle v3 est promu uniquement si:
- tous les seuils rollout sont PASS
- aucune correction ad hoc n'est ajoutée côté C#
- la stabilité se confirme sur seeds non vus

## 8) Notes d'implémentation

- exporter les nouvelles têtes en ONNX avec noms de sorties stables
- versionner le schéma de sortie (`model_output_version`)
- conserver compatibilité v2/v3 via adaptateur explicite côté runner ONNX

