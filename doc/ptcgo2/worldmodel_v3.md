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

## 2) Pistes v3 retenues

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

Version v3.1 (recommandée et unique):
- Piste A + Piste C
- garder régression continue pour limiter la complexité d'intégration

## 4) Datasets et split

Invariants:
- dataset brut inchangé
- split train/val/test par `game_id` (jamais par transition)
- sampler pondéré conservé

Ajouts:
- contrôle des transitions d'attaque rares

## 5) Plan d'expérimentation

Runs minimum:
- R0: v2 baseline (référence)
- R1: v3.1 (tête HP + poids attaque x10)
- R2: v3.1 (tête HP + poids attaque x20)

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

## 9) Rejeté

### Piste B — Dégâts en classification discrète

Statut:
- rejetée pour le design cible

Raison:
- la classification discrète fonctionne en POC car l'ensemble des dégâts est petit et quasi fini
- dans les versions ultérieures du jeu, l'espace des dégâts et modificateurs devient beaucoup plus large et contextuel
- cette approche impose de redéfinir les classes à chaque extension/règle, ce qui crée une forte dette de maintenance
- elle risque de ne pas scaler proprement avec évolutions, effets d'attaque, outils et modificateurs conditionnels

Décision:
- privilégier une approche pérenne basée sur régression contrainte (Piste A + Piste C)
- conserver la piste B uniquement comme référence historique, non planifiée en implémentation
