# World Model v2 — Correction du modèle

Suite au dual backend sur la partie Abra vs Farfetch'd (seed 4439695), le pipeline C# est blanchi (5/5 tests normalisation OK). Les divergences viennent du modèle lui-même.

## Divergences observées

```
Step  3  Attack0  P1.HP  C#=40  WM=42  Δ=+2   (sous-estime dégâts)
Step  5  Pass     P0.HP  C#=30  WM=27  Δ=-3   (hallucine dégâts sur Pass)
Step  8  Attack0  P1.HP  C#=30  WM=32  Δ=+2   (sous-estime dégâts)
Step 10  Attack0  P0.HP  C#= 0  WM= 9  Δ=+9   (rate le KO)
```

## Diagnostic

1. **Biais de régression vers la moyenne** : le réseau prédit des deltas HP "moyens" au lieu des valeurs exactes, car les gros dégâts et les KO sont rares dans le dataset (~5% done)
2. **Pas de distinction action→delta** : le réseau n'a pas appris que Pass implique strictement delta_HP=0
3. **Déséquilibre dataset** : 45% Pass, 28% AttachEnergy, 27% Attack — les transitions critiques (KO) sont noyées

## Corrections

### 1. Loss pondérée par type de transition

```python
# Poids par transition
w = torch.ones(B)
w[is_terminal]    *= 10.0   # transitions done=true
w[is_attack]      *=  3.0   # actions 2-5
w[is_ko]          *= 20.0   # done=true ET HP cible <= 0

obs_loss = (w.unsqueeze(1) * (pred_delta - delta_obs) ** 2).mean()
```

Les poids sont cumulatifs : un KO reçoit 10 × 20 = 200× le poids d'un Pass normal.

### 2. Loss auxiliaire de cohérence métier

```python
# Pénalité Pass : si action=Pass, delta HP doit être 0
pass_mask = (action == 0)  # [B]
hp_idx = [11, 71]          # self.HP, opp.HP
pass_hp_delta = pred_delta[pass_mask][:, hp_idx]
pass_loss = (pass_hp_delta ** 2).mean()

# Pénalité KO manqué : si HP cible <= 0, pénaliser HP prédit > 0
ko_mask = (target_hp_opp <= 0)
pred_hp_opp = obs[ko_mask, 71] + pred_delta[ko_mask, 71]
ko_miss_loss = F.relu(pred_hp_opp).mean()

total_loss = obs_loss + rew_loss + 0.5 * done_loss + 2.0 * pass_loss + 5.0 * ko_miss_loss
```

### 3. Rééquilibrage du sampling

```python
class WeightedTransitionSampler(Sampler):
    def __init__(self, dataset):
        weights = torch.ones(len(dataset))
        weights[dataset.dones == 1]      *= 10.0
        weights[dataset.actions >= 2]    *=  3.0
        # KO = done ET HP adverse final <= 0
        ko = (dataset.dones == 1) & (dataset.next_obs[:, 71] <= 0)
        weights[ko]                      *= 20.0
        self.sampler = WeightedRandomSampler(weights, len(dataset))
```

Split train/val par partie (pas par transition) pour éviter les fuites :

```python
game_ids = dataset.game_ids
unique_games = game_ids.unique()
val_games = unique_games[torch.randperm(len(unique_games))[:len(unique_games)//10]]
val_mask = torch.isin(game_ids, val_games)
```

### 4. Hyperparamètres ajustés

```
Optimizer       Adam
Learning rate   1e-3
Batch size      256
Epochs          100        (était 50)
Scheduler       CosineAnnealing (T_max=100)
```

Architecture inchangée (~200K params). Si les métriques rollout ne passent pas après ces corrections, envisager un réseau plus large (512 hidden) ou une tête HP séparée.

## Validation rollout-first

La val_loss offline ne suffit plus. Le critère principal devient le benchmark dual backend.

### Benchmark

```bash
dotnet run -- benchmark --games 200 --seeds-from 100000 --backend dual --model models/worldmodel_v2.onnx
```

200 parties random sur des seeds non vus pendant l'entraînement.

### Métriques et seuils bloquants

```
Métrique                          Seuil PASS     Observé v1
──────────────────────────────    ──────────     ──────────
HP divergence moyenne (MAE)       < 1.0 HP       ~4.0 HP
HP divergence sur Pass            = 0            ≠ 0
Taux de KO ratés (WM dit vivant) < 1%           ~25% (1/4)
Done F1                           > 0.95         ~0.75 (estimé)
Reward MAE                        < 0.05         inconnu
Dérive cumulée par partie (MAE)   < 5.0 HP       inconnu
```

### Critères de promotion

Le modèle v2 remplace v1 si et seulement si :
- TOUS les seuils bloquants sont PASS
- HP divergence sur Pass = 0 strict (tolérance 0.5 HP max)
- Aucun KO raté sur les 200 parties benchmark
- Au moins une partie jouée par un humain sans incohérence visible

## Pipeline mis à jour

```
1. Collecte        (inchangé, réutiliser data/transitions.jsonl)
2. Dataset          ajouter game_id, split par partie, WeightedSampler
3. Entraînement     loss pondérée + auxiliaires, 100 époques
4. Export ONNX      (inchangé)
5. Benchmark        200 parties dual backend, seeds non vus
6. Promotion        si PASS → worldmodel_v2.onnx remplace worldmodel.onnx
```

## Hors scope v2

- Changement d'architecture (reste MLP 2×256)
- Ajout de features dans l'observation
- Prédiction de légalité
- Entraînement sur données non-random (human, IA)

## Auteurs

Q Humain, spécification produit, le 2026-02-14
Claude Opus 4.6 (Anthropic), rédaction, le 2026-02-14
Codex 5.3 (OpenAI), co-rédaction, le 2026-02-14
