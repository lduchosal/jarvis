# World Model — Étape 2

Entraîner un MLP qui approxime `CSharpBackend.Step()` et l'intégrer comme second `ITransitionBackend`. Le moteur C# reste la vérité terrain ; le world model est un substitut appris, différentiable et rapide.

Rappel : le world model est un BACKEND. Il prédit les conséquences d'une action, il ne choisit jamais quelle action jouer.

## Pipeline complète

```
1. Collecte       CSharpBackend joue N parties random → transitions JSONL
2. Dataset         Python charge les transitions, calcule deltas + one-hot
3. Entraînement    PyTorch MLP, loss MSE + BCE, monitoring Wandb
4. Export          PyTorch → ONNX
5. Intégration     WorldModelBackend (C#) charge le .onnx via OnnxRuntime
6. Validation      Mode dual backend : même partie, C# vs WM, mesure divergence
```

## Convention de perspective (CRITIQUE)

Règle unique appliquée identiquement en collecte ET en inférence :

> `obs` et `next_obs` sont TOUJOURS du point de vue du joueur qui agit, même si `current_player` change après l'action.

En collecte :
1. Observer l'état AVANT le step, du point de vue du joueur courant → `obs`
2. Exécuter l'action via `CSharpBackend.Step()`
3. Observer l'état APRÈS le step, mais du point de vue du MÊME joueur qu'avant → `next_obs`
4. Enregistrer (obs, action, next_obs, reward, done)

En inférence (`WorldModelBackend.Step()`) :
1. Observer l'état courant du point de vue du joueur courant → `obs`
2. Prédire le delta via le MLP
3. Appliquer le delta sur le GameState
4. NE PAS changer la perspective — le delta est déjà exprimé dans la perspective du joueur qui agit

Toute incohérence de perspective entre collecte et inférence rendra les prédictions incohérentes.

## Séparation champs statiques / dynamiques (CRITIQUE)

Le WM ne prédit PAS l'état complet. Il prédit un delta appliqué aux seuls champs dynamiques du GameState. Les champs statiques restent inchangés et sont conservés hors modèle.

### Champs statiques (ne changent JAMAIS dans le POC)

```
Champ                 Raison
────────────────────  ─────────────────────────────────────
cardIndex             identifiant carte (affichage/debug)
pokemonType           pas d'évolution dans le POC
MaxHP                 pas d'effet qui modifie MaxHP
weaknessType          fixe par carte
resistanceType        fixe par carte
resistanceValue       fixe par carte
atkExists[4]          fixe par carte
atkDamage[4]          fixe par carte (pas d'effets)
atkTotalCost[4]       fixe par carte
```

### Champs dynamiques (modifiables par le WM)

```
Champ                 Dims   Prédiction attendue
────────────────────  ─────  ─────────────────────────────
HP (joueur courant)     1    dégâts reçus → HP diminue
HP (adversaire)         1    dégâts infligés → HP diminue
energy[11] (courant)   11    +1 si AttachEnergy
energy[11] (adverse)   11    inchangé (pas de vol d'énergie)
energyAttached (crt)    1    0→1 si AttachEnergy, reset à 0 en fin de tour
energyAttached (adv)    1    inchangé
Phase                   1    StartTurn→Main, Main→StartTurn/GameOver
TurnIndex               1    +1 en fin de tour
```

Total dynamique : 28 dims sur 122.

### Invariants après application du delta

Après chaque `WorldModelBackend.Step()`, vérifier :
- `HP >= 0` (clamper si le WM prédit une valeur négative)
- `energy[i] >= 0` pour tout i
- `energyAttached` ∈ {0, 1} (arrondir au plus proche)
- `Phase` ∈ {0, 0.5, 1} (arrondir au plus proche)
- Tous les champs statiques sont inchangés (assertion en debug)
- `cardIndex` est préservé tel quel

## Étape 2.1 — Collecte de données

### Commande

```bash
dotnet run --project src/Ptcgo2.Console -- collect --games 10000 --output data/transitions.jsonl
```

### Format de sortie

Un fichier JSONL, une ligne par transition :

```json
{
  "obs": [0,0,0,1,0,0,0,0,0,0,0, 0.3, 0.3, ...],
  "action": 2,
  "next_obs": [0,0,0,1,0,0,0,0,0,0,0, 0.23, 0.3, ...],
  "reward": 0.0,
  "done": false
}
```

- `obs` et `next_obs` : vecteur 122 dims (spec poc.md), perspective du joueur qui agit (cf. convention de perspective ci-dessus)
- `action` : entier 0-5
- `reward` : 0.0 (en cours), +1.0 (victoire), -1.0 (défaite), -0.01 (infraction)
- `done` : bool

### Volume cible

- 10 000 parties random × ~10 steps/partie = ~100 000 transitions
- Taille estimée : ~50 Mo en JSONL

## Étape 2.2 — Observation (Observe)

Fonction qui convertit le GameState canonique en vecteur 122 dims :

```
Dims  Bloc                         Contenu
────  ───────────────────────────  ─────────────────────────────────
 60   Joueur courant (slot)        pokemonTypeOH[11], HP, MaxHP,
                                   weaknessTypeOH[11], resistanceTypeOH[11],
                                   resistanceValue, atkExists[4], atkDamage[4],
                                   atkTotalCost[4], effectiveEnergy[11],
                                   energyAttached
 60   Adversaire (slot)            même layout
  1   Phase                        normalisé (StartTurn=0, Main=0.5, GameOver=1)
  1   TurnIndex                    normalisé /50
────
122
```

Les features statiques (type, attaques, faiblesse, résistance) sont lookées dans le CardRegistry via `cardIndex`. Les features dynamiques (HP, énergie, energyAttached) viennent du runtime.

`Observe()` prend toujours la perspective du joueur spécifié. En collecte, on force la perspective du joueur qui a agi pour `next_obs`, même si `current_player` a changé.

## Étape 2.3 — Dataset Python

```python
class TransitionDataset(Dataset):
    def __init__(self, path):
        # Charger JSONL
        self.obs = ...          # [N, 122] float32
        self.actions = ...      # [N] int64
        self.next_obs = ...     # [N, 122] float32
        self.rewards = ...      # [N] float32
        self.dones = ...        # [N] float32

        # Pré-calcul
        self.delta_obs = self.next_obs - self.obs          # [N, 122]
        self.action_oh = F.one_hot(self.actions, 6).float() # [N, 6]
```

Split : 90% train, 10% validation.

Note : le delta des champs statiques devrait être ~0 sur toutes les transitions. C'est un bon sanity check — si des dims statiques ont un delta non nul, il y a un bug dans `Observe()` ou dans la collecte.

## Étape 2.4 — Architecture WorldModelNet

```
Input: [B, 128] = cat(obs[122], action_one_hot[6])
  │
  ├─→ Linear(128, 256) → LayerNorm → ReLU           = h1
  ├─→ Linear(256, 256) → LayerNorm → ReLU + h1       = h2  (residual)
  │
  ├─→ Linear(256, 122) ──→ delta_obs    (changement d'état)
  ├─→ Linear(256, 1)   ──→ reward       (+1/-1/0/-0.01)
  └─→ Linear(256, 1)   ──→ done_logit   (probabilité fin de partie)

~200K paramètres
```

Le réseau prédit le delta complet (122 dims) mais en pratique seuls ~28 dims dynamiques auront des deltas non nuls dans les données. Le réseau apprendra naturellement à prédire ~0 pour les dims statiques.

## Étape 2.5 — Entraînement

### Loss

```python
# Masquer les transitions terminales pour obs_loss
not_done = (1 - done).unsqueeze(1)  # [B, 1]
obs_loss = (not_done * (pred_delta - delta_obs) ** 2).mean()
rew_loss = F.mse_loss(pred_reward, reward)
done_loss = F.binary_cross_entropy_with_logits(pred_done_logit, done)

total_loss = obs_loss + rew_loss + 0.5 * done_loss
```

- `obs_loss` : MSE sur le delta, masqué pour les transitions terminales (delta non significatif quand done=1)
- `rew_loss` : MSE sur le reward
- `done_loss` : BCE, pondéré ×0.5 (échelles différentes MSE vs BCE)

### Hyperparamètres

```
Optimizer       Adam
Learning rate   1e-3
Batch size      256
Epochs          50
Scheduler       aucun (POC)
```

### Monitoring Wandb

Chaque époque :
- `train/loss`, `train/obs_loss`, `train/rew_loss`, `train/done_loss`
- `val/loss`, `val/obs_loss`, `val/rew_loss`, `val/done_loss`

Toutes les 10 époques :
- MSE par composant d'observation (statiques vs dynamiques séparément)
- Done : accuracy, precision, recall, F1
- Table de prédictions échantillonnées (actual vs predicted)
- Sanity check : MSE sur les dims statiques (devrait être ~0)

## Étape 2.6 — Export ONNX

```python
dummy_input = torch.randn(1, 128)
torch.onnx.export(model, dummy_input, "worldmodel.onnx",
                  input_names=["input"],
                  output_names=["delta_obs", "reward", "done_logit"],
                  dynamic_axes={"input": {0: "batch"}})
```

Le fichier `.onnx` est versionné avec le hash du dataset et les métriques de validation.

## Étape 2.7 — Intégration C# (WorldModelBackend)

```csharp
public class WorldModelBackend : ITransitionBackend
{
    private InferenceSession _session;  // OnnxRuntime

    public StepResult Step(in GameState state, int actionId)
    {
        // 1. Convertir GameState → obs[122] via Observe(state, state.CurrentPlayer)
        // 2. One-hot action → action_oh[6]
        // 3. Concat → input[128]
        // 4. Run ONNX → delta_obs[122], reward[1], done_logit[1]
        // 5. Appliquer delta sur les CHAMPS DYNAMIQUES UNIQUEMENT :
        //    - HP courant/adverse, energy[], energyAttached, phase, turnIndex
        // 6. Conserver les champs statiques du GameState original :
        //    - cardIndex, type, MaxHP, attaques, faiblesse, résistance
        // 7. Appliquer les invariants :
        //    - HP = clamp(HP, 0, MaxHP)
        //    - energy[i] = max(0, round(energy[i]))
        //    - energyAttached = round(energyAttached) ∈ {0, 1}
        //    - Phase = nearest({0, 0.5, 1})
        //    - done = sigmoid(done_logit) > 0.5
        // 8. Retourner StepResult
    }

    public ReadOnlySpan<int> GetLegalActions(in GameState state)
    {
        // Même logique que CSharpBackend (règles déterministes)
        // La légalité est calculée à partir du GameState, pas prédite par le WM
    }
}
```

Règle : `GetLegalActions` utilise les règles déterministes, pas le world model. La légalité est basée sur le GameState canonique.

### Lancement

```bash
dotnet run --project src/Ptcgo2.Console -- --p1 human --p2 random --backend worldmodel --model worldmodel.onnx
```

Le champ `"backend"` dans le protocole JSON passe de `"csharp"` à `"worldmodel"`.

## Étape 2.8 — Mode dual backend

Même partie, mêmes policies, mêmes actions. Les deux backends (C# et WM) exécutent chaque step en parallèle. Affichage des divergences :

```
Tour 3 — Action: Attack0
  C#:  HP adverse 20 → 0, done=true,  reward=+1
  WM:  HP adverse 20 → 2, done=false, reward=0
  ⚠ DIVERGENCE: HP (Δ=2), done, reward
```

Métrique globale : % de steps où le WM diverge du C#, décomposé par composant (HP, énergie, phase, done, reward).

## Critères de succès Étape 2

- Le pipeline collecte → train → export → intégration tourne sans erreur
- La loss converge sur 50 époques (val_loss < train_loss × 1.1)
- Prédiction done : F1 > 0.8
- Prédiction dégâts : MSE < 5 (sur les dims HP)
- Prédiction énergie : MSE < 1
- Dims statiques : MSE ≈ 0 (sanity check)
- Mode dual backend : < 5% de divergences significatives sur 100 parties random
- Un humain peut jouer une partie complète contre le WM backend

## Structure de code Étape 2

```text
ptcgo2/
  src/
    Ptcgo2.Core/           (existant)
    Ptcgo2.Console/        (existant, ajout --backend worldmodel)
    Ptcgo2.Backend.WM/
      WorldModelBackend.cs
      Observe.cs
  training/
    collect.py             (ou commande C# → JSONL)
    dataset.py
    model.py
    train.py
    export_onnx.py
  models/
    worldmodel.onnx
    training_config.json
  tests/
    Ptcgo2.Tests/          (existant, ajout tests WM)
```

## Hors scope Étape 2

- Policy IA entraînée (on utilise human/random)
- Optimisation du réseau (architecture, hyperparamètres)
- Banc, évolution, effets, dresseurs
- Prédiction de légalité par le WM

## Auteurs

Q Humain, spécification produit, le 2026-02-14
Claude Opus 4.6 (Anthropic), rédaction, le 2026-02-14
Codex 5.3 (OpenAI), co-rédaction, le 2026-02-14
