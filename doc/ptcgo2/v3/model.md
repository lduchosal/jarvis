# Architecture WorldModelNet v3

Document de référence pour le MLP du world model. Décrit l'architecture exacte, les entrées/sorties, la normalisation, et la séparation training vs inférence.

## Vue d'ensemble

```
Input [128] = concat(obs[122], action_onehot[6])
    │
    ├─ Linear(128, 256) → ReLU
    ├─ Linear(256, 256) → ReLU
    │         │
    │      (backbone partagé)
    │         │
    ├─→ delta_head:  Linear(256, 122) → delta_obs     état complet
    ├─→ reward_head: Linear(256,   1) → reward        scalaire
    ├─→ done_head:   Linear(256,   1) → done_logit    avant sigmoid
    └─→ hp_head:     Linear(256,   2) → delta_hp      [self, opp]

~130K paramètres
```

## Entrée

| Composant | Dims | Description |
|-----------|------|-------------|
| obs | 122 | Observation du GameState (perspective joueur courant) |
| action_onehot | 6 | One-hot de l'action (0=Pass, 1=AttachEnergy, 2-5=Attack0-3) |
| **Total** | **128** | Concaténation obs + action |

### Layout de l'observation (122 dims)

```
Dims    Bloc                         Contenu
────    ───────────────────────────  ──────────────────────────────
 0-10   self.pokemonType             one-hot 11 types
   11   self.HP                      normalisé /300
   12   self.MaxHP                   normalisé /300
13-23   self.weaknessType            one-hot 11 types
24-34   self.resistanceType          one-hot 11 types
   35   self.resistanceValue         normalisé /300
36-39   self.atkExists[4]            0 ou 1
40-43   self.atkDamage[4]            normalisé /300
44-47   self.atkTotalCost[4]         normalisé /10
48-58   self.energy[11]              compteur par type (raw)
   59   self.energyAttached          0 ou 1
60-119  opp.* (même layout)          miroir adversaire
  120   meta.Phase                   0=StartTurn, 0.5=Main, 1=GameOver
  121   meta.TurnIndex               normalisé /50
```

- **94 dims statiques** : ne changent jamais dans le POC (type, MaxHP, attaques, faiblesse, résistance)
- **28 dims dynamiques** : modifiables par le WM (HP, energy, energyAttached, Phase, TurnIndex)

## Sorties

### 4 têtes de sortie

| Tête | Dims | Activation | Usage training | Usage inférence C# |
|------|------|------------|----------------|---------------------|
| `delta_obs` | 122 | aucune (linéaire) | loss_delta (MSE pondérée) | Toutes dims SAUF HP |
| `reward` | 1 | aucune (linéaire) | loss_reward (MSE) | Reward prédit |
| `done_logit` | 1 | sigmoid en post | loss_done (BCE) | sigmoid > 0.5, puis réconciliation HP |
| `delta_hp` | 2 | aucune (linéaire) | loss_hp_aux + loss_attack_hp (Huber) | HP self et HP opp (remplace delta_obs[11] et delta_obs[71]) |

### Routage en inférence (CRITIQUE)

En inférence côté C#, les HP sont pris depuis `delta_hp`, pas depuis `delta_obs` :

```
Pour i dans [0..121]:
    si i == 11 (self.HP):  delta[i] = delta_hp[0]
    si i == 71 (opp.HP):   delta[i] = delta_hp[1]
    sinon:                 delta[i] = delta_obs[i]
```

Ce routage est contractuel : le backend C# doit l'implémenter exactement.

## Backbone

Deux couches fully-connected avec activation ReLU :

```python
self.shared = nn.Sequential(
    nn.Linear(128, 256),    # 128 * 256 + 256 = 33 024 params
    nn.ReLU(),
    nn.Linear(256, 256),    # 256 * 256 + 256 = 65 792 params
    nn.ReLU(),
)
```

Pas de LayerNorm, pas de Dropout, pas de connexion résiduelle dans la version actuelle.
Le backbone est partagé entre les 4 têtes — il extrait les features communes.

## Décompte des paramètres

| Composant | Paramètres |
|-----------|-----------|
| Linear(128, 256) + biais | 33 024 |
| Linear(256, 256) + biais | 65 792 |
| delta_head Linear(256, 122) + biais | 31 354 |
| reward_head Linear(256, 1) + biais | 257 |
| done_head Linear(256, 1) + biais | 257 |
| hp_head Linear(256, 2) + biais | 514 |
| **Total** | **~131 198** |

## Normalisation

### Entrée (dans Observe)

| Feature | Normalisation | Raison |
|---------|--------------|--------|
| HP, MaxHP, resistanceValue | /300 | Borne max raisonnable des HP du Base Set |
| atkDamage | /300 | Même échelle que les HP |
| atkTotalCost | /10 | Coût max raisonnable |
| TurnIndex | /50 | Parties rarement > 50 tours |
| Phase | 0, 0.5, 1 | 3 valeurs discrètes |
| energy | raw (compteur entier) | Pas normalisé, valeurs typiquement 0-10 |
| Tout le reste | 0 ou 1 | One-hot ou booléen |

### Sortie (delta)

Les deltas sont dans la même échelle que les entrées :
- `delta_hp = -0.1` signifie -30 HP en réel (0.1 × 300)
- `delta_energy = +1.0` signifie +1 énergie
- `delta_phase = +0.5` signifie StartTurn → Main

### Dénormalisation (dans WorldModelBackend.Step)

```
HP_réel = round(delta_hp * 300)
energy_réel = round(delta_energy)
phase_réel = nearest({0, 0.5, 1})
```

## Loss complète

```
loss = loss_delta
     + loss_reward
     + 0.5 * loss_done
     + 2.0 * loss_pass       (w_pass, configurable)
     + 5.0 * loss_ko
     + 5.0 * loss_hp_aux     (w_hp_aux, configurable)
     + 10.0 * loss_attack_hp (w_attack_hp, configurable)
```

| Terme | Type | Cible | Condition |
|-------|------|-------|-----------|
| loss_delta | MSE pondérée (×3 attaque, ×10 terminal, ×20 KO) | delta_obs complet | Toutes transitions |
| loss_reward | MSE | reward scalaire | Toutes transitions |
| loss_done | BCE | done binaire | Toutes transitions |
| loss_pass | MSE | delta_obs[HP_IDX] == 0 | action == Pass |
| loss_ko | ReLU | pred_hp_opp <= 0 | target HP opp <= 0 |
| loss_hp_aux | Huber (delta=1.0) | delta_hp[2] | Toutes transitions |
| loss_attack_hp | Huber (delta=1.0) | delta_hp[1] (opp seulement) | action >= 2 (attaques) |

## Training-only vs Inférence

| Aspect | Training | Inférence C# |
|--------|----------|--------------|
| Loss | Oui (7 termes) | Non |
| Sigmoid sur done_logit | Non (BCE le fait) | Oui (sigmoid > 0.5) |
| Réconciliation done | Non | Oui (done OR HP <= 0) |
| Routage HP | Non (les deux têtes sont supervisées) | Oui (HP depuis hp_head) |
| Clamp invariants | Non | Oui (HP >= 0, energy >= 0, etc.) |
| current_player update | Non | Oui (déterministe) |

## Export ONNX

```python
dummy = torch.randn(1, 128)
torch.onnx.export(model, dummy, "worldmodel.onnx",
    input_names=["input"],
    output_names=["delta_obs", "reward", "done_logit", "delta_hp"],
    dynamic_axes={"input": {0: "batch"}})
```

4 sorties nommées. Le backend C# lit chacune par nom.

## Évolutions possibles (hors scope actuel)

- LayerNorm ou BatchNorm entre les couches
- Connexion résiduelle (skip connection)
- Hidden plus large (512) si capacité insuffisante
- Têtes supplémentaires (énergie, phase) si nouvelles dérives apparaissent
- Embedding learnable pour evoFamilyId

## Auteurs

Q Humain, spécification produit, le 2026-02-16
Claude Opus 4.6 (Anthropic), rédaction, le 2026-02-16
Codex 5.3 (OpenAI), co-rédaction, le 2026-02-16
