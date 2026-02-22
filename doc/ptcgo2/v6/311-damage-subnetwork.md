# World Model v3.11 — Reducing cumul drift to < 0.1 HP/game

Objectif : passer de drift 2.22 HP/game (v3.10-F3-final) a < 0.1 HP/game.

## 1) Etat des lieux

### v3.10-F3-final (meilleur modele a ce jour)

| Metrique | Valeur | Seuil | Status |
|----------|--------|-------|--------|
| HP divergence MAE | 0.0518 | < 1.0 | OK |
| HP divergence on Pass | 0.0015 | < 0.02 | OK |
| KO miss rate | 0.0000 | < 1% | OK |
| Done F1 | 0.9889 | > 0.95 | OK |
| Reward MAE | 0.0147 | < 0.05 | OK |
| Cumul drift/game (HP) | **2.2218** | < 5.0 | OK |

Config : w_done=5.0, w_pass=5, w_hp_aux=5, w_attack_hp=10, attack_hp_loss=l1, 200 epochs, 100K games.

### Historique de la reduction du drift

| Version | Drift | Levier principal |
|---------|-------|-----------------|
| v2.1 | 35.05 | — (baseline) |
| v3.1 | 5.80 | Tete HP dediee + poids attaque |
| v3.5 | 7.57 | Parity fix (regression temporaire) |
| v3.8 | 7.24 | Loss duale Pass |
| v3.9-final | 1.64 | L1 attack HP (mais Done F1 = 0.79) |
| v3.10-F3-final | **2.22** | w_done=5.0 (recupere Done F1 = 0.99) |

L'ingenierie de loss a donne un gain de 16x (35 → 2.2). Les rendements sont decroissants. Le levier loss est epuise pour cette architecture.

### Source du drift residuel

Diagnostic v3.9 (valable pour v3.10, meme source d'erreur) :

| Action | Defender HP error | Count/game | Contribution drift |
|--------|------------------|------------|--------------------|
| Attack0 | ~1.34 HP | ~4.3 | ~5.4 HP/game |
| Attack1 | ~1.55 HP | ~1.3 | ~2.0 HP/game |
| Pass | 0.00 HP | ~9.7 | 0.0 |
| AttachEnergy | 0.00 HP | ~6.1 | 0.0 |

Distribution des erreurs sur les attaques :

```
0 HP:  39.3%  — correct
1 HP:  28.9%  — arrondi
2 HP:  14.3%
3 HP:   7.8%
4 HP:   3.9%
5+ HP:  5.8%  — erreurs significatives
```

### Pourquoi le drift residuel existe

Les degats dans le POC (Set de Base) sont **deterministes** :

```
dmg = attack.Damage          (fixe, parse de la carte)
si weakness:  dmg × 2
si resistance: dmg - resistance_value
clamp(dmg, 0, ∞)
```

Toute l'information est dans l'observation (atkDamage, pokemonType, weaknessType, resistanceType, resistanceValue). L'erreur est 100% eliminable en theorie.

Le goulot d'etranglement est l'**interaction weakness** : le MLP 2×256 doit apprendre `dot(selfType[0:10], oppWeakness[73:83]) → damage × 2`. C'est un produit de features distantes a travers des couches ReLU — faisable mais difficile en pratique avec cette capacite.

### Ce que drift < 0.1 exige

~21 steps/game, ~5 attaques/game. Pour drift < 0.1 :
- Erreur moyenne par attaque < 0.02 HP
- **~98% des attaques doivent etre exactes** (0 HP d'erreur)
- Actuellement 39% exactes → besoin de 2.5x ce taux

## 2) Pistes d'amelioration

### Piste A — Backbone plus large/profond

Principe :
- Passer de 2×256 (~131K params) a 3×512 ou 4×256 avec residual + LayerNorm
- ~500K params, toujours rapide sur MPS

Interet :
- Plus de capacite pour les interactions cross-features
- Pas de changement d'interface ONNX
- Implementation simple

Limite :
- Ne resout pas le probleme structurel : les interactions multiplicatives (weakness) restent implicites
- Gain attendu modere

Estimation de drift : **0.5 – 1.5 HP/game**

```python
class WorldModelNet(nn.Module):
    def __init__(self, hidden=512, n_layers=3):
        super().__init__()
        layers = [nn.Linear(INPUT_DIM, hidden), nn.LayerNorm(hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.ReLU()]
        self.shared = nn.Sequential(*layers)
        # ... memes tetes
```

### Piste B — Sous-reseau de degats (damage sub-network)

Principe :
- Ajouter un module dedie qui recoit uniquement les features pertinentes au calcul de degats
- Ce module apprend la formule de degats (weakness, resistance) avec un fort biais inductif
- Sa sortie alimente la tete HP principale

Features d'entree du sous-reseau (38 dims) :

| Feature | Dims | Source obs |
|---------|------|-----------|
| selfPokemonType | 11 | obs[0:11] |
| selfAtkDamage | 4 | obs[40:44] |
| oppWeakness | 11 | obs[73:84] |
| oppResistanceType | 11 | obs[84:95] |
| oppResistanceValue | 1 | obs[95] |

Architecture :

```python
class DamageEstimator(nn.Module):
    """Learns damage formula from type/attack features."""
    def __init__(self, hidden=64):
        super().__init__()
        # 38 = selfType(11) + atkDmg(4) + oppWeak(11) + oppRes(11) + oppResVal(1)
        self.net = nn.Sequential(
            nn.Linear(38, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),  # estimated normalized damage
        )

    def forward(self, damage_features):
        return self.net(damage_features)  # [B, 1]


class WorldModelNet(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.damage_est = DamageEstimator(hidden=64)

        # Input: obs(122) + action_onehot(6) + damage_estimate(1) = 129
        self.shared = nn.Sequential(
            nn.Linear(INPUT_DIM + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.delta_head = nn.Linear(hidden, OBS_DIM)
        self.reward_head = nn.Linear(hidden, 1)
        self.done_head = nn.Linear(hidden, 1)
        self.hp_head = nn.Linear(hidden, 2)

    def forward(self, x):
        obs = x[:, :OBS_DIM]  # [B, 122]
        # Extract damage-relevant features
        damage_feats = torch.cat([
            obs[:, 0:11],    # selfPokemonType
            obs[:, 40:44],   # selfAtkDamage[0:4]
            obs[:, 73:84],   # oppWeakness
            obs[:, 84:95],   # oppResistanceType
            obs[:, 95:96],   # oppResistanceValue
        ], dim=1)  # [B, 38]

        dmg_est = self.damage_est(damage_feats)  # [B, 1]
        h = self.shared(torch.cat([x, dmg_est], dim=1))
        return (
            self.delta_head(h),
            self.reward_head(h),
            self.done_head(h),
            self.hp_head(h),
        )
```

Interet :
- Biais inductif fort : le sous-reseau voit exactement les features de la formule de degats
- Le petit MLP 2×64 a largement la capacite d'apprendre la table de degats du Set de Base
- Pas de hard-coding : les regles sont apprises, pas imposees
- L'interface ONNX ne change pas (meme entree, memes sorties)
- Respecte la contrainte "pas de post-processing C#"

Limite :
- Necessite de connaitre les indices des features pertinentes (deja documentes dans feature_layout.py)
- Ajoute ~5K params (negligeable)

Variante — **supervision directe du sous-reseau** : ajouter une loss auxiliaire qui compare la sortie du DamageEstimator au delta HP reel sur les transitions d'attaque. Cela force le sous-reseau a apprendre exactement la table de degats.

```python
# Loss auxiliaire pour le damage estimator
if is_attack.any():
    dmg_est_atk = dmg_est[is_attack, 0]
    target_dmg = -batch_delta[is_attack, OPP_HP_IDX]  # degats = -delta HP opp
    loss_dmg_est = (dmg_est_atk - target_dmg).abs().mean()
```

Estimation de drift : **0.05 – 0.5 HP/game**

### Piste C — Classification discrete des degats

Principe :
- Le Set de Base a un ensemble fini de degats possibles
- Au lieu de regression continue → classification sur ~20 classes

Degats uniques dans le Set de Base :

```
0, 10, 20, 30, 40, 50, 60, 80, 100    (degats de base)
× {1, 2} pour weakness                  (×2)
- {0, 30} pour resistance               (-30)
= ~20 valeurs uniques apres clamp ≥ 0
```

Une tete de classification avec softmax donnerait des predictions **exactes** (pas d'erreur d'arrondi).

```python
# Damage classes pour le Set de Base
DAMAGE_CLASSES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 100, 120, 160, 200]
# Normalises /300 pour correspondre a delta_hp
DAMAGE_BINS = [d / 300.0 for d in DAMAGE_CLASSES]

class DamageClassifier(nn.Module):
    def __init__(self, hidden=64, n_classes=13):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(38, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_classes),
        )
    def forward(self, x):
        return self.net(x)  # logits [B, n_classes]
```

Interet :
- Precision parfaite si la classification est correcte
- Cross-entropy loss, pas de probleme d'arrondi

Limite :
- **Rejete pour scalabilite** dans worldmodel_v3.md : chaque extension/regle ajoute de nouvelles valeurs de degats
- Viable uniquement pour le POC (Set de Base)
- Necessite un mapping explicite degats → classes

Estimation de drift : **0.0 – 0.05 HP/game** (potentiellement parfait)

### Combinaisons recommandees

| Combinaison | Pistes | Drift attendu | Complexite |
|-------------|--------|---------------|------------|
| Conservative | A seule | 0.5 – 1.5 | Faible |
| **Recommandee** | **A + B** | **0.05 – 0.3** | Moyenne |
| Aggressive | A + B + supervision directe | 0.02 – 0.1 | Moyenne |
| POC-only | A + C | ~0.0 | Moyenne (non scalable) |

## 3) Leviers complementaires (orthogonaux a l'architecture)

### Plus de donnees

- 100K → 300K+ games : meilleure couverture des paires rares (weakness)
- Bot d'attaque (doc/ptcgo2/data_attack_random_bot.md si existant) : enrichit les transitions d'attaque

### Plus d'entrainement

- 200 → 400+ epochs avec warmup + cosine
- Reduire le LR final du cosine schedule (actuellement → 0, peut etre trop agressif)

### Ponderation par difficulte

- Upweight les paires (attaquant, defenseur) qui ont les plus grosses erreurs
- Curriculum learning : d'abord les cas simples (pas de weakness), puis les cas avec weakness

## 4) Plan d'experimentation propose

### Phase 1 — Backbone elargi (1 run, ~3h)

| Run | Architecture | Params | Notes |
|-----|-------------|--------|-------|
| G1 | 3×512 + LayerNorm | ~530K | Baseline architecture amelioree |

Meme config que v3.10-F3 (w_done=5, w_attack_hp=10, l1). Mesure l'apport de la capacite seule.

### Phase 2 — Damage sub-network (2 runs, ~6h)

| Run | Architecture | DamageEstimator | Supervision directe | Notes |
|-----|-------------|-----------------|--------------------|----|
| G2 | 2×256 + DamageEst | 2×64 | Non | Sous-reseau en feature, backbone inchange |
| G3 | 2×256 + DamageEst | 2×64 | Oui (L1, w=10) | Sous-reseau supervise directement |

### Phase 3 — Combinaison (1 run, ~3h)

| Run | Architecture | DamageEstimator | Supervision | Notes |
|-----|-------------|-----------------|------------|-------|
| G4 | 3×512 + LN + DamageEst | 2×64 | Oui | Meilleure combinaison de Phase 1+2 |

### Phase 4 — Plus de donnees (optionnel, 1 run, ~8h)

Si G4 est entre 0.1 et 0.3 :
- Regenerer 300K games
- Retrain avec la meilleure architecture

## 5) Criteres d'acceptation

### Objectif

| Metrique | v3.10-F3 | Cible v3.11 |
|----------|----------|-------------|
| Cumul drift/game (HP) | 2.22 | **< 0.1** |
| % attaques exactes (0 HP erreur) | ~39% | **> 95%** |

### Garde-fous (pas de regression)

| Metrique | Seuil |
|----------|-------|
| Done F1 | > 0.95 |
| Pass HP | < 0.02 |
| KO miss rate | 0 |
| Reward MAE | < 0.05 |

### Decision

Si aucun run ne passe sous 0.1 :
- Accepter le meilleur resultat obtenu (probablement 0.1 – 0.3)
- Evaluer si la piste C (classification) est acceptable pour le POC
- Ou accepter le drift comme limite de l'approche regression pour le Set de Base

## 6) Fichiers a modifier

| Fichier | Changement |
|---------|-----------|
| `training/model.py` | DamageEstimator, architecture paramétrable (hidden, n_layers, layernorm) |
| `training/train.py` | Loss damage_est auxiliaire, args architecture |
| `training/feature_layout.py` | Constantes pour les indices damage-relevant |
| `training/export_onnx.py` | Inchange (meme interface ONNX) |
| Aucun changement C# | Meme modele ONNX, memes sorties |

## Auteurs

Q Humain, specification produit, le 2026-02-20
Claude Opus 4.6 (Anthropic), redaction, le 2026-02-20