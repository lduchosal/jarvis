# Board — 6 slots par joueur

## Actuel : 14 dims/slot

```
cardIndex (1)  ← int brut 0–19012, inutile pour le WM
HP (1)
MaxHP (1)
Energy[11] (11)
```

## Problèmes

1. `cardIndex` est un identifiant catégoriel casté en float. Le WM ne sait pas :
   - Si c'est un basic ou un stage 2
   - Son type (Fire, Water...)
   - Ses attaques, faiblesses, résistances, coût de retraite

2. Les catégories (type, stade, faiblesse, résistance) en scalaire brut injectent un faux ordre numérique (ex: Fire=1 "plus proche" de Grass=2 que de Darkness=8).

3. `energy[11]` en simple compteur par type perd des informations critiques :
   - Nature de l'énergie (basic vs special)
   - Multiplicité (ex: Double Turbo fournit 2 énergies)
   - Effets continus qui modifient dégâts/règles

## Proposé : 49 dims/slot (v2.1)

```
Dims  Feature                  Source                           Notes
────  ───────────────────────  ───────────────────────────────  ──────────────────────────
  1   occupied                 boardPokemon != -1               0/1
  3   evoStageOH[3]            EvolutionStage                   one-hot (basic/stg1/stg2)
  1   evoFamilyId              CardEvolutionFamily              id de lignée normalisé
 11   pokemonTypeOH[11]        CardPokemonType                  one-hot (11 types)
  1   HP                       BoardHP (runtime)                normalisé /300
  1   MaxHP                    BoardMaxHP (runtime)             normalisé /300
 11   weaknessTypeOH[11]       CardWeaknessType                 one-hot (11 types)
 11   resistanceTypeOH[11]     CardResistanceType               one-hot (11 types)
  1   resistanceValue          CardResistanceValue              normalisé (typ. -30)
  1   retreatCost              CardRetreatCost                  normalisé /4
  4   atkExists[4]             AttackExists                     0/1
  4   atkDamage[4]             AttackBaseDamage                 normalisé /200
  4   atkTotalCost[4]          AttackTotalCost                  normalisé /5
 11   effectiveEnergy[11]      Runtime agrégé                   total payable par type
  1   specialEnergyCount       Runtime agrégé                   nb cartes énergie spéciale
  1   specialProvidesExtra     Runtime agrégé                   énergie bonus totale
  1   specialDamageMod         Runtime agrégé                   malus dégâts (ex: -20 DTE)
────
 49
```

### Notes de conception

- **One-hot catégoriel** : `evoStage`, `pokemonType`, `weaknessType`, `resistanceType` passent en one-hot pour éliminer le biais ordinal. Le coût en dims (+27) est justifié par la suppression d'un signal trompeur.
- **Famille d'évolution explicite** : `evoFamilyId` rend visible la contrainte de légalité d'évolution (ex: Charmander -> Charmeleon autorisé, Charmander -> Wartortle interdit). Ce champ doit aussi exister côté main pour que le WM puisse apprendre l'appariement board/main.
- **Attaques conservées** : `atkExists`, `atkDamage`, `atkTotalCost` (12 dims) sont indispensables pour que le WM prédise les dégâts infligés.
- **Énergie en deux niveaux** :
  - `effectiveEnergy[11]` = total d'énergie payable par type, toutes sources confondues (basiques + contribution des spéciales). C'est ce qui détermine si un coût d'attaque est satisfait.
  - Le bloc spécial (3 dims) capture les propriétés agrégées des énergies spéciales attachées : combien il y en a, combien d'énergie bonus elles fournissent, et le malus de dégâts cumulé.
- **Normalisation** : bornes fixes (HP/300, dégâts/200, coût/5, retraite/4) pour stabilité entre sets.
- **Masquage** : quand `occupied = 0`, toutes les features sont multipliées par 0.

Toutes les features statiques viennent du `CardRegistry` via `index_select(0, cardIdx.clamp(0))` — lookup tensoriel, pas de boucle.

## Comparaison v1 → v2.1

```
                          v1 (14)    v1-proposé (32)    v2.1 (49)
──────────────────────    ───────    ───────────────    ───────
cardIndex brut              1            —                —
occupied                    —            1                1
evoStage                    —            1 (scalaire)     3 (OH)
evoFamilyId                 —            —                1
pokemonType                 —            1 (scalaire)    11 (OH)
HP                          1            1                1
MaxHP                       1            1                1
weaknessType                —            1 (scalaire)    11 (OH)
resistanceType              —            1 (scalaire)    11 (OH)
resistanceValue             —            1                1
retreatCost                 —            1                1
atkExists[4]                —            4                4
atkDamage[4]                —            4                4
atkTotalCost[4]             —            4                4
energy[11] (compteur)      11           11                —
effectiveEnergy[11]         —            —               11
specialEnergyCount          —            —                1
specialProvidesExtra        —            —                1
specialDamageMod            —            —                1
──────────────────────    ───────    ───────────────    ───────
Total                      14           32               49
```

## Total board

12 slots × 49 dims = **588 dims** (actuel : 168)
