# POC — World Model Pokémon TCG (scope minimal)

Prototype de validation du pipeline complet : moteur tensorisé, collecte de transitions, entraînement d'un world model MLP. Scope réduit au strict minimum pour prouver que l'architecture fonctionne avant d'ajouter de la complexité.

## Distinction fondamentale : Backend vs Policy

```
┌──────────────────────────────────────────────────────────┐
│                                                          │
│   Policy (qui choisit l'action)                          │
│   ─────────────────────────────                          │
│   Humain, IA (Claude/Codex), Random                      │
│   → décide quelle action jouer                           │
│                                                          │
│   Backend (qui simule les conséquences)                   │
│   ────────────────────────────────────                    │
│   Moteur C# (vérité terrain) ou World Model (appris)     │
│   → prédit le prochain état après l'action                │
│                                                          │
│   Le world model est un BACKEND, pas un joueur.           │
│   Il remplace le moteur C#, pas la décision humaine/IA.  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

Le world model apprend à simuler l'environnement : étant donné un état et une action, il prédit l'état suivant. Il ne choisit jamais quelle action jouer. Cette responsabilité appartient à la policy (humain ou IA).

## Règles du POC

- 1 seul Pokémon actif par joueur, pas de banc
- Basic uniquement, pas d'évolution
- Énergies basiques uniquement, pas de spéciales
- Attaques = dégâts bruts uniquement, pas d'effets secondaires
- Pas de cartes Dresseur / Supporter
- Pas d'outils (Pokémon Tools)
- Pas de statuts (poison, brûlure, paralysie, sommeil, confusion)
- Pas de retraite / switch (pas de banc)
- Condition de victoire : KO du Pokémon adverse (HP <= 0)

## Actions (espace réduit)

```
ID    Action              Description
───   ──────────────────  ─────────────────────────────────
  0   Pass                fin de tour
  1   AttachEnergy        attacher 1 énergie basique (1/tour)
  2   Attack0             utiliser attaque 0
  3   Attack1             utiliser attaque 1
  4   Attack2             utiliser attaque 2
  5   Attack3             utiliser attaque 3
───
  6   total
```

Légalité : AttachEnergy uniquement si pas encore fait ce tour. Attack N uniquement si atkExists[N] = 1 et effectiveEnergy satisfait atkTotalCost[N].

## Tour de jeu

```
1. Piocher 1 énergie (automatique, pool infini typé selon le Pokémon)
2. Optionnel : AttachEnergy (attacher l'énergie piochée)
3. Optionnel : Attack (si coût satisfait)
4. Pass → fin de tour, switch de joueur
```

Note : la pioche d'énergie est simplifiée (pas de deck). Chaque tour, le joueur reçoit automatiquement 1 énergie du type de son Pokémon. Cela évite de modéliser le deck et la main pour le POC.

## Board — 1 slot par joueur

### GameState canonique (par slot)

```
Champ              Type      Notes
─────────────────  ────────  ──────────────────────
cardIndex          int64     identifiant carte (affichage/debug)
HP                 int32     HP courants
MaxHP              int32     HP max
energy[11]         int32     énergies attachées par type
energyAttached     bool      flag : déjà attaché ce tour
```

### Observation WM (par slot)

```
Dims  Feature                  Source                   Notes
────  ───────────────────────  ───────────────────────  ──────────────────────
 11   pokemonTypeOH[11]        CardPokemonType          one-hot (11 types)
  1   HP                       BoardHP (runtime)        normalisé /300
  1   MaxHP                    BoardMaxHP (runtime)     normalisé /300
 11   weaknessTypeOH[11]       CardWeaknessType         one-hot (11 types)
 11   resistanceTypeOH[11]     CardResistanceType       one-hot (11 types)
  1   resistanceValue          CardResistanceValue      normalisé
  4   atkExists[4]             AttackExists             0/1
  4   atkDamage[4]             AttackBaseDamage         normalisé /200
  4   atkTotalCost[4]          AttackTotalCost          normalisé /5
 11   effectiveEnergy[11]      BoardEnergy (runtime)    énergies attachées par type
  1   energyAttached           Runtime                  flag 0/1
────
 60
```

### Total observation

```
Composant                     Dims
────────────────────────────  ────
Joueur courant (slot)           60
Adversaire (slot)               60
Méta (phase, tour)               2
────────────────────────────  ────
Total                          122
```

## Réseau WorldModelNet (POC)

```
Input: [B, 128] = cat(obs[122], action_one_hot[6])
  │
  ├─→ Linear(128, 256) → LayerNorm → ReLU
  ├─→ Linear(256, 256) → LayerNorm → ReLU  (+ residual)
  │
  ├─→ Linear(256, 122) ──→ delta_obs
  ├─→ Linear(256, 1)   ──→ reward
  └─→ Linear(256, 1)   ──→ done_logit

~200K paramètres
```

Le WorldModelNet est un **backend appris** : il approxime la fonction `GameEnv.Step()`. Il prend (état, action) et prédit (état suivant, reward, done). Il ne choisit pas d'action.

## Pipeline

```
CardRegistry (Basic only)
       │
   GameEnv (règles simplifiées, B=256)
       │
   TrajectoryCollector (random play)
       │
   TransitionDataset (delta + one-hot action)
       │
   WorldModelNet (~200K params)
       │
   Wandb (loss, per-component MSE, done F1)
```

## UI — Jeu interactif

### Protocole JSON unifié

Le moteur expose un protocole JSON unique sur stdin/stdout. Toute policy (humaine ou IA) consomme le même format.

A chaque step, le moteur émet sur stdout :

```json
{
  "turn": 5,
  "current_player": 0,
  "backend": "csharp",
  "board": [
    {
      "player": 0,
      "name": "Pikachu",
      "card_index": 1,
      "hp": 40,
      "max_hp": 60,
      "type": "Electric",
      "energy": {"Electric": 3},
      "attacks": [
        {"id": 2, "name": "Gnaw", "damage": 10, "cost": "C"},
        {"id": 3, "name": "Thunder Jolt", "damage": 30, "cost": "EC"}
      ]
    },
    {
      "player": 1,
      "name": "Vulpix",
      "card_index": 5,
      "hp": 50,
      "max_hp": 50,
      "type": "Fire",
      "energy": {"Fire": 2},
      "attacks": [
        {"id": 2, "name": "Confuse Ray", "damage": 10, "cost": "FC"},
        {"id": 3, "name": "Fire Blast", "damage": 30, "cost": "FFC"}
      ]
    }
  ],
  "actions": [
    {"id": 0, "name": "Pass", "legal": true},
    {"id": 1, "name": "AttachEnergy", "legal": false, "reason": "ILL02"},
    {"id": 2, "name": "Gnaw (10 dmg)", "legal": true},
    {"id": 3, "name": "Thunder Jolt (30 dmg)", "legal": false, "reason": "ILL04"}
  ],
  "last_result": null
}
```

La policy répond sur stdin :

```json
{"action": 2}
```

Le moteur exécute le step et renvoie le prochain état avec `last_result` rempli :

```json
{
  "last_result": {
    "action": 2,
    "action_name": "Gnaw",
    "damage": 20,
    "weakness": true,
    "infraction": null
  }
}
```

### Adaptateur humain

L'adaptateur console transforme le JSON en affichage lisible :

```
═══════════════════════════════════════
  Tour 5 — Joueur 1          [C# engine]
═══════════════════════════════════════
  [Pikachu]  HP 40/60   ⚡⚡⚡
  vs
  [Vulpix]   HP 50/50   🔥🔥
═══════════════════════════════════════
  0. Passer
  1. Attacher énergie          [ILLEGAL: déjà attaché]
  2. Gnaw (10 dmg, coût: C)
  3. Thunder Jolt (30 dmg)     [ILLEGAL: énergie insuffisante]
  > _
```

L'humain tape un numéro, l'adaptateur émet `{"action": N}`.

### Adaptateur IA

L'IA (Claude/Codex) reçoit le JSON brut et répond `{"action": N}`. Pas d'adaptateur nécessaire.

### Modes de jeu

L'UI combine une **policy** (qui choisit) et un **backend** (qui simule) :

```
Policy (joueur 1)     Policy (joueur 2)     Backend
──────────────────    ──────────────────    ────────────────
Humain                Humain                Moteur C#
Humain                IA (Claude/Codex)     Moteur C#
Humain                Random                World Model
IA                    IA                    Moteur C#
...                   ...                   ...
```

Toute combinaison policy × policy × backend est possible.

### Mode dual backend

Même partie, mêmes actions, exécutées en parallèle sur le moteur C# et le world model. Affiche les divergences de prédiction pour mesurer la qualité du WM.

## Critère de succès

- Le pipeline tourne de bout en bout sans erreur
- La loss converge sur 50 époques
- Le WM prédit correctement : dégâts (avec faiblesse/résistance), variation d'énergie, game over
- La prédiction de done a un F1 > 0.8
- Un humain peut jouer une partie complète via l'UI console
- Une IA peut jouer une partie complète via le protocole JSON stdin/stdout

## Chemin vers la version complète

```
POC                          → V1
─────────────────────────    ──────────────────────────
1 Pokémon actif              → 1 actif + 5 banc
pas d'évolution              → Basic/Stage1/Stage2
énergie basique auto         → deck + main + pioche
pas d'effets                 → effets d'attaque
pas de Dresseur              → Dresseur/Supporter
pas de statuts               → poison, brûlure, etc.
pas d'outils                 → Pokémon Tools
6 actions                    → 520 actions
~200K params                 → ~3.3M params
```

## Auteurs

Q Humain, le 2026-02-14
Claude Opus 4.6 (Anthropic), rédaction, le 2026-02-14
Codex 5.3 (OpenAI), co-rédaction, le 2026-02-14
