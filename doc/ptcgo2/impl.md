# Implémentation POC — Backend C# + UI (Étape 1)

Ce document décrit les choix d'implémentation pour la première étape du POC: construire un backend C# jouable par un humain via UI terminal. Le world model n'est pas un joueur; il sera ajouté ensuite comme backend alternatif.

## Objectif Étape 1

Livrer une boucle de jeu complète et testable avec:
- règles POC minimales (réf: `doc/ptcgo2/poc.md`)
- backend C# déterministe
- UI console pour humain
- séparation stricte `Policy` (choix d'action) vs `TransitionBackend` (simulation)
- protocole JSON unifié (même format pour humain et IA)

Critère de sortie Étape 1: un humain ET une IA terminent une partie complète via le protocole JSON, avec logs reproductibles.

## Contrat d'architecture

Règle non négociable: le backend ne choisit jamais l'action.

```text
Policy (Human/Random/AI) -> ActionId
TransitionBackend.Step(state, action) -> nextState, reward, done, info
```

Interfaces C# proposées:

```csharp
public interface IActionProvider
{
    int SelectAction(in GameState state, ReadOnlySpan<int> legalActions);
}

public interface ITransitionBackend
{
    StepResult Step(in GameState state, int actionId);
    ReadOnlySpan<int> GetLegalActions(in GameState state);
}
```

Nommage à respecter partout:
- `ActionProvider` = joueur (policy)
- `TransitionBackend` = simulateur d'environnement (moteur C# puis world model)

Note: pas de `rngSeed` dans `Step` — le POC est entièrement déterministe (pas de coin flip, pas de dégâts aléatoires).

## Protocole JSON unifié

Le moteur de match expose un protocole JSON sur stdin/stdout. C'est le contrat unique entre le moteur et toute policy (humaine ou IA).

### Émission (moteur → policy) sur stdout, une ligne JSON par step :

```json
{
  "turn": 5,
  "current_player": 0,
  "phase": "Main",
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
        {"id": 2, "name": "Gnaw", "damage": 10, "cost": "C", "legal": true},
        {"id": 3, "name": "Thunder Jolt", "damage": 30, "cost": "EC", "legal": false, "reason": "ILL04"}
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
  "last_result": null,
  "done": false
}
```

### Réception (policy → moteur) sur stdin :

```json
{"action": 2}
```

### Après exécution, `last_result` est rempli dans le prochain état :

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

En cas d'infraction :

```json
{
  "last_result": {
    "action": 1,
    "action_name": "AttachEnergy",
    "damage": 0,
    "weakness": false,
    "infraction": {"code": "ILL02", "message": "Double attache énergie"}
  }
}
```

### Adaptateur humain (ConsoleRenderer)

Transforme le JSON en affichage lisible dans le terminal. L'humain tape un numéro, l'adaptateur émet `{"action": N}` sur le même protocole. L'adaptateur est un client du protocole, pas une exception.

### Adaptateur IA

L'IA reçoit le JSON brut et répond `{"action": N}`. Aucun adaptateur spécifique nécessaire.

## Scope fonctionnel Étape 1

- 1 Pokémon actif par joueur
- 6 actions: `Pass`, `AttachEnergy`, `Attack0..3`
- énergie basique auto (pool infini typé)
- pas de banc, pas d'évolution, pas de statuts, pas de dresseurs
- victoire par KO (HP <= 0)

## CardRegistry POC — Basic Base Set uniquement

Pokémon Basic du Base Set sans évolution ou dont on ignore la lignée :

```
#     Nom             Type        HP    Faiblesse    Résistance    Attaques
───   ──────────────  ──────────  ───   ──────────   ──────────    ─────────────────────────────
 1    Pikachu         ⚡ Electric  40    Fighting     —             Gnaw (10, C), Thunder Jolt (30, EC)
 2    Electabuzz      ⚡ Electric  70    Fighting     —             Thundershock (10, C), Thunderpunch (30, ECC)
 3    Magnemite       ⚡ Electric  40    Fighting     —             Thunder Wave (10, EC), Selfdestruct (40, ECC)
 4    Voltorb         ⚡ Electric  40    Fighting     —             Tackle (10, C)
 5    Vulpix          🔥 Fire      50    Water        —             Confuse Ray (10, FC), Fire Blast (30, FFC)
 6    Ponyta          🔥 Fire      40    Water        —             Smash Kick (20, CC), Flame Tail (30, FCC)
 7    Growlithe       🔥 Fire      60    Water        —             Flare (20, FC)
 8    Magmar          🔥 Fire      50    Water        —             Fire Punch (30, FFC)
 9    Staryu          💧 Water     40    Electric     —             Slap (20, WC)
10    Seel            💧 Water     60    Electric     —             Headbutt (10, W)
11    Squirtle        💧 Water     40    Electric     —             Bubble (10, W), Withdraw (0, WC)
12    Tangela         🌿 Grass     50    Fire         —             Bind (20, GCC), Poisonpowder (20, GGC)
13    Bulbasaur       🌿 Grass     40    Fire         —             Leech Seed (20, GG)
14    Sandshrew       ⚔ Fighting  40    Grass        —             Sand-attack (10, FC)
15    Machop          ⚔ Fighting  50    Psychic      —             Low Kick (20, FC)
16    Onix            ⚔ Fighting  90    Grass        —             Rock Throw (10, FC), Harden (0, FF)
17    Gastly          👻 Psychic   30    —            Fighting      Sleeping Gas (10, P), Destiny Bond (0, PC)
18    Abra            👻 Psychic   30    Psychic      —             Psyshock (10, P)
19    Drowzee         👻 Psychic   50    Psychic      —             Pound (10, C), Confuse Ray (10, PC)
20    Jynx            👻 Psychic   70    Psychic      —             Doubleslap (10, PC), Meditate (20, PPC)
21    Rattata         ⬜ Normal    30    Fighting     —             Bite (20, C)
22    Doduo           ⬜ Normal    50    Electric     —             Fury Attack (10, CC)
23    Farfetch'd      ⬜ Normal    50    Electric     Resist(F-30)  Leek Slap (30, CC), Pot Smash (30, CCC)
```

Note: les dégâts et coûts sont simplifiés. Les effets secondaires (paralysie, confusion, self-damage, etc.) sont ignorés dans le POC — seuls les dégâts bruts sont appliqués.

## Modèle de données minimal

`GameState` minimal Étape 1:
- `CurrentPlayer` (0/1)
- `TurnIndex`
- `Phase` (StartTurn, Main, GameOver)
- `Player[2].CardIndex`
- `Player[2].HP`, `Player[2].MaxHP`
- `Player[2].Energy[11]`
- `Player[2].EnergyAttachedThisTurn` (bool)
- `Player[2].PendingEnergy` (int, type de l'énergie reçue en début de tour)

`StepResult`:
- `NextState`
- `RewardCurrentPlayer` (float)
- `Done`
- `Infraction` (null si action légale, sinon code + message)

## Règles de simulation (ordre exact)

Dans `Step`:
0. Si `Phase == StartTurn`: auto-grant 1 énergie du type du Pokémon courant dans `PendingEnergy`; passer en `Phase = Main`.
1. Vérifier légalité de `actionId` (voir table d'infractions).
2. Si action illégale: état inchangé, reward = -0.01, tour conservé, retourner `Infraction`.
3. Si `AttachEnergy`: transférer `PendingEnergy` dans `Energy[]`; poser `EnergyAttachedThisTurn = true`.
4. Si `AttackN`: vérifier `atkExists[N]` et coût; calculer dégâts avec faiblesse/résistance; appliquer HP adverse.
5. Si HP adverse <= 0: `Phase = GameOver`, `Done = true`, reward terminal (+1/-1).
6. Si `Pass` ou après `AttackN`: fin de tour → reset `EnergyAttachedThisTurn`; switch `CurrentPlayer`; `TurnIndex++`; `Phase = StartTurn`.

## Infractions (Option A — verrouillée)

L'UI autorise toute saisie. Le backend accepte toute action et applique la règle d'infraction si illégale. L'état ne change pas, le tour est conservé, reward = -0.01.

```
Code    Infraction                    Condition
─────   ────────────────────────────  ────────────────────────────────────
ILL01   Action hors plage             actionId < 0 ou actionId >= 6
ILL02   Double attache énergie        AttachEnergy alors que EnergyAttachedThisTurn == true
ILL03   Attaque inexistante           AttackN alors que atkExists[N] == false
ILL04   Coût énergie insuffisant      AttackN alors que effectiveEnergy < atkTotalCost[N]
ILL05   Action après fin de partie    toute action alors que Phase == GameOver
```

Chaque infraction est loggée avec son code dans le JSONL.

## Structure de code

```text
ptcgo2/
  src/
    Ptcgo2.Core/
      GameState.cs
      StepResult.cs
      ActionId.cs
      CardRegistry.cs
      Legality.cs
      Damage.cs
      JsonProtocol.cs
    Ptcgo2.Console/
      Program.cs
      ConsoleRenderer.cs
      HumanConsolePolicy.cs
      RandomPolicy.cs
      MatchLogger.cs
  tests/
    Ptcgo2.Tests/
```

## Logging et reproductibilité

Chaque match produit un `.jsonl` (une ligne JSON par événement) :
- ligne 1 : `{"event": "init", "seed": 42, "state": {...}}`
- lignes suivantes : `{"event": "step", "turn": N, "action": M, "infraction": null, "state": {...}, "reward": 0.0, "done": false}`
- dernière ligne : `{"event": "end", "winner": 0, "turns": N}`

Règles:
- seed explicite obligatoire
- replay déterministe d'un log = même résultat final

## Plan de livraison

Milestone 1: Core + backend C# + protocole JSON
- implémenter `GameState`, `StepResult`, `ActionId`, `CardRegistry`
- implémenter `CSharpBackend.Step` + légalité + infractions
- implémenter `JsonProtocol` (sérialisation état → JSON, parsing action ← JSON)
- tests unitaires règles de base

Milestone 2: UI console humain + IA
- adaptateur console (JSON → affichage lisible, saisie → JSON)
- boucle match complète
- mode `--human` (adaptateur console) et mode `--json` (stdin/stdout brut pour IA)

Milestone 3: logs + replay
- logger jsonl
- commande `replay --log <file>`
- test de déterminisme

## Tests minimum (gating)

- `AttachEnergy` 2 fois dans le même tour → infraction ILL02
- `AttackN` avec coût insuffisant → infraction ILL04
- `AttackN` avec attaque inexistante → infraction ILL03
- faiblesse ×2 et résistance -30 appliquées correctement
- KO termine la partie (`Phase = GameOver`) et fige l'état
- action après GameOver → infraction ILL05
- même seed + mêmes actions → mêmes états
- UI accepte toute saisie sans crasher
- protocole JSON parsable par un client externe (test round-trip)

## Hors scope Étape 1

- backend world model
- policy IA entraînée
- mode dual backend
- deck/main/pioche réelle
- évolutions, banc, statuts, dresseurs, outils
- effets d'attaque (paralysie, confusion, défausse énergie, etc.)

## Étape 2 (prévue)

Ajouter un second `ITransitionBackend`: `WorldModelBackend`.
Le reste ne change pas: mêmes `ActionProvider`, même protocole JSON, mêmes logs. L'utilisateur choisit simplement le backend au lancement.

## Auteurs

Q Humain, spécification produit, le 2026-02-14
Claude Opus 4.6 (Anthropic), co-rédaction, le 2026-02-14
Codex 5.3 (OpenAI), co-rédaction, le 2026-02-14
