# POC v5 — Tournois

## Contexte

La v4 a validé le world model sur des matchs isolés (benchmark PASS sur 200K games). La v5 introduit une couche au-dessus du match : le **tournoi à élimination directe**, implémenté uniquement dans le CLI avec le moteur C#. L'objectif est de tester si le choix de carte (la seule décision stratégique du POC) a un impact mesurable sur le taux de victoire en tournoi.

## Règles du tournoi

### Format

- **Élimination directe** (single elimination bracket)
- **Best of 1** : chaque round = 1 match, le perdant est éliminé
- **N joueurs** : puissance de 2 (8, 16, 32, 64...)
- L'arbre est construit aléatoirement (tirage au sort des positions)

### Sélection de carte

- Chaque joueur **choisit 1 carte** en début de tournoi
- La carte est **fixe** pour toute la durée du tournoi (même carte à chaque round)
- Le choix est fait **à l'aveugle** (le joueur ne connaît pas les cartes adverses)

### Déroulement

```
Phase 1 — Inscription + sélection de carte
  Chaque joueur choisit sa carte (parmi les 69 Basic du Set de Base)

Phase 2 — Bracket
  Round 1 : N/2 matchs en parallèle
  Round 2 : N/4 matchs (vainqueurs du round 1)
  ...
  Finale : 1 match

Phase 3 — Résultat
  Classement final (vainqueur, finaliste, demi-finalistes, ...)
```

### Règles de match (inchangées par rapport au POC)

- 1 Pokémon actif, pas de banc
- Énergies auto-piochées (pool infini)
- 6 actions (Pass, AttachEnergy, Attack0-3)
- Condition de victoire : KO adverse

## Architecture

### Composants existants (inchangés)

```
CardRegistry          — catalogue des 69 Basic
GameStateFactory      — crée un match (card1, card2)
CSharpBackend         — moteur de règles
IActionProvider       — interface policy (Random, Human, IA)
MatchRunner           — exécute un match complet
```

### Nouveaux composants

```
TournamentRunner      — orchestre le bracket complet
TournamentBracket     — structure de l'arbre (matchups, avancement)
TournamentResult      — résultat complet (classement, stats par carte)
ICardSelector         — interface de sélection de carte
```

### Interfaces

```csharp
// Sélection de carte
public interface ICardSelector
{
    int SelectCard(int playerId, int tournamentSize);
}

// Implémentations
RandomCardSelector       — choix aléatoire (baseline)
FixedCardSelector        — carte pré-assignée (pour tests)
HumanCardSelector        — prompt console
```

### TournamentRunner

```
Input:
  - N joueurs (puissance de 2)
  - 1 ICardSelector par joueur (ou partagé)
  - 1 IActionProvider par joueur (policy de jeu)
  - 1 CSharpBackend
  - 1 seed (reproductibilité)

Output:
  - TournamentResult (arbre complet, vainqueur, stats)
```

### Bracket

```
Tournoi 8 joueurs — exemple :

Round 1 (quarts)        Round 2 (demis)       Finale
─────────────────       ───────────────       ──────
J1 [Pikachu]   ─┐
                 ├─→ Vainqueur A ─┐
J2 [Vulpix]    ─┘                 │
                                  ├─→ Vainqueur E ─┐
J3 [Machop]    ─┐                 │                │
                 ├─→ Vainqueur B ─┘                │
J4 [Geodude]   ─┘                                  │
                                                    ├─→ Champion
J5 [Abra]      ─┐                                  │
                 ├─→ Vainqueur C ─┐                │
J6 [Gastly]    ─┘                 │                │
                                  ├─→ Vainqueur F ─┘
J7 [Ponyta]    ─┐                 │
                 ├─→ Vainqueur D ─┘
J8 [Staryu]    ─┘
```

## Métriques et analyse

### Par tournoi

| Métrique | Description |
|---|---|
| `winner_card` | Carte du vainqueur |
| `rounds_played` | Nombre de rounds |
| `total_matches` | N-1 matchs au total |
| `avg_match_turns` | Durée moyenne d'un match (en tours) |

### Agrégées (sur K tournois)

| Métrique | Description |
|---|---|
| `win_rate[card]` | % de tournois gagnés par carte |
| `top4_rate[card]` | % de présence en demi-finale |
| `elimination_round[card]` | Round moyen d'élimination |
| `matchup_matrix[i,j]` | Win rate de carte i vs carte j |

### Questions à résoudre

1. **Le type a-t-il un impact ?** — Les cartes Feu gagnent-elles plus souvent face à des cartes Plante ?
2. **Y a-t-il des cartes dominantes ?** — Certaines cartes ont-elles un win rate significativement supérieur ?
3. **Les HP sont-ils le facteur dominant ?** — Ou le ratio dégâts/coût compte-t-il aussi ?

## CLI

```
dotnet run --project src/Ptcgo2.Console -- tournament [options]

Options:
  --players       Nombre de joueurs (puissance de 2)     [default: 8]
  --policy        Policy de jeu: random                  [default: random]
  --card-select   Sélection de carte: random | human     [default: random]
  --tournaments   Nombre de tournois à jouer             [default: 1]
  --seed          Seed de départ                         [default: 42]
  --verbose       Afficher chaque match                  [default: false]
```

### Exemples

```bash
# 1 tournoi de 8 joueurs, tout random
dotnet run --project src/Ptcgo2.Console -- tournament

# 1000 tournois de 16 joueurs pour analyse statistique
dotnet run --project src/Ptcgo2.Console -- tournament \
  --players 16 --tournaments 1000 --seed 0

# Tournoi interactif (humain choisit sa carte)
dotnet run --project src/Ptcgo2.Console -- tournament \
  --players 8 --card-select human --verbose
```

## Output

### Mode verbose (1 tournoi)

```
═══════════════════════════════════════════════
  Tournoi #1 — 8 joueurs — seed 42
═══════════════════════════════════════════════

  Sélection de cartes:
    J1: Pikachu (60 HP, Electric)
    J2: Vulpix (50 HP, Fire)
    J3: Machop (50 HP, Fighting)
    J4: Geodude (50 HP, Fighting)
    J5: Abra (30 HP, Psychic)
    J6: Gastly (30 HP, Psychic)
    J7: Ponyta (40 HP, Fire)
    J8: Staryu (40 HP, Water)

  Round 1 (quarts):
    J1 Pikachu  vs  J2 Vulpix    → J1 gagne (8 tours)
    J3 Machop   vs  J4 Geodude   → J3 gagne (12 tours)
    J5 Abra     vs  J6 Gastly    → J6 gagne (6 tours)
    J7 Ponyta   vs  J8 Staryu    → J8 gagne (10 tours)

  Round 2 (demis):
    J1 Pikachu  vs  J3 Machop    → J1 gagne (9 tours)
    J6 Gastly   vs  J8 Staryu    → J8 gagne (7 tours)

  Finale:
    J1 Pikachu  vs  J8 Staryu    → J8 gagne (11 tours)

  Champion: J8 — Staryu (Water)
═══════════════════════════════════════════════
```

### Mode statistique (K tournois)

```
═══════════════════════════════════════════════
  Tournois: 1000  Joueurs/tournoi: 16  Seed: 0
  Policy: random  Card select: random
═══════════════════════════════════════════════

  Top 10 — Win rate (victoire tournoi):
    #1  Chansey      (120 HP, Colorless)   4.8%
    #2  Hitmonchan   (70 HP, Fighting)     3.9%
    #3  Nidoking     (90 HP, Grass)        3.2%
    ...

  Top 10 — Top 4 rate (demi-finale):
    #1  Chansey      (120 HP, Colorless)  18.2%
    #2  Hitmonchan   (70 HP, Fighting)    14.7%
    ...

  Matchup matrix (top 5 cartes, win rate %):
              Chansey  Hitmonchan  Nidoking  Poliwrath  Raichu
  Chansey        —        62%        71%       58%       55%
  Hitmonchan    38%        —         54%       61%       49%
  Nidoking      29%       46%         —        52%       44%
  Poliwrath     42%       39%        48%        —        63%
  Raichu        45%       51%        56%       37%        —
```

## Implémentation

### Phase 1 — Bracket + TournamentRunner (core)

1. `TournamentBracket` : structure d'arbre, avancement des vainqueurs
2. `TournamentRunner` : orchestration (sélection → rounds → résultat)
3. `RandomCardSelector` : baseline
4. Commande CLI `tournament`

### Phase 2 — Statistiques et analyse

5. Agrégation sur K tournois (win rate, top4 rate, matchup matrix)
6. Output formaté console

## Critères de succès

- Le pipeline tournoi tourne de bout en bout (sélection → bracket → résultat)
- Les tournois sont reproductibles (même seed = même résultat)
- Sur 1000+ tournois random, des tendances émergent (cartes à hauts HP / bon type avantagées)

## Limitations (hors scope v4)

- Pas de sélection de carte intelligente (policy de draft) — random uniquement
- Pas de swiss-system ou de poules — élimination directe uniquement
- Pas de side-board ou de changement de carte entre les rounds
- Pas de seeding basé sur un classement (tirage purement aléatoire)

## Auteurs

Q Humain, le 2026-02-19
Claude Opus 4.6 (Anthropic), rédaction, le 2026-02-19
