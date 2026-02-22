# POC v5 — Draft (Construction du deck)

## Contexte

La v4 a entraîné une PolicyNet tactique via REINFORCE dans le world model. La v5 a introduit un tournoi à élimination directe pour analyser l'impact du choix de carte. La question naturelle qui émerge : **comment faire que l'IA construise son deck et le joue**, sachant que ces deux décisions sont intimement liées.

## Le problème

Le choix de carte et le jeu tactique sont interdépendants :
- Une excellente tactique ne compense pas un deck intrinsèquement faible
- Une excellente carte ne garantit pas la victoire si le jeu tactique est mauvais
- Un agent qui apprend les deux simultanément risque de diverger ou de converger vers un équilibre instable

## Insight critique : bracket resilience ≠ matchup strength

Les données empiriques v5 (1024 tournois × 1024 joueurs, `stats.txt`) révèlent un méta non transitif de type pierre-feuille-ciseaux :

```
Matchup matrix (top 5, win rate %):
                Farfetch'd  Electabuzz  Chansey  Hitmonchan  Nidoran ♂
Farfetch'd            —          0%       50%       100%        49%
Electabuzz         100%           —        0%         0%       100%
Chansey             50%        100%         —         0%        49%
Hitmonchan           0%        100%      100%          —       100%
Nidoran ♂           51%          0%       51%         0%          —
```

FarFetch'd est #1 en win rate tournoi (1.9%) malgré un 0% absolu contre Electabuzz. Pourquoi ? Parce qu'Electabuzz (faiblesse terre) se fait éliminer en route par les Fighting avant d'atteindre FarFetch'd. Dans un bracket à élimination directe de 10 rounds, **la carte qui survit n'est pas celle qui gagne le plus de 1v1, mais celle dont les counters se font éliminer par le field**.

Conséquence fondamentale : **l'objectif du DraftAgent n'est pas de maximiser le win rate 1v1 moyen contre le méta, mais de maximiser la probabilité de survie sur N rounds consécutifs dans un bracket**.

```python
# ❌ FAUX : optimiser le win rate 1v1 pondéré par le méta
win_rate_vs_meta = sum(
    meta_distribution[opp] * evaluate_1v1(my_card, opp)
    for opp in meta_distribution
)

# ✅ CORRECT : optimiser la survie en bracket complet
tournament_survival = simulate_bracket(
    my_card=candidate_card,
    opponent_distribution=meta_distribution,
    num_rounds=10  # profondeur du bracket
)
```

De plus, l'agent doit être évalué sur **des distributions adverses variables** (pas un méta fixe unique) pour éviter l'overfitting au tournoi courant.

## Approches proposées

### Approche 1 : Hiérarchique (deux agents spécialisés)

Un **DraftAgent** apprend à choisir les cartes pour maximiser les wins, étant donné que la **PolicyNet v4 reste fixe** et constante.

- **Avantage** : découplage clair, chaque agent résout un sous-problème
- **Signal** : la matrice de matchups de v5 tournois fournit le feedback direct (cette carte gagne X% vs celle-ci)
- **Limitation** : le DraftAgent ne peut pas améliorer le jeu tactique, seulement l'adapter

### Approche 2 : Co-évolution

Entraîner les deux agents en alternance : draft, tactical, draft, tactical, jusqu'à convergence.

- **Avantage** : meilleure solution globale potentiellement
- **Limitation** : coûteux computationnellement, instabilité possible si les deux agents divergent

### Approche 3 : End-to-end conditionné

Un unique agent qui apprend à la fois à construire son deck et à le jouer, conditionné par le card_id.

- **Avantage** : flexibilité maximale, représentation unifiée
- **Limitation** : espace de recherche énorme, convergence difficile

## Baseline recommandée pour v6

Avant d'entraîner un DraftAgent ou du co-training, valider d'abord une **recherche exhaustive simple** :

### Étape 1 : Classement des cartes via tournois C# (ground truth)

Le système de tournoi v5 produit déjà un classement naturel des cartes. Lancer un grand nombre de tournois random et lire le top 10 win rate :

```bash
# 10 000 tournois × 16 joueurs = 150 000 matchs, cartes random
dotnet run --project src/Ptcgo2.Console -- tournament \
  --players 16 --tournaments 10000 --policy attackbot --seed 0
```

Le output `Top 10 — Win rate` donne directement le classement ground truth des cartes dans le moteur C#. Pas besoin de `--card-select fixed:N` (un tournoi où tout le monde a la même carte = 50/50 pur bruit).

### Étape 2 : Évaluation des cartes via rollouts WM

Pour chaque carte parmi les 69 Basic, faire N rollouts batchés dans le world model avec la PolicyNet v4 fixe :

```
pour chaque card_id in [0, 68]:
    obs_initial = construct_obs(card_id, world_state_default)
    logprobs, rewards, dones, lengths = batched_rollout(
        policy=policy_v4_fixed,
        wm=wm,
        initial_obs=obs_initial,  # Obs construite à partir de card_id
        B=256,
        max_steps=50
    )
    win_rate[card_id] = (rewards.sum(dim=0) > 0.5).float().mean()
    avg_return[card_id] = rewards.sum(dim=0).mean()
```

Résultat : classement des 69 cartes par win rate moyen dans le WM.

### Étape 3 : Valider la corrélation WM vs moteur C#

Comparer le classement WM (étape 2) avec le classement ground truth (étape 1). Si la corrélation est bonne (Spearman > 0.7), le WM a appris quelles cartes sont fortes.

### Étape 4 : Baseline DraftAgent = argmax

Une fois la corrélation validée, un DraftAgent basique est juste :

```python
def draft_agent_bracket(card_pool, wm, policy, meta_distribution, num_rounds=10):
    """Choisir la carte qui maximise la survie en bracket, pas le win rate 1v1."""
    best_card = None
    best_survival = -inf

    for card_id in card_pool:
        # Simuler des brackets complets, pas des 1v1 isolés
        survival_rate = simulate_bracket(
            my_card=card_id,
            opponent_distribution=meta_distribution,
            wm=wm, policy=policy,
            num_rounds=num_rounds,
            num_simulations=256
        )
        if survival_rate > best_survival:
            best_survival = survival_rate
            best_card = card_id

    return best_card
```

C'est la baseline corrigée : évaluation exhaustive en bracket complet (pas en 1v1). Si ce baseline beat le random significativement en tournoi C#, tu as prouvé que la hiérarchie + bracket resilience marche.

### Étape 5 : Valider l'argmax en tournoi mixte

Faire jouer l'argmax (carte fixe = meilleure carte WM) contre des joueurs random en tournoi :

```bash
# J1 joue la meilleure carte (ex: card 42), les 15 autres = random
dotnet run --project src/Ptcgo2.Console -- tournament \
  --players 16 --tournaments 1000 --policy attackbot \
  --human-player 0 --seed 0
  # (ou via FixedCardSelector pour automatiser)
```

Si le win rate tournoi de la carte argmax >> 1/16 (6.25%), le signal de draft est réel.

## Après la baseline : progression vers un vrai agent

### v6a : DraftAgent entraîné (hiérarchique)

Une fois la baseline validée, entraîner un agent de draft via :
- **Input** : liste des cartes disponibles, **distribution du méta adverse** (fréquence observée de chaque carte/archétype), état du tournoi
- **Output** : probabilité de sélectionner chaque carte
- **Signal** : taux de survie en bracket simulé (pas win rate 1v1) via rollouts WM ou table de lookup pré-calculée

Algorithme simple : REINFORCE ou Q-learning discret sur l'espace des 69 cartes.

### v6b : Co-training léger

Une fois le DraftAgent convergé, alterner :
1. Figer DraftAgent → entraîner PolicyNet contre cette stratégie de draft
2. Figer PolicyNet → entraîner DraftAgent contre cette tactique

Itérer 5-10 fois. Risqué d'oscillation, à monitorer.

### v6c : Vrai deckbuilding (hors scope)

Passer de "1 carte parmi 69" à "60 cartes parmi 300+ avec synergies" multipliera l'espace par plusieurs ordres de magnitude. À réserver pour plus tard une fois la preuve de concept hiérarchique validée. Intermédiaire viable : 10 decks pré-construits à choisir.

## Architecture mise à jour

```
world_model.pt       → core simulator (frozen après v4)
policy_v4.pt         → tactical agent (frozen pour v6 baseline)
draft_agent.pt       → NEW: card selection via RL (v6a+)

wm_env.py            → legal_mask_from_obs (existant)
draft_eval.py        → NEW: evaluate_card(card_id, wm, policy)
draft_agent.py       → NEW: DraftAgent(draft_net, optimizer)
train_draft.py       → NEW: training loop REINFORCE sur cartes
```

## Critères de succès

**v6 baseline (bracket-aware argmax)**
- Tournois C# (10K+) produisent un classement stable des cartes (ground truth)
- Les 10 meilleures cartes en WM corrèlent avec le classement C# (Spearman > 0.7)
- Bracket argmax en tournoi mixte : win rate tournoi >> 1/N (significativement au-dessus du hasard)
- Validation sur distributions adverses variables (pas un seul méta fixe)

**v6a (DraftAgent entraîné)**
- DraftAgent convergé en < 1000 iterations
- Win rate vs random > 75% (amélioration vs argmax baseline)
- Compétitif vs vrai joueur humain (si applicable)

**v6b (Co-training)**
- PolicyNet améliore face à DraftAgent adaptée (vs PolicyNet v4 fixe)
- Pas d'oscillation (métriques stabilisent en < 500 iterations)

## Limitations (hors scope v6)

- Pas de vrai deckbuilding (60 cartes, synergies, contraintes)
- Pas de métagame adaptatif dynamique (adversaires qui adaptent en temps réel)
- Le méta est fourni en input statique, pas appris en ligne
- Pas de draft au sens Pokémon TCG réel (multiple rounds, constraints, cartes évolutives)
- L'adversaire reste random — pas de co-draft simultanée

## Auteurs

Q Humain, le 2026-02-21
Claude Opus 4.6 (Anthropic), discussion, le 2026-02-21
Claude Haiku 4.5 (Anthropic), rédaction, le 2026-02-21
