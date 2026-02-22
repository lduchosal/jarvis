# POC v6 — Draft (Construction du deck)

## Contexte

La v4 a entraîné une PolicyNet tactique via REINFORCE dans le world model. La v5 a introduit un tournoi à élimination directe pour analyser l'impact du choix de carte. La question naturelle qui émerge : **comment faire que l'IA construise son deck et le joue**, sachant que ces deux décisions sont intimement liées.

## Pipeline technique : entraînement Python, inférence C#

Le split est le même que pour la PolicyNet v4 :

- **Entraînement** (Python / PyTorch) : world model, PolicyNet, DraftAgent — gradient, backprop, REINFORCE
- **Inférence** (C# / ONNX) : les modèles entraînés sont exportés en `.onnx` et chargés dans le moteur C# pour les tournois et l'évaluation

Les snippets Python dans ce document concernent le côté entraînement. Les commandes CLI sont en C# (côté évaluation).

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

Note : cette matrice provient de "decks" mono-carte avec policy attackbot uniquement. Elle illustre la non-transitivité du méta, pas un classement absolu.

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
- **Signal** : la matrice de matchups v5 fournit le feedback direct (cette carte gagne X% vs celle-ci)
- **Limitation** : le DraftAgent ne peut pas améliorer le jeu tactique, seulement l'adapter

### Approche 2 : Co-évolution

Entraîner les deux agents en alternance : draft, tactical, draft, tactical, jusqu'à convergence.

- **Avantage** : meilleure solution globale potentiellement
- **Limitation** : coûteux computationnellement, instabilité possible si les deux agents divergent

### Approche 3 : End-to-end conditionné

Un unique agent qui apprend à la fois à choisir sa carte et à la jouer, conditionné par le card_id.

- **Architecture** : une phase draft (69-way softmax → card_id), puis une PolicyNet dont l'input est concaténé avec un embedding de la carte choisie
- **Avantage** : flexibilité maximale, représentation unifiée, le gradient tactique informe le choix de carte
- **Espace de recherche** : 69 cartes × 520 actions/tour — tractable. Le vrai défi est l'attribution du crédit : la décision de carte est prise une seule fois tandis que les actions tactiques sont prises à chaque tour. Avec REINFORCE, le reward signal est dilué sur toutes les décisions. Mitigation : baseline par carte (soustraire le return moyen de chaque carte pour isoler le signal tactique du signal draft)
- **Viabilité** : approche viable pour un POC à 69 cartes. Devient problématique quand l'espace de draft grandit (60 cartes dans un deck, synergies combinatoires)

**Recommandation** : démarrer par l'approche 1 (hiérarchique) pour la v6 baseline car elle isole clairement le signal de draft. L'approche 3 est une alternative crédible à explorer en étape 6 si le signal hiérarchique est trop faible.

## Baseline recommandée pour v6

Avant d'entraîner un DraftAgent ou du co-training, valider d'abord une **recherche exhaustive simple** :

### Étape 1 : Classement des cartes via tournois C# (ground truth)

Le système de tournoi v5 produit déjà un classement naturel des cartes. Lancer un grand nombre de tournois random et lire le top 10 win rate :

```bash
# 10 000 tournois × 16 joueurs = 150 000 matchs, cartes random
dotnet run --project src/Ptcgo2.App -- tournament \
  --players 16 --tournaments 10000 --policy attackbot --seed 0
```

Le output `Top 10 — Win rate` donne directement le classement ground truth des cartes dans le moteur C#.

**Délivrable** : `rankings_csharp.csv` — classement des 69 cartes avec win rate tournoi et rank, issu de 10K+ tournois C#. Sert de ground truth pour valider le world model.

### Étape 2 : Matrice de matchups via rollouts WM (round-robin)

Évaluer chaque paire de cartes (i, j) dans le world model pour construire la matrice de matchups complète :

```python
# Round-robin : 69 × 69 = 4 761 paires, B rollouts par paire
matchup_matrix = torch.zeros(69, 69)

for i in range(69):
    for j in range(69):
        if i == j:
            matchup_matrix[i, j] = 0.5
            continue
        # Construire l'obs initiale : carte i (joueur) vs carte j (adversaire)
        obs_batch = construct_obs_batch(my_card=i, opp_card=j, B=256)
        rewards = batched_rollout(
            policy=policy_v4_fixed, wm=wm,
            initial_obs=obs_batch, max_steps=50
        )
        matchup_matrix[i, j] = (rewards > 0.5).float().mean()

# Win rate moyen par carte (vs opposition random uniforme)
win_rate_wm = matchup_matrix.mean(dim=1)  # [69]
```

La matrice complète sert à deux choses :
1. **Win rate moyen** (moyenne des lignes) → corrélation avec le classement C# (étape 3)
2. **Simulation de brackets** (étape 4) → lookups instantanés, pas de rollouts supplémentaires

Coût : 4 761 × 256 = ~1.2M rollouts. À 50 steps/rollout, ~60M forward passes WM — faisable en quelques minutes sur GPU.

**Délivrable** : `matchup_matrix.pt` — tenseur `[69, 69]` des win rates paire-à-paire + `win_rate_wm.pt` — vecteur `[69]` des win rates moyens par carte. Ces fichiers servent d'input aux étapes 3 et 4.

### Étape 3 : Valider la corrélation WM vs moteur C#

Comparer le classement WM (étape 2) avec le classement ground truth (étape 1) via la **corrélation de Spearman**.

**Spearman ρ** mesure si deux classements sont cohérents, indépendamment des valeurs absolues. Contrairement à Pearson (qui compare les valeurs), Spearman compare uniquement les **rangs** : si la carte #1 en C# est aussi #1 dans le WM, etc. Formellement c'est le coefficient de Pearson appliqué aux rangs.

| ρ | Interprétation |
|---|---|
| 1.0 | Classements identiques |
| 0.7+ | Corrélation forte — les meilleures cartes WM sont aussi les meilleures en C# |
| 0.4–0.7 | Corrélation modérée — le WM capture la tendance mais pas les détails |
| ≈ 0 | Aucune corrélation — le WM n'a pas appris quelles cartes sont fortes |

**Seuil ρ > 0.7** : on veut que le WM classe correctement le top tier. Avec 69 cartes, ρ = 0.7 signifie que les 10–15 meilleures cartes en WM et en C# se chevauchent largement. En dessous, le WM n'est pas fiable pour simuler des brackets — il faudrait investiguer si le problème vient du WM (mauvaise modélisation des types/dégâts) ou de la métrique (effets de bracket non capturés par le win rate moyen).

```python
from scipy.stats import spearmanr

rho, p_value = spearmanr(ranking_csharp, ranking_wm)
print(f"Spearman ρ = {rho:.3f}, p = {p_value:.1e}")
```

**Délivrable** : `correlation_report.md` — Spearman ρ + p-value, scatter plot (rank C# vs rank WM), analyse des outliers (cartes mal classées par le WM). Décision **go/no-go** : ρ > 0.7 → continuer vers étape 4, sinon investiguer le WM.

### Étape 4 : Baseline DraftAgent = argmax bracket

Une fois la corrélation validée, un DraftAgent basique évalue chaque carte par simulation de bracket :

```python
def draft_agent_bracket(card_pool, matchup_matrix, meta_distribution, num_rounds=10):
    """Choisir la carte qui maximise la survie en bracket, pas le win rate 1v1."""
    best_card = None
    best_survival = -inf

    for card_id in card_pool:
        # Simuler des brackets complets via la matrice pré-calculée (lookup, pas de rollout)
        survival_rate = simulate_bracket(
            my_card=card_id,
            matchup_matrix=matchup_matrix,
            opponent_distribution=meta_distribution,
            num_rounds=num_rounds,
            num_simulations=1024
        )
        if survival_rate > best_survival:
            best_survival = survival_rate
            best_card = card_id

    return best_card
```

La matrice pré-calculée (étape 2) rend chaque simulation de bracket quasi-instantanée : un tirage aléatoire d'adversaires + des lookups dans la matrice, pas de rollouts WM.

**Délivrable** : `bracket_rankings.csv` — classement des 69 cartes par survival rate en bracket (1024 simulations × 10 rounds chacune), avec la carte argmax identifiée. Fichier `draft_eval.py` contenant `simulate_bracket()` et le round-robin.

### Étape 5 : Valider l'argmax en tournoi mixte

Faire jouer l'argmax (carte fixe = meilleure carte WM) contre des joueurs random en tournoi :

```bash
# J1 joue la meilleure carte (ex: card 42), les 15 autres = random
dotnet run --project src/Ptcgo2.App -- tournament \
  --players 16 --tournaments 1000 --policy attackbot \
  --fixed-card 0:42 --seed 0
```

Si le win rate tournoi de la carte argmax >> 1/16 (6.25%), le signal de draft est réel.

**Délivrable** : `validation_report.md` — win rate tournoi de la carte argmax vs baseline random (6.25%), avec intervalle de confiance sur 1000+ tournois. Confirmation que le signal de draft est exploitable. Résumé de la baseline v6 complète.

## Après la baseline : progression vers un vrai agent

### Étape 6 : DraftAgent entraîné via RL (hiérarchique)

Une fois la baseline validée, entraîner un agent de draft via RL.

#### Observation du DraftAgent

```
obs_draft = {
    meta_dist:      Float[69]    # fréquence de chaque carte dans le méta adverse
}
```

Deux designs possibles :

**Design A — Meta-only (recommandé pour l'étape 6)**

Le réseau reçoit uniquement la distribution du méta et apprend implicitement les propriétés des cartes via ses poids :

```python
class DraftNet(nn.Module):
    def __init__(self, num_cards=69, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_cards, hidden),      # meta_dist [69] → hidden
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, num_cards),      # → logits [69]
        )

    def forward(self, meta_dist):              # [B, 69]
        return self.net(meta_dist)             # [B, 69] logits
```

Input : `[69]` = 69 floats. Output : `[69]` logits → softmax → sélection de carte.
Simple, suffisant pour 69 cartes fixes.

**Design B — Card-aware (si généralisation requise)**

Le réseau utilise les features de chaque carte (HP, type, dégâts) pour généraliser à des cartes jamais vues :

```python
class DraftNetCardAware(nn.Module):
    def __init__(self, card_feat_dim=20, hidden=128):
        super().__init__()
        self.card_encoder = nn.Linear(card_feat_dim, 64)
        self.meta_encoder = nn.Linear(69, 64)
        self.scorer = nn.Linear(128, 1)

    def forward(self, meta_dist, card_features):   # [B, 69], [69, F]
        meta_emb = self.meta_encoder(meta_dist)    # [B, 64]
        card_emb = self.card_encoder(card_features)  # [69, 64]
        # Score chaque carte conditionnellement au méta
        meta_exp = meta_emb.unsqueeze(1).expand(-1, 69, -1)  # [B, 69, 64]
        card_exp = card_emb.unsqueeze(0).expand(meta_exp.size(0), -1, -1)
        combined = torch.cat([meta_exp, card_exp], dim=-1)    # [B, 69, 128]
        return self.scorer(combined).squeeze(-1)   # [B, 69] logits
```

Le design A suffit pour 69 cartes fixes. Le design B sera nécessaire quand le pool de cartes change (extensions, bans).

#### Signal d'entraînement

- **Reward** : taux de survie en bracket simulé (pas win rate 1v1)
- **Algo** : REINFORCE sur l'espace discret des 69 cartes
- **Meta** : la distribution adverse est échantillonnée (variée à chaque episode pour éviter l'overfitting)

**Délivrable** : `draft_agent.pt` — DraftNet entraîné + `draft_agent.onnx` pour inférence C#. Courbes de convergence (loss, win rate vs random au fil des steps). Tournoi comparatif : DraftAgent RL vs argmax baseline (étape 5) vs random, sur 1000+ tournois C#. Fichiers `draft_agent.py` et `train_draft.py`.

### v6b : Co-training léger

Une fois le DraftAgent convergé, alterner :
1. Figer DraftAgent → entraîner PolicyNet contre cette stratégie de draft
2. Figer PolicyNet → entraîner DraftAgent contre cette tactique

Itérer 5-10 fois. Risqué d'oscillation, à monitorer.

### v6c : Vrai deckbuilding (hors scope)

Passer de "1 carte parmi 69" à "60 cartes parmi 300+ avec synergies" multipliera l'espace par plusieurs ordres de magnitude. À réserver pour plus tard une fois la preuve de concept hiérarchique validée. Intermédiaire viable : 10 decks pré-construits à choisir.

## Architecture mise à jour

```
# Entraînement (Python / PyTorch)
world_model.pt       → core simulator (frozen après v4)
policy_v4.pt         → tactical agent (frozen pour v6 baseline)
draft_agent.pt       → NEW: card selection via RL (v6a+)

wm_env.py            → legal_mask_from_obs (existant)
draft_eval.py        → NEW: round-robin matchup matrix, bracket simulation
draft_agent.py       → NEW: DraftNet + REINFORCE training
train_draft.py       → NEW: training loop, meta sampling, bracket eval

# Inférence (C# / ONNX)
policy_v4.onnx       → tactical agent pour tournois C#
draft_agent.onnx     → NEW: export pour sélection de carte en tournoi C#
```

## Critères de succès

**v6 baseline (bracket-aware argmax)**
- Tournois C# (10K+) produisent un classement stable des cartes (ground truth)
- Les 10 meilleures cartes en WM corrèlent avec le classement C# (Spearman ρ > 0.7)
- Bracket argmax en tournoi mixte : win rate tournoi >> 1/N (significativement au-dessus du hasard)
- Validation sur distributions adverses variables (pas un seul méta fixe)

**Étape 6 (DraftAgent entraîné)**
- DraftAgent convergé en < 1000 gradient steps
- Win rate vs random > 75% en tournoi (amélioration vs argmax baseline)

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
Claude Haiku 4.5 (Anthropic), rédaction initiale, le 2026-02-21