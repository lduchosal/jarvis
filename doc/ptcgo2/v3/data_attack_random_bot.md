# Data v3.5 — Bot Attach-Attack + Mix Random

Objectif : améliorer la qualité des transitions d'entraînement en mélangeant un bot stratégique (attach-attack) avec le bot random existant.

## Diagnostic

Le bot random produit ~45% de Pass, beaucoup d'infractions (ILL02, ILL03, ILL04), et des parties longues (~21 steps). Le réseau passe une grande partie de sa capacité à apprendre des transitions peu utiles. Les parties réelles (humain ou IA) suivent une stratégie évidente : attacher dès que possible, attaquer dès que le coût est payé.

## Bot Attach-Attack — Logique

```
Si Phase == StartTurn:
    → auto (le moteur gère)

Si Phase == Main:
    1. Si energyAttached == 0 et énergie disponible → AttachEnergy
    2. Si une attaque est légale (coût payé) → attaque la plus puissante
    3. Sinon → Pass
```

Pas d'infractions. Pas de Pass inutiles. Parties plus courtes et plus proches du jeu réel.

## Contraintes d'isolation

- Architecture réseau : identique à v3.4 (130K params, tête HP dédiée)
- Loss et poids : identiques à v3.4 (w_pass=2, w_hp_aux=5, w_attack_hp=10)
- Hyperparamètres training : identiques (batch=1024, lr=2e-3, 100+ époques)
- Benchmark : mêmes seeds, mêmes seuils
- Seul le dataset change (policy de collecte)

## Mix de policies

| Config | Random | Attach-Attack | Parties totales |
|--------|--------|---------------|-----------------|
| R0 (baseline) | 100% | 0% | 100K |
| R1 | 50% | 50% | 100K |
| R2 | 30% | 70% | 100K |

Chaque config produit ~2M transitions (volume comparable).

### Pourquoi un mix ?

- 100% attach-attack manquerait de diversité : le bot fait toujours la séquence optimale, le réseau ne verrait jamais d'actions sous-optimales
- Un humain peut faire des Pass inattendus ou des erreurs — le WM doit gérer ça
- Le random couvre les cas "bizarres" (infractions, séquences inhabituelles)
- Le attach-attack fournit les transitions d'attaque réalistes qui manquent

## Implémentation du bot

Fichier cible : `src/Ptcgo2.Core/AttackBot.cs` (implémente `IActionProvider`)

```csharp
public class AttackBot : IActionProvider
{
    public int SelectAction(in GameState state, ReadOnlySpan<int> legalActions)
    {
        // 1. AttachEnergy si pas encore fait ce tour et légal
        if (legalActions.Contains((int)ActionId.AttachEnergy)
            && !state.Current.EnergyAttachedThisTurn)
            return (int)ActionId.AttachEnergy;

        // 2. Attaque la plus puissante parmi les légales
        int bestAttack = -1;
        int bestDamage = 0;
        for (int a = 2; a <= 5; a++)
        {
            if (!legalActions.Contains(a)) continue;
            var card = CardRegistry.Get(state.Current.CardIndex);
            int atkIdx = a - 2;
            if (atkIdx < card.Attacks.Length && card.Attacks[atkIdx].Damage > bestDamage)
            {
                bestDamage = card.Attacks[atkIdx].Damage;
                bestAttack = a;
            }
        }
        if (bestAttack >= 0) return bestAttack;

        // 3. Pass
        return (int)ActionId.Pass;
    }
}
```

## Collecte

```bash
# R1 : 50/50
dotnet run -- collect --games 50000 --policy random --output data/transitions_random_50k.jsonl
dotnet run -- collect --games 50000 --policy attack --output data/transitions_attack_50k.jsonl
# Concatener
cat data/transitions_random_50k.jsonl data/transitions_attack_50k.jsonl > data/transitions_mix50.jsonl

# R2 : 30/70
dotnet run -- collect --games 30000 --policy random --output data/transitions_random_30k.jsonl
dotnet run -- collect --games 70000 --policy attack --output data/transitions_attack_70k.jsonl
cat data/transitions_random_30k.jsonl data/transitions_attack_70k.jsonl > data/transitions_mix70.jsonl
```

## Métriques attendues

Le bot attach-attack devrait produire :
- Moins de transitions Pass (cible <20% vs 45% actuellement)
- Plus de transitions d'attaque avec dégâts réels
- Parties plus courtes (~10-15 steps vs ~21)
- Zéro infraction

## Vérification dataset

Lancer `check_dataset_quality.py` sur chaque dataset. Vérifier en particulier :
- Distribution des actions (histogramme) — le mix doit montrer un ratio Pass réduit
- Taux de done aligné ou légèrement plus élevé (parties plus courtes)
- Static MSE toujours ~0

## Plan d'exécution

1. Implémenter `AttackBot` dans le moteur C#
2. Ajouter `--policy attack` à la CLI de collecte
3. Collecter R1 (50/50) et R2 (30/70)
4. Quality check sur les deux datasets
5. Entraîner avec config v3.4 exacte sur R1, puis R2
6. Benchmark dual backend (200K parties, mêmes seeds)
7. Comparer R0 (baseline random) vs R1 vs R2

## Critères de promotion

Le dataset mix est retenu si :
- `Cumul drift/game (HP)` en baisse vs baseline R0
- `KO miss rate` reste à 0
- `Done F1 > 0.95`
- Pas de régression critique sur `HP divergence MAE`

Si aucun mix n'améliore le drift, le levier data-policy est écarté (comme la v2.1 data-only).

## Séquençage

Ce run est indépendant du tuning w_pass. On peut le faire en parallèle ou après :
- Si le v3.4 + 200 époques passe le seuil de drift → data-policy devient optionnel (amélioration future)
- Si le v3.4 + 200 époques ne passe pas → data-policy devient prioritaire

## Auteurs

Q Humain, spécification produit, le 2026-02-16
Claude Opus 4.6 (Anthropic), rédaction, le 2026-02-16
Codex 5.3 (OpenAI), co-rédaction, le 2026-02-16
