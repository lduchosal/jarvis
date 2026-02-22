# Étape 3b — Hygiène expérimentale avant v3.11

## Contexte

Le correlation report (étape 3) donne Spearman ρ = 0.624 (NO-GO, seuil 0.7). Avant d'investir dans v3.11 (damage sub-network), cette étape vérifie que le ρ observé n'est pas artificiellement bas à cause d'artefacts expérimentaux. Coût total estimé : **< 1h**.

Si l'hygiène révèle que ρ est sous-estimé par le protocole, on économise l'effort v3.11. Si ρ reste sous 0.7 après correction, on a éliminé le doute et on investit dans v3.11 en confiance.

## Check 1 — Puissance statistique (B=256 est-il suffisant ?)

### Problème

Avec B=256 rollouts par paire, l'erreur standard sur un win rate à 0.50 est :

```
SE = sqrt(0.5 × 0.5 / 256) ≈ 0.031
```

Le spread WM observé est de 0.09 (0.46–0.55). L'écart entre le 1er et le 10e est ~0.047. Avec SE=0.031, une bonne partie du classement WM repose sur du bruit — deux cartes séparées de 0.01 en WR sont indistinguables statistiquement.

### Action

Re-run `draft_eval.py` avec B=1024 (4× plus de rollouts) :

```bash
python training/draft_eval.py --batch 1024
```

Coût : 4761 × 1024 = ~4.9M rollouts (vs 1.2M à B=256). ~4× le temps, soit ~10–15 min sur GPU.

### Métrique attendue

| B | SE (à p=0.5) | Spread nécessaire pour 2σ séparation |
|---|---|---|
| 256 | 0.031 | 0.062 |
| 1024 | 0.016 | 0.031 |

Avec B=1024, des différences de 0.03 en WR deviennent significatives (vs 0.06 à B=256). Si le spread WM s'élargit (certaines paires étaient sous-échantillonnées), ρ pourrait monter.

### Délivrable

`matchup_matrix_b1024.pt`, `win_rate_wm_b1024.pt` — matrice et win rates à B=1024.

## Check 2 — Bootstrap CI sur ρ

### Problème

Un seul ρ = 0.624 ne dit pas si on est à 0.62 ± 0.01 ou 0.62 ± 0.10. Sans intervalle de confiance, on ne sait pas si ρ > 0.7 est dans le domaine du plausible.

### Action

Ajouter un bootstrap à `correlation_check.py` :

```python
import numpy as np
from scipy.stats import spearmanr

def bootstrap_spearman(wr_csharp, wr_wm, n_boot=10000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(wr_csharp)
    rhos = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        rho, _ = spearmanr(wr_csharp[idx], wr_wm[idx])
        rhos.append(rho)
    rhos = np.array(rhos)
    return {
        "median": np.median(rhos),
        "ci_lower": np.percentile(rhos, 2.5),
        "ci_upper": np.percentile(rhos, 97.5),
    }
```

### Décision

| Résultat | Interprétation |
|----------|---------------|
| CI_upper < 0.7 | ρ > 0.7 est exclu — v3.11 nécessaire, pas de doute |
| 0.7 ∈ [CI_lower, CI_upper] | ρ > 0.7 plausible — le bruit pourrait expliquer le NO-GO |
| CI_lower > 0.7 | Le check 1 (B=1024) a probablement résolu le problème |

### Délivrable

Ajout au `correlation_report.md` : bootstrap median, IC 95%, histogramme ρ.

## Check 3 — Vérification du mapping card_index

### État actuel (code vérifié)

Le mapping dans `draft_eval.py` fonctionne ainsi :
1. Charge les cartes via CardLoader (même code que C#)
2. Filtre `supertype == "Pokémon"`, trie par card ID (string sort)
3. Position dans la liste triée = `card_index` (identique au CardRegistry C#)
4. Filtre `subtype == "Basic"` → 39 cartes

La corrélation dans `correlation_check.py` aligne par `card_index` via `card_index_map.json`.

### Action

Ajouter un assert de sanity check dans `correlation_check.py` :

```python
# Vérifier que les noms matchent entre C# et WM
for mi, ci in enumerate(card_indices):
    if ci in csharp:
        assert card_names[mi] == csharp[ci]["name"], \
            f"Mismatch: matrix[{mi}] = {card_names[mi]}, C# = {csharp[ci]['name']}"
```

Vérifier aussi que `card_index_map.json` a 39 entrées correspondant aux 39 lignes de `rankings_csharp.csv`.

### Délivrable

Le script passe sans erreur, ou identifie le(s) mismatch(es).

## Check 4 — Reproductibilité (seed)

### Problème

`draft_eval.py` ne set aucune seed. Deux runs produisent des matrices légèrement différentes. Pour un résultat reproductible et pour que le bootstrap CI soit interprétable, la matrice doit être déterministe.

### Action

Ajouter un seed au round-robin :

```python
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
```

Puis comparer deux runs avec le même seed : les matrices doivent être identiques (bit-for-bit).

### Délivrable

`--seed 0` ajouté comme default dans `draft_eval.py`. Deux runs identiques confirment la reproductibilité.

## Check 5 — Biais de tie-breaking

### État actuel (code vérifié)

Le code de rollout dans `draft_eval.py` gère les timeouts (max_steps atteint sans done) par comparaison HP :

```
win si self_hp > opp_hp
lose si self_hp <= opp_hp   (draw → défaite)
```

Les draws comptent comme des défaites. Si le WM produit beaucoup de timeouts (jeux qui ne terminent pas en 50 steps), ce biais tire tous les win rates vers le bas, mais de manière uniforme — ce qui compresse le spread.

### Action

Instrumenter le round-robin pour mesurer le taux de timeout :

```python
n_done = done.float().sum().item()
n_timeout = B - n_done
timeout_rate = n_timeout / B
```

Logger `timeout_rate` par paire. Si le taux moyen est > 20%, les rollouts ne terminent pas assez souvent et le tie-breaking domine le signal.

### Métrique attendue

| Timeout rate | Interprétation |
|---|---|
| < 5% | OK — la plupart des jeux terminent naturellement |
| 5–20% | Modéré — certaines paires slow (Chansey vs Chansey ?) |
| > 20% | Problématique — augmenter max_steps ou investiguer |

### Délivrable

`timeout_stats.json` — taux de timeout par paire (matrice 39×39) + taux moyen global.

## Pipeline d'exécution

```
Check 3 (mapping assert)       ~2 min    ← élimine bug trivial
    ↓
Check 4 (seed)                 ~5 min    ← modifier draft_eval.py
    ↓
Check 1 (B=1024)               ~15 min   ← re-run round-robin
    + Check 5 (timeout stats)             ← instrumenté dans le même run
    ↓
Check 2 (bootstrap CI)         ~2 min    ← sur la nouvelle matrice
    ↓
Nouveau correlation_report     ~5 min    ← re-run correlation_check.py
```

Temps total : **~30 min** (dont ~15 min de compute GPU).

## Critère de décision

| Résultat bootstrap (B=1024) | Action |
|---|---|
| CI_upper < 0.65 | v3.11 confirmé nécessaire, spread WM = bruit du WM, pas du protocole |
| 0.65 ≤ CI_upper < 0.70 | v3.11 probable, mais le protocole contribuait au problème |
| CI_upper ≥ 0.70 | **Le protocole expliquait le NO-GO** — re-évaluer avant v3.11 |

Dans tous les cas, les corrections (seed, B=1024, timeout stats) restent en place pour la suite du pipeline.

## Auteurs

Q Humain, specification produit, le 2026-02-22
Claude Opus 4.6 (Anthropic), redaction, le 2026-02-22
Codex (OpenAI), contre-analyse motivant ces checks, le 2026-02-22
