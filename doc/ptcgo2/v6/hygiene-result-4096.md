# Étape 3b — Hygiène expérimentale — Résultats B=4096

## Convergence de ρ

| B | ρ | Bootstrap median | CI 95% |
|---|---|---|---|
| 256 | 0.624 | 0.625 | [0.348, 0.814] |
| 1024 | 0.723 | 0.720 | [0.495, 0.856] |
| **4096** | **0.724** | **0.722** | **[0.499, 0.857]** |

ρ a convergé — passer de 1024 à 4096 n'a rien changé (0.723 → 0.724). La largeur du CI ne bouge quasiment pas non plus. Le CI large vient de N=39 cartes (bootstrap resample 39 points), pas du bruit de rollout.

## Checks

| Check | Status | Résultat |
|-------|--------|----------|
| Check 3 (INV-1 mapping) | OK | 39 cartes bijectif, noms matchent |
| Check 4 (seed) | OK | `--seed 0` |
| Check 1 (B=4096) | OK | ρ = **0.724** (stable vs B=1024) |
| Check 5 (timeout) | OK | Taux global = **1.0%** |
| Check 2 (bootstrap CI) | OK | median=0.722, CI=[0.499, 0.857] |
| INV-5 (symétrie) | OK | matrice + transposée = ones |

## Résultat

**Zone = GRISE** — l'estimation ponctuelle passe GO (ρ=0.724 > 0.70), mais CI_lower=0.499 < 0.70. Selon la table de décision pré-enregistrée : **v3.11**.

Le fait que ρ soit stable entre B=1024 et B=4096 confirme que le bruit de rollout n'est plus le facteur limitant. La variance résiduelle du CI vient du petit N=39 et ne peut pas être réduite par plus de rollouts. Le WM lui-même est le bottleneck.

## Artefacts produits

| Artefact | Présent |
|----------|---------|
| `matchup_matrix_raw_b4096.pt` | oui |
| `matchup_matrix_b4096.pt` | oui |
| `win_rate_wm_b4096.pt` | oui |
| `timeout_stats.json` | oui |
| `run_meta.json` | oui |
| `correlation_report_b4096.md` | oui |
| `bootstrap_histogram_b4096.png` | oui |
| `correlation_plot_b4096.png` | oui |

## Run metadata

```json
{
  "seed": 0,
  "B": 4096,
  "max_steps": 50,
  "date": "2026-02-22",
  "git_commit": "10ce113"
}
```
