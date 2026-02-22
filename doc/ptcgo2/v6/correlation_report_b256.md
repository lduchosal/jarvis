# Étape 3b — Correlation Report: WM vs C# Rankings (Hygiène)

## Résultat

| Métrique | Valeur |
|----------|--------|
| Spearman ρ | **0.624** |
| p-value | 2.2e-05 |
| Cartes alignées | 39 |
| Seuil go/no-go | ρ > 0.7 |
| **Décision (point)** | **NO-GO** |

## Bootstrap CI (Check 2)

| Métrique | Valeur |
|----------|--------|
| Bootstrap median | **0.625** |
| CI 95% lower | 0.348 |
| CI 95% upper | 0.814 |
| **Zone** | **GRISE** |
| **Action** | v3.11 + re-run B=4096 |

![Bootstrap histogram](bootstrap_histogram_b256.png)

## Scatter plot

![Correlation plot](correlation_plot_b256.png)

## Top 10 comparison

| Rank | C# (ground truth) | WR C# | WM (world model) | WR WM |
|------|--------------------|--------|-------------------|-------|
| 1 | Farfetch'd | 0.473 | Zapdos | 0.547 |
| 2 | Hitmonchan | 0.345 | Chansey | 0.523 |
| 3 | Chansey | 0.236 | Farfetch'd | 0.520 |
| 4 | Zapdos | 0.220 | Onix | 0.518 |
| 5 | Electabuzz | 0.209 | Electabuzz | 0.515 |
| 6 | Nidoran ♂ | 0.182 | Pikachu | 0.513 |
| 7 | Charmander | 0.164 | Magnemite | 0.512 |
| 8 | Machop | 0.145 | Hitmonchan | 0.502 |
| 9 | Magmar | 0.112 | Magikarp | 0.500 |
| 10 | Magnemite | 0.100 | Mewtwo | 0.500 |

## Outliers (|rank diff| >= 13)

| Carte | Rank C# | Rank WM | Diff | WR C# | WR WM |
|-------|---------|---------|------|--------|-------|
| Jynx | 11 | 36 | -25 | 0.0829 | 0.487 |
| Magikarp | 30 | 9 | +21 | 0.0000 | 0.500 |
| Mewtwo | 29 | 10 | +19 | 0.0000 | 0.500 |
| Charmander | 7 | 26 | -19 | 0.1641 | 0.499 |
| Rattata | 16 | 32 | -16 | 0.0108 | 0.496 |
| Weedle | 39 | 24 | +15 | 0.0000 | 0.500 |

## Analyse

**ZONE GRISE** (0.65 ≤ CI_upper < 0.70 ET CI_lower < 0.70). Le WM est probablement le bottleneck, mais le bruit contribue.

**Décision : v3.11 + re-run B=4096** — lancer v3.11 en parallèle d'un run B=4096 pour trancher.
