# Étape 3b — Hygiène expérimentale — Résultats

## Modifications appliquées

**`training/draft_eval.py`** — seed (`--seed 0`), `eval_matchup` returns `(win_rate, timeout_rate)`, INV-5 symmetry assertions, suffixed artefacts (`_b{B}`), `timeout_stats.json`, `run_meta.json` avec SHA-256 hashes.

**`training/correlation_check.py`** — INV-1 mapping asserts (39 cartes, noms matchent), bootstrap CI (10K resamples), zone determination (ROUGE/GRISE/VERTE), timeout garde-fou, `--timeout-stats` et `--output-suffix` args.

## Checks

| Check | Status | Résultat |
|-------|--------|----------|
| Check 3 (INV-1 mapping) | OK | 39 cartes bijectif, noms matchent |
| Check 4 (seed) | OK | `--seed 0` mécanisme en place |
| Check 1 (B=1024) | OK | ρ passe de 0.624 → **0.723** |
| Check 5 (timeout) | OK | Taux global = **1.0%** (bien sous 5%) |
| Check 2 (bootstrap CI) | OK | median=0.720, CI=[0.495, 0.856] |
| INV-5 (symétrie) | OK | matrice + transposée = ones |

## Résultat principal

Avec B=1024, **ρ = 0.723 (GO sur estimation ponctuelle)**, contre 0.624 à B=256. Le bruit statistique tirait effectivement ρ vers le bas.

Cependant, **zone = GRISE** (CI_upper=0.856 > 0.65 mais CI_lower=0.495 < 0.70), donc selon la table de décision pré-enregistrée : **v3.11 + re-run B=4096**.

## Artefacts produits

| Artefact | Présent |
|----------|---------|
| `matchup_matrix_raw_b1024.pt` | oui |
| `matchup_matrix_b1024.pt` | oui |
| `win_rate_wm_b1024.pt` | oui |
| `timeout_stats.json` | oui |
| `run_meta.json` | oui |
| `correlation_report_b1024.md` | oui |
| `bootstrap_histogram_b1024.png` | oui |
| `correlation_plot_b1024.png` | oui |

## Run metadata

```json
{
  "seed": 0,
  "B": 1024,
  "max_steps": 50,
  "sha256_card_index_map": "92a900b28a743ad9cfcd9f6a04b273e82ab5221f3cb22be3cde29d8ab54051ba",
  "sha256_matchup_matrix": "923db72ee33560cc4c6f3317367662104cd9b1b4cadbdf81e6865c87564e945b",
  "date": "2026-02-22T13:47:06.069319+00:00",
  "git_commit": "10ce113"
}
```
