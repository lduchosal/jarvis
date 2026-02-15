# Dataset Quality — POC WM (Étape 2)

Objectif: valider que `data/transitions.jsonl` est exploitable pour entraîner `WorldModelNet` sans injecter de biais ou d'incohérences de simulation.

Ce document définit:
- les checks qualité obligatoires
- les seuils de validation
- une proposition d'implémentation simple (script Python)

## Entrée et sortie

Entrée:
- `data/transitions.jsonl`
- format attendu par ligne: `{obs[122], action[0..5], next_obs[122], reward, done}`

Sorties:
- `reports/dataset_quality.json` (métriques structurées)
- `reports/dataset_quality.txt` (résumé lisible)
- code de sortie processus: `0` si PASS, `1` si FAIL

## Critères qualité

### 1) Intégrité schéma

Checks:
- toutes les lignes sont du JSON valide
- champs requis présents: `obs`, `action`, `next_obs`, `reward`, `done`
- `len(obs) == 122` et `len(next_obs) == 122`
- `action` entier dans `[0,5]`
- `done` booléen
- `reward` numérique

Seuil PASS:
- 0 erreur de parsing
- 0 ligne invalide

### 2) Cohérence numérique

Checks:
- pas de NaN/Inf dans `obs` et `next_obs`
- valeurs bornées raisonnablement (features normalisées)
- pas de vecteur entièrement nul sur `obs`/`next_obs`

Seuil PASS:
- NaN/Inf = 0
- taux de lignes hors bornes < 0.1%

### 3) Distribution actions

Checks:
- histogramme des actions 0..5
- ratio action la plus fréquente / total
- ratio actions d'attaque (2..5)

Seuil PASS (POC random):
- chaque action apparaît au moins 0.5% (sauf actions N/A structurelles, voir note)
- action max < 70% du total

Note:
- si `Attack2/Attack3` sont souvent impossibles, elles peuvent rester faibles; dans ce cas valider surtout 0/1/2/3.

### 4) Distribution rewards et done

Checks:
- histogramme rewards (`-0.01`, `0`, `+1`, `-1`)
- taux `done=true`
- cohérence terminale: `done=true` doit coïncider avec reward terminal non nul

Seuil PASS:
- `done_rate` entre 1% et 20% (ordre de grandeur attendu)
- incohérences `done/reward` < 0.01%

### 5) Deltas statiques vs dynamiques

Checks:
- calcul `delta = next_obs - obs`
- dims statiques: delta attendu ~0
- dims dynamiques: delta non nul plausible

Seuil PASS:
- MSE statiques < 1e-6
- au moins une dim dynamique non nulle dans > 30% des transitions

### 6) Sanity checks règles POC

Checks approximatifs depuis obs/delta:
- énergie n'évolue pas de manière impossible (variation excessive en un step)
- HP ne monte pas (POC sans soin)
- terminal done cohérent avec chute HP (si observable)

Seuil PASS:
- violations < 0.1%

### 7) Déduplication et couverture

Checks:
- taux de transitions dupliquées exactes (`obs,action,next_obs,reward,done`)
- cardinalité des paires `(pokemon_current, pokemon_opponent)` si encodage récupérable

Seuil PASS:
- duplicats exacts < 95% (sinon dataset trop répétitif)

## Verdict global

Règle:
- FAIL si un check critique échoue (`schéma`, `NaN/Inf`, `MSE statiques`)
- FAIL si plus de 2 checks non critiques échouent
- sinon PASS

## Proposition d'implémentation

Script proposé: `training/check_dataset_quality.py`

CLI:
- `python training/check_dataset_quality.py --input data/transitions.jsonl --out reports/`

Étapes:
1. lecture streaming JSONL (compteur + collecte métriques)
2. validation schéma par ligne
3. accumulation histogrammes actions/rewards/done
4. calcul incrémental des stats delta (moyenne, MSE statiques)
5. génération `dataset_quality.json` + `dataset_quality.txt`
6. exit code selon verdict

## Pseudo-code minimal

```python
for line in file:
    obj = parse_json(line)
    validate_schema(obj)
    obs = np.array(obj["obs"], dtype=np.float32)
    nxt = np.array(obj["next_obs"], dtype=np.float32)
    delta = nxt - obs

    update_action_hist(obj["action"])
    update_reward_hist(obj["reward"])
    update_done_hist(obj["done"])

    check_nan_inf(obs, nxt)
    check_bounds(obs, nxt)
    update_static_mse(delta[STATIC_IDX])
    update_dynamic_activity(delta[DYNAMIC_IDX])
    check_rule_sanity(obs, nxt, delta, obj)

report = build_report(metrics, thresholds)
write_json(report)
write_txt(report)
exit(0 if report["verdict"] == "PASS" else 1)
```

## Index statiques/dynamiques

Source de vérité: `doc/ptcgo2/worldmodel.md`.

Implémentation recommandée:
- créer un fichier `training/feature_layout.py` qui expose:
  - `STATIC_IDX`
  - `DYNAMIC_IDX`
  - noms de features par index

Éviter les index codés en dur dans plusieurs scripts.

## Intégration CI

Ajouter une étape avant entraînement:
- `check_dataset_quality.py`
- si FAIL: bloquer l'entraînement
- si PASS: lancer `train.py`

## Artefacts attendus

- `reports/dataset_quality.json`
- `reports/dataset_quality.txt`
- log console avec top 5 alertes

## Critère de sortie (done)

La phase "qualité dataset" est validée quand:
- script exécutable en local sur le dataset complet
- verdict PASS sur `data/transitions.jsonl`
- rapport versionné pour audit
