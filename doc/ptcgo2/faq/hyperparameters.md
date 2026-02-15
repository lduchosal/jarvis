# FAQ — Hyperparamètres

## C'est quoi un hyperparamètre ?

Un réglage choisi par nous AVANT l'entraînement, que le réseau ne peut pas modifier lui-même.
C'est le contraire d'un paramètre appris (poids et biais des couches), qui est ajusté automatiquement par l'optimizer à chaque batch.

Analogie : les hyperparamètres sont les règles de l'entraînement (vitesse, durée, priorités). Les paramètres appris sont les connaissances du réseau (ce qu'il a retenu).

## Paramètres appris vs hyperparamètres

| Type | Exemples | Qui le fixe | Quand |
|------|----------|-------------|-------|
| Paramètre appris | Poids des couches Linear, biais | L'optimizer (Adam) | Pendant le training, à chaque batch |
| Hyperparamètre | Learning rate, batch size, poids de loss, taille hidden | Nous | Avant le training, fixé pour tout le run |

## Quels sont nos hyperparamètres ?

### Architecture

| Hyperparamètre | Valeur actuelle | Ce qu'il contrôle |
|----------------|----------------|-------------------|
| `hidden` | 256 | Taille des couches cachées du MLP. Plus grand = plus de capacité, plus lent |
| Nombre de couches | 2 (fixe) | Profondeur du réseau |
| Nombre de têtes | 4 (delta, reward, done, hp) | Sorties spécialisées |

### Training

| Hyperparamètre | Valeur actuelle | Ce qu'il contrôle |
|----------------|----------------|-------------------|
| `lr` (learning rate) | 1e-3 | Vitesse d'apprentissage. Trop haut = instable, trop bas = trop lent |
| `batch_size` | 256 | Nombre de transitions vues à chaque step d'entraînement |
| `epochs` | 100 | Nombre de passes complètes sur le dataset |
| Scheduler | CosineAnnealing (T_max=100) | Réduit le lr progressivement jusqu'à 0 |
| `weight_decay` | 1e-5 | Régularisation pour éviter l'overfitting |
| `max_norm` (gradient clip) | 1.0 | Limite la taille des gradients pour stabiliser le training |

### Poids de loss

| Hyperparamètre | Valeur actuelle | Ce qu'il contrôle |
|----------------|----------------|-------------------|
| `loss_delta` | 1.0 | Poids de la prédiction d'état global |
| `loss_reward` | 1.0 | Poids de la prédiction de reward |
| `loss_done` | 0.5 | Poids de la prédiction de fin de partie (BCE, échelle différente) |
| `w_pass` | 2.0 | Pénalité si Pass modifie les HP. Plus haut = plus strict |
| `w_ko` | 5.0 | Pénalité pour KO raté |
| `w_hp_aux` | 5.0 | Poids de la tête HP dédiée |
| `w_attack_hp` | 10.0 | Pénalité HP sur les attaques (le plus élevé = priorité max) |

### Pondération par échantillon (dans loss_delta)

| Hyperparamètre | Valeur actuelle | Ce qu'il contrôle |
|----------------|----------------|-------------------|
| Poids attaque | ×3 | Les transitions d'attaque comptent 3× plus |
| Poids terminal | ×10 | Les fins de partie comptent 10× plus |
| Poids KO | ×20 | Les KO comptent 20× plus (cumulatif avec terminal) |

## Comment choisir les bons hyperparamètres ?

On ne peut pas les calculer, il faut les tester. Méthodes courantes :

1. **A/B testing** : tester 2-3 valeurs et comparer (c'est ce qu'on fait avec w_pass=5 vs 10)
2. **Grid search** : tester toutes les combinaisons sur une grille prédéfinie
3. **Bon sens** : partir de valeurs standard (lr=1e-3, batch=256) et ajuster si les métriques stagnent

Règle importante : évaluer sur les métriques rollout (dual backend), pas sur la val_loss seule.

## Pourquoi ne pas laisser le réseau apprendre les poids de loss ?

Parce que les poids de loss définissent l'objectif d'apprentissage. Si le réseau pouvait les modifier, il choisirait les objectifs les plus faciles et ignorerait les cas difficiles (KO, attaques). C'est nous qui décidons ce qui est important.

## Historique des changements d'hyperparamètres dans le projet

| Version | Changement | Résultat |
|---------|-----------|----------|
| v1 | Valeurs par défaut | Loss excellente offline, divergence forte en rollout |
| v2 | Ajout w_pass=2, w_ko=5, pondération échantillons | Pass HP -97%, KO miss = 0, drift encore élevé |
| v2.1 | Dataset élargi (69 cartes), hyperparamètres identiques | Pas d'amélioration (insuffisance data-only) |
| v3 | Ajout tête HP (w_hp_aux=5, w_attack_hp=10), 69 cartes, 2.14M transitions | MAE HP -70% après clamp fix, drift 5.80 |
| v3.1 | Augmentation w_pass (5 ou 10) | En cours de test |
