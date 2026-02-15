# FAQ — Training Loss

## C'est quoi la loss ?

La loss (fonction de coût) mesure à quel point le modèle se trompe. C'est un nombre : plus il est bas, mieux le modèle prédit. L'objectif du training est de faire baisser ce nombre le plus possible.

Elle est utilisée à trois moments :
- **Training** : calculée à chaque batch, elle sert à calculer les gradients et mettre à jour les poids du réseau. C'est le moteur de l'apprentissage.
- **Validation** : calculée sur des données non vues pendant le training, sans mise à jour des poids. Elle sert à vérifier que le modèle généralise et ne fait pas du par-coeur (overfitting).
- **Inférence (jeu réel)** : la loss n'est plus utilisée. Le modèle est figé et on utilise uniquement ses prédictions.

Analogie : c'est comme un professeur qui corrige des copies. Pendant le training, l'élève (le modèle) apprend de ses erreurs. En validation, il passe un examen blanc. En inférence, c'est le jour de l'examen final — plus de corrections, il doit se débrouiller seul.

## Quelle est la loss totale utilisée ?

`loss = loss_delta + loss_reward + 0.5 * loss_done + 2.0 * loss_pass + 5.0 * loss_ko + 5.0 * loss_hp_aux + 10.0 * loss_attack_hp`

## Tableau des termes de loss

| Terme | Poids | Type | Ce qu'il mesure | Pourquoi |
|-------|-------|------|-----------------|----------|
| `loss_delta` | 1.0 | MSE pondérée | Erreur sur le delta d'observation complet (122 dims) | Prédiction principale de l'état. Déjà pondérée par échantillon : ×3 attaques, ×10 terminales, ×20 KO |
| `loss_reward` | 1.0 | MSE | Erreur sur le reward prédit | Prédire +1 (victoire), -1 (défaite), 0 (en cours), -0.01 (infraction) |
| `loss_done` | 0.5 | BCE | Erreur sur la prédiction de fin de partie | Sortie binaire (oui/non), BCE adaptée. Poids 0.5 car BCE a une échelle plus grande que MSE |
| `loss_pass` | 2.0 | MSE | Delta HP prédit quand action = Pass | Pass ne doit jamais modifier les HP. Corrige les "dégâts fantômes" |
| `loss_ko` | 5.0 | ReLU | HP adverses prédits > 0 quand la cible est KO | Empêche les KO ratés (Pokémon mort qui survit) |
| `loss_hp_aux` | 5.0 | Huber | Erreur de la tête HP dédiée (`delta_hp[2]`) | Double supervision sur les HP : signal plus fort et plus direct sur la variable critique |
| `loss_attack_hp` | 10.0 | Huber | Erreur HP adverse pendant les steps d'attaque | Corrige la sous-estimation systématique des dégâts. Poids le plus élevé = priorité maximale |

## Types de loss utilisés

| Type | Formule simplifiée | Quand l'utiliser |
|------|-------------------|------------------|
| MSE | erreur² moyenne | Valeurs continues (HP, reward, énergie). Punit fort les grosses erreurs |
| BCE | -log(probabilité) | Sorties binaires oui/non (done). Mesure la confiance d'une probabilité |
| Huber | MSE si petite erreur, linéaire si grosse | Valeurs continues avec des extrêmes. Robuste aux outliers (ex: dégâts de 10 à 100) |
| ReLU | max(0, valeur) | Pénalité à sens unique : ne punit que les valeurs positives (HP qui restent au-dessus de 0) |

## Notes

- `loss_delta` est pondérée à deux niveaux : par échantillon (×3/×10/×20) ET par les loss auxiliaires (`pass`, `ko`, `hp_aux`, `attack_hp`)
- Les poids (0.5, 2, 5, 10) équilibrent l'importance produit, pas l'échelle mathématique
- `w_hp_aux` et `w_attack_hp` sont configurables en CLI (`--w-hp-aux`, `--w-attack-hp`)

## Comment lire les metriques de validation associees ?

Les metriques les plus utiles pour ce projet:
- `val/pass_hp_mae`: devrait tendre vers 0
- `val/ko_miss_rate`: cible 0
- `val/attack_hp_mae`: devrait baisser nettement
- `val/done_acc`: devrait rester tres eleve

La decision finale se fait surtout en rollout dual-backend, pas uniquement sur la val_loss globale.
