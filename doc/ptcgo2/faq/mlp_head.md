# FAQ — MLP Heads dédiées

## Background rapide du projet
Le projet entraîne un world model pour approximer la dynamique du backend de jeu Pokémon TCG POC.
Entrée modèle: état observé + action.
Sorties modèle: changement d'état (`delta_obs`), reward, done, et en v3 une sortie HP dédiée (`delta_hp`).

## C'est quoi `delta_obs` ?
`delta_obs` est la variation prédite du vecteur d'observation entre deux steps.
Formellement: `delta_obs = next_obs - obs`.
Le modèle ne prédit pas directement `next_obs`, il prédit le changement à appliquer à `obs`.

## C'est quoi `delta_hp` ?
`delta_hp` est une sortie spécialisée HP (tête dédiée), typiquement de taille 2:
- variation HP du joueur courant
- variation HP de l'adversaire
Cette sortie est utilisée pour renforcer la précision des transitions de combat.

## Pourquoi prédire des deltas plutôt que l'état complet ?
Prédire un delta simplifie l'apprentissage car beaucoup de dimensions changent peu ou pas à chaque step.
Le modèle apprend "ce qui bouge" au lieu de reconstruire tout l'état depuis zéro.
En pratique, cela stabilise l'entraînement et réduit le bruit sur les dimensions statiques.

## Exemple concret

Farfetch'd (50 HP) attaque Rattata (30 HP) avec Leek Slap (30 dégâts) :



Le delta est presque tout à zéro (94 dims statiques ne bougent jamais).
Seuls les HP adverses changent ici : -0.100 en normalisé = -30 HP en réel = KO.

Si le réseau prédisait l'état complet, il devrait reconstruire les 122 dims à chaque step.
En prédisant le delta, il se concentre sur les ~28 dims qui peuvent bouger.

Pour , la tête dédiée extrait juste les 2 valeurs critiques :


## Qu'est-ce qu'une head dédiée ?
Une head dédiée est une sortie supplémentaire branchée sur le même backbone MLP.
Le backbone extrait des features communes, puis chaque head prédit une cible spécialisée.
Exemple: une head globale pour `delta_obs` et une head HP pour `delta_hp`.

## Quelle différence avec une seule head globale ?
Avec une seule head, toutes les dimensions partagent la même projection finale et le même compromis d'apprentissage.
Avec plusieurs heads, on garde un tronc partagé mais on spécialise la sortie sur des sous-objectifs critiques.
Cela réduit la dilution du signal sur les variables importantes.

## Pourquoi utiliser une head dédiée ?
Pour renforcer la supervision sur une cible qui cause des erreurs produit visibles (ex: HP en attaque).
La head dédiée donne un chemin de gradient plus direct et mieux pondérable.
Elle permet d'améliorer une partie du comportement sans alourdir fortement l'architecture.

## Quand en ajouter une ?
Quand les métriques globales semblent bonnes mais qu'une sous-métrique métier reste mauvaise en rollout.
Quand l'erreur est localisée et répétée (ex: sous-estimation des dégâts, KO ratés).
Pas besoin de head dédiée si le problème est global ou vient d'un bug pipeline.

## Combien de heads faut-il ?
Le minimum efficace.
On commence généralement avec 1 head dédiée sur la variable la plus critique.
On ajoute d'autres heads seulement si un besoin mesuré persiste après benchmark.

## Quel coût en paramètres ?
Faible dans la plupart des cas.
Une head linéaire `Linear(H, K)` ajoute environ `H*K + K` paramètres.
Exemple: `H=256`, `K=2` => 514 paramètres (très faible vs un MLP ~100k+ paramètres).

## Quel coût en entraînement ?
Un léger surcoût de calcul et de mémoire pour les sorties/loss supplémentaires.
Le coût principal vient surtout de la complexité de tuning des poids de loss.
En pratique, le surcoût runtime est souvent marginal.

## Quel coût en inférence/ONNX ?
Nécessite d'exporter et parser une sortie supplémentaire.
Il faut versionner le schéma de sortie pour éviter les incompatibilités côté backend.
Le coût de latence est généralement faible pour une petite head.

## Comment l'utiliser côté backend ?
Le backend fait un routage explicite des sorties.
Exemple: HP depuis `delta_hp`, autres dims depuis `delta_obs`.
Ce routage doit être documenté et testé pour éviter les incohérences.

## Quels risques ?
Sur-pondérer une head peut dégrader les autres objectifs.
Plus de heads = plus de complexité de debugging et de maintenance.
Des gradients contradictoires peuvent apparaître entre heads.

## Comment savoir si ça marche ?
Mesurer des métriques ciblées avant/après (ex: `attack_hp_mae`, `ko_miss_rate`, drift HP).
Valider en rollout dual-backend sur seeds non vus.
Ne pas se baser uniquement sur la val_loss globale.

## Head dédiée vs plus de données ?
Ce sont deux leviers complémentaires.
Plus de données améliore la couverture des cas, la head dédiée améliore la focalisation d'apprentissage.
Si l'erreur est très localisée, la head dédiée donne souvent un gain plus direct.

## Head dédiée et évolutivité long terme
Approche généralement scalable si le nombre de heads reste limité et justifié.
Il faut garder une gouvernance simple: une head = un besoin métier mesuré.
Sinon, l'architecture devient difficile à maintenir.
