# Spec : Détection de silence intelligente (Gate RMS)

## Problème

La détection de silence actuelle dans `listen_until_silence()` est basée sur les tokens padding du modèle STT (tokens 0 et 3). Après 15 tokens vides consécutifs (~1.2s), on considère que l'utilisateur a fini de parler.

Problèmes observés :
- **Coupure prématurée** : le modèle STT peut émettre des tokens padding entre deux mots ou pendant une hésitation, causant une coupure en milieu de phrase.
- **Chauffe CPU/GPU** : le modèle STT tourne en permanence sur chaque chunk audio même quand personne ne parle. Inférence MLX inutile = GPU Metal sollicité pour rien.

## Solution proposée

### Gate d'énergie audio (RMS) avant inférence

Avant de passer un chunk au modèle STT, calculer l'énergie RMS du PCM brut. Si l'énergie est sous un seuil (= silence micro), on skip l'inférence.

```
rms = sqrt(mean(pcm ** 2))
if rms < seuil:
    skip inférence STT
```

**Avantages** :
- Zéro dépendance, on a déjà le PCM dans la callback
- Réduit la charge GPU quand personne ne parle
- Calcul trivial (~0.01ms par chunk)

**Paramètres à tuner** :
- `rms_threshold` : seuil d'énergie (à calibrer, probablement ~0.005-0.02)
- `silence_duration` : durée de silence avant coupure (~2.0s)
- Garder un minimum de chunks silencieux envoyés au modèle pour que le STT puisse finaliser sa transcription

## Implémentation

### Fichier `src/jarvis/stt.py`

Modifier `listen_until_silence()` :

1. Calculer le RMS de chaque chunk PCM
2. Si RMS < seuil, incrémenter un compteur de silence basé sur le temps réel (pas sur les tokens)
3. Si RMS >= seuil, reset le compteur et passer le chunk au modèle STT normalement
4. Ne couper que si silence confirmé pendant ~2s ET du texte a déjà été transcrit

### Dépendances

Aucune. Numpy est déjà utilisé.

### Flux modifié

```
chunk PCM 24kHz du micro
    |
    +-- RMS < seuil ? -> incrémenter compteur silence, skip inférence STT
    |
    +-- RMS >= seuil -> reset compteur, encode -> tokens -> STT inférence -> texte
    |
    +-- Si silence pendant 2s ET texte accumulé -> fin de tour
```

## Alternatives considérées

| Approche | Pour | Contre |
|---|---|---|
| Augmenter le seuil fixe (15->25) | Trivial | Ne résout pas les faux positifs ni la chauffe |
| **RMS seul** | **Zéro dépendance, simple** | **Sensible au bruit ambiant** |
| WebRTC VAD | Léger | API plus complexe, dépendance externe |

## Risques

- **Calibration RMS** : le seuil dépend du micro et de l'environnement. Prévoir un mode calibration ou un seuil adaptatif.
- **Bruit ambiant** : si l'environnement est bruyant, le RMS peut rester au-dessus du seuil même sans parole. Solution possible : seuil adaptatif basé sur le bruit de fond mesuré au démarrage.
