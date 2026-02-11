# Feat: Fillers — phrases d'attente prégénérées pour réduire la latence perçue

## Problème

Entre la fin de la parole utilisateur et le début de la réponse audio, il y a un silence de plusieurs secondes (STT → Claude → TTS 1re phrase). Ce silence donne une impression de lenteur.

## Objectif

Jouer un filler audio ("Hmm", "Voyons", "Alors"...) immédiatement après la fin de la parole, pendant que Claude et le TTS travaillent. La latence perçue tombe à quasi zéro.

## Principe

1. Au démarrage du daemon, prégénérer une banque de fillers en WAV
2. Quand l'utilisateur finit de parler, jouer un filler aléatoire instantanément
3. Enchaîner avec la vraie réponse TTS dès qu'elle est prête

## Fillers prévus

```python
FILLERS = {
    "French": ["Hmm.", "Voyons.", "Alors.", "Bonne question.", "Voyons voir."],
    "English": ["Hmm.", "Let me think.", "Well.", "Good question.", "Let's see."],
}
```

Court (1-3 mots), naturel, neutre. Pas de contenu sémantique pour éviter les contradictions avec la vraie réponse.

## Architecture

### Démarrage du daemon (warm-up fillers)

Après le chargement du modèle, le daemon génère tous les fillers en WAV et les stocke dans un dossier cache.

```
~/.cache/jarvis/fillers/
├── fr_00.wav  # "Hmm."
├── fr_01.wav  # "Voyons."
├── fr_02.wav  # "Alors."
├── ...
├── en_00.wav  # "Hmm."
└── en_01.wav  # "Let me think."
```

La génération ne se fait qu'une fois. Au démarrage suivant, si les fichiers existent déjà, on les saute.

### Nouveau message socket : `get_filler`

Le client demande un filler au daemon :

```python
# request
{"action": "get_filler", "language": "French"}

# response
{"status": "ok", "path": "/Users/q/.cache/jarvis/fillers/fr_02.wav"}
```

Le daemon choisit un filler aléatoire pour la langue demandée et renvoie son chemin.

### Talk pipeline : jouer le filler avant la réponse

```
User parle → STT → texte
                      │
                      ├──► play filler (instantané, ~0.5s)
                      │
                      └──► Claude stream → sentences → gen → play
                                                              │
                                                     (enchaîne après filler)
```

### Timeline

```
Avant:
[silence 3-5s.................][phrase1][phrase2]

Après:
[filler ~0.5s][silence ~2-4s..][phrase1][phrase2]
              OU
[filler ~0.5s][phrase1][phrase2]  ← si gen1 finit pendant le filler
```

## Changements

### `src/jarvis/daemon.py` — EDIT

- Après `model = load_model()`, appeler `warm_fillers(model)` pour prégénérer les WAV
- Nouveau handler `get_filler` : retourne un chemin aléatoire
- Stocker les chemins fillers dans un dict `filler_cache`

### `src/jarvis/talk.py` — EDIT

- Dans `conversation_turn()`, avant de lancer le stream Claude, demander un filler au daemon et le jouer en arrière-plan
- Le play_worker attend que le filler finisse avant de jouer la première vraie phrase
- Le filler doit aussi être interruptible via barge-in

### `src/jarvis/handlers.py` — AUCUN CHANGEMENT

### `src/jarvis/cli.py` — AUCUN CHANGEMENT

## Détails d'implémentation

### warm_fillers (daemon.py)

```python
FILLERS = {
    "French": ["Hmm.", "Voyons.", "Alors.", "Bonne question.", "Voyons voir."],
    "English": ["Hmm.", "Let me think.", "Well.", "Good question.", "Let's see."],
}

def warm_fillers(model):
    cache_dir = Path.home() / ".cache" / "jarvis" / "fillers"
    cache_dir.mkdir(parents=True, exist_ok=True)

    filler_cache = {}
    for lang, phrases in FILLERS.items():
        prefix = lang[:2].lower()
        paths = []
        for i, phrase in enumerate(phrases):
            path = cache_dir / f"{prefix}_{i:02d}.wav"
            if not path.exists():
                # Génère le filler via le modèle TTS
                generate_to_wav(model, phrase, lang, str(path))
            paths.append(str(path))
        filler_cache[lang] = paths
    return filler_cache
```

### conversation_turn (talk.py)

```python
async def conversation_turn(text, session_id, language, bundle):
    # 1. Demander un filler au daemon
    filler_resp = send_request({"action": "get_filler", "language": language})
    filler_path = filler_resp.get("path")

    keys = KeyMonitor()
    keys.start()

    # 2. Jouer le filler en tâche de fond
    filler_done = asyncio.Event()
    async def play_filler():
        if filler_path:
            await asyncio.to_thread(play_interruptible, filler_path, keys.barge_in)
        filler_done.set()

    filler_task = asyncio.create_task(play_filler())

    # 3. Lancer Claude stream + pipeline TTS en parallèle
    # Le play_worker attend filler_done avant de jouer la 1re phrase
    ...
```

Note : `play_interruptible` ne doit PAS supprimer le fichier filler (contrairement aux fichiers temp TTS). Ajouter un paramètre `delete=False`.

## Risques et mitigations

| Risque | Mitigation |
|--------|-----------|
| Fillers lents à générer au 1er démarrage | ~5s par filler, ~25s total. One-time cost. Afficher progression |
| Filler sonne faux / robotique | Tester différentes phrases, garder les plus naturelles |
| Filler contredit la réponse | Phrases neutres uniquement (pas de "Oui", "Non", "Bien sûr") |
| Cache corrompu | Vérifier que les WAV sont lisibles au démarrage, regénérer si invalides |
| Latence réseau daemon pour get_filler | Socket Unix local, <1ms |

## Vérification

1. `jah serve` → observer les logs de prégénération des fillers
2. 2e `jah serve` → vérifier que les fillers ne sont pas regénérés (cache hit)
3. `jah talk` → poser une question → entendre un filler immédiatement
4. Le filler est suivi de la vraie réponse sans silence
5. Espace pendant le filler → interruption fonctionne
6. Vérifier `~/.cache/jarvis/fillers/` contient les WAV

## Évolutions futures

- Fillers adaptatifs : choisir le filler en fonction du contexte (question complexe → "Voyons voir", question simple → "Hmm")
- Fillers vocaux personnalisés via `instruct` (ton, style)
- Pool de fillers plus large pour éviter la répétition
