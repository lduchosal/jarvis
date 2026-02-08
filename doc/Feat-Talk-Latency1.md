# Feat: Pipeline TTS — réduction de latence `jah talk`

## Problème

Actuellement, chaque phrase passe par un cycle séquentiel :

```
[gen1 3s][play1 2s][gen2 3s][play2 2s][gen3 3s][play3 2s] = 15s total
```

La génération audio et la lecture sont bloquantes et séquentielles.
L'utilisateur attend `gen + play` avant d'entendre chaque phrase.

## Observation clé

Le daemon TTS **skip déjà le playback** quand le paramètre `output` est défini :

```python
# handlers.py ligne 30
play_audio = output_path is None  # play on speaker only if no output file
```

On peut exploiter ce mécanisme : le daemon génère et sauvegarde dans un fichier temp,
le client joue le fichier pendant que le daemon génère la phrase suivante.

## Architecture cible

Pipeline 3 étages, toutes les étapes concurrentes :

```
Claude stream → [sentence_queue] → gen_worker → [audio_queue] → play_worker
                                        │                            │
                                   daemon: génère                sounddevice:
                                   audio → fichier               joue le fichier
```

### Timeline

```
Avant (séquentiel):
[gen1 3s][play1 2s][gen2 3s][play2 2s][gen3 3s][play3 2s] = 15s

Après (pipeline):
Claude: [stream text....................]
Daemon: [gen1 3s][gen2 3s][gen3 3s]
Play:            [play1 2s][play2 2s][play3 2s]
                                               = 11s, -27%
```

- La lecture de sentence 1 chevauche la génération de sentence 2
- Le daemon reste mono-thread (pas de changement)
- Gain : `sum(play_i)` économisé sauf pour la dernière phrase

## Changements

### `src/jarvis/handlers.py` — AUCUN CHANGEMENT

Le mécanisme `output` + skip playback existe déjà.

### `src/jarvis/talk.py` — REFACTOR tts_worker

Remplacer le `tts_worker` unique par deux workers :

**gen_worker** : sentence_queue → daemon (output=tempfile) → audio_queue
```python
async def gen_worker(sentence_queue, audio_queue, language):
    i = 0
    while True:
        sentence = await sentence_queue.get()
        if sentence is None:
            await audio_queue.put(None)
            break
        path = f"/tmp/jarvis_tts_{os.getpid()}_{i:03d}.wav"
        i += 1
        resp = await asyncio.to_thread(send_request, {
            "action": "generate",
            "text": sentence,
            "language": language,
            "output": path,
        })
        if resp.get("status") == "ok":
            await audio_queue.put(path)
```

**play_worker** : audio_queue → sounddevice play → cleanup fichier
```python
async def play_worker(audio_queue):
    while True:
        path = await audio_queue.get()
        if path is None:
            break
        data, sr = sf.read(path)
        await asyncio.to_thread(play_and_wait, data, sr)
        os.unlink(path)
```

### `src/jarvis/daemon.py` — AUCUN CHANGEMENT

### `src/jarvis/cli.py` — AUCUN CHANGEMENT

## Dépendances

- `soundfile` (déjà installé, utilisé par handlers.py)
- `sounddevice` (déjà installé, utilisé par stt.py)

## Vérification

1. `jah serve` dans un terminal
2. `jah talk` dans un autre
3. Poser une question → observer que la 1ère phrase joue PENDANT que le daemon génère la 2ème
4. Vérifier qu'il n'y a pas de fichiers temp orphelins dans `/tmp/`
5. Ctrl+C → arrêt propre, cleanup des fichiers temp

## Optimisation future (phase 2)

**Streaming audio chunks over socket** : au lieu de générer toute la phrase puis sauvegarder,
le daemon enverrait les chunks audio au fil de la génération. Le client jouerait les chunks
dès réception. Cela réduirait le time-to-first-audio de ~3s (sentence complète) à ~0.3s
(premier chunk). Nécessite un changement de protocole (binary frames).
