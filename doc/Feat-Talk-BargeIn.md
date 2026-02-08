# Feat: Barge-in — interruption pendant la lecture TTS

## Problème

Quand Claude répond avec plusieurs phrases, l'utilisateur doit attendre la fin de TOUTE la lecture audio avant de pouvoir parler. S'il veut interrompre ou rediriger la conversation, il est bloqué.

## Objectif

L'utilisateur appuie sur Espace pendant que le TTS lit → la lecture s'arrête immédiatement, le système passe en écoute.

## Approches de déclenchement

| Approche | Complexité | Fiabilité | Problème d'écho |
|----------|:----------:|:---------:|:---------------:|
| **Touche clavier (Espace)** | **Faible** | **Parfaite** | **Non** |
| VAD micro (seuil RMS) | Moyenne | Correcte | Oui — faux triggers |
| VAD hardware (Neural Engine M4) | Élevée | Bonne | Partiel |

**Décision : touche Espace pour le PoC.** Zéro faux positif, zéro calibration, zéro écho. Le VAD micro sera une évolution future (nécessite AEC ou casque).

## Architecture

### Pipeline actuelle (Latency1)

```
Claude stream → sentence_queue → gen_worker → audio_queue → play_worker
                                     │                          │
                                  daemon:                   sd.play()
                                  gen fichier               sd.wait()  ← BLOQUANT
```

`sd.play()` + `sd.wait()` = pas d'interruption possible.

### Pipeline cible

```
Claude stream → sentence_queue → gen_worker → audio_queue → play_worker
       │              │               │             │            │
       │         [cancel]         [cancel]      [drain]     [abort]
       │              ▲               ▲             ▲            ▲
       │              └───────────────┴─────────────┴────────────┘
       │                                    │
       ▼                              barge_in Event
  [task.cancel()]                           ▲
                                            │
                                      key_monitor
                                      (thread stdin)
                                      détecte Espace
```

## Détails d'implémentation

### 1. key_monitor : écoute clavier non-bloquante

Un thread dédié met le terminal en mode `cbreak` (caractères sans Enter) et lit les touches. Espace → set l'event.

```python
import sys
import tty
import termios
import threading

class KeyMonitor:
    """Watch for keypress in a background thread."""

    def __init__(self, trigger_key=" "):
        self.trigger_key = trigger_key
        self.barge_in = threading.Event()
        self._thread = None
        self._stop = False
        self._old_settings = None

    def start(self):
        self.barge_in.clear()
        self._stop = False
        self._old_settings = termios.tcgetattr(sys.stdin)
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def _watch(self):
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not self._stop:
                ch = sys.stdin.read(1)
                if ch == self.trigger_key:
                    self.barge_in.set()
                    break
        except (OSError, ValueError):
            pass  # stdin closed or not a TTY

    def stop(self):
        self._stop = True
        if self._old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
```

Points clés :
- Thread daemon — meurt avec le process parent
- `tty.setcbreak()` — lit caractère par caractère, sans echo
- Restaure les settings terminal dans `stop()` (critique pour ne pas casser le terminal)

### 2. play_worker : lecture interruptible

Remplacer `sd.play()` + `sd.wait()` par un `sd.OutputStream` avec écriture par chunks de 40ms. Vérifie le barge-in event entre chaque chunk.

```python
def play_interruptible(path: str, barge_in: threading.Event) -> bool:
    """Play a WAV file, stoppable via barge_in event.

    Returns True if interrupted, False if completed normally.
    """
    data, sr = sf.read(path, dtype="float32")
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
    stream.start()

    chunk_size = int(sr * 0.04)  # 40ms chunks → ~25 checks/sec
    offset = 0

    try:
        while offset < len(data):
            if barge_in.is_set():
                stream.abort()  # Stop immediately, discard buffers
                return True
            end = min(offset + chunk_size, len(data))
            stream.write(data[offset:end])
            offset = end
    finally:
        stream.stop()
        stream.close()
        try:
            os.unlink(path)
        except OSError:
            pass

    return False
```

Latence d'interruption : au plus 40ms (1 chunk). Imperceptible.

### 3. gen_worker : annulation propre

Vérifie le barge-in event avant d'envoyer au daemon. Inutile de générer de l'audio qu'on ne jouera pas.

```python
async def gen_worker(sentence_queue, audio_queue, language, barge_in):
    i = 0
    while True:
        sentence = await sentence_queue.get()
        if sentence is None or barge_in.is_set():
            await audio_queue.put(None)
            break
        path = f"/tmp/jarvis_tts_{os.getpid()}_{i:03d}.wav"
        i += 1
        ok = await asyncio.to_thread(generate_to_file, sentence, language, path)
        if barge_in.is_set():
            # Génération terminée mais on n'en veut plus
            try:
                os.unlink(path)
            except OSError:
                pass
            await audio_queue.put(None)
            break
        if ok:
            await audio_queue.put(path)
```

### 4. play_worker : propagation du barge-in

```python
async def play_worker(audio_queue, barge_in):
    while True:
        path = await audio_queue.get()
        if path is None:
            break
        interrupted = await asyncio.to_thread(play_interruptible, path, barge_in)
        if interrupted:
            # Drain remaining files
            while not audio_queue.empty():
                remaining = audio_queue.get_nowait()
                if remaining is None:
                    break
                try:
                    os.unlink(remaining)
                except OSError:
                    pass
            break
```

### 5. conversation_turn : orchestration

```python
async def conversation_turn(text, session_id, language, bundle):
    opts = ClaudeCodeOptions(...)

    sentence_queue = asyncio.Queue()
    audio_queue = asyncio.Queue()

    keys = KeyMonitor()
    keys.start()

    gen_task = asyncio.create_task(
        gen_worker(sentence_queue, audio_queue, language, keys.barge_in)
    )
    play_task = asyncio.create_task(
        play_worker(audio_queue, keys.barge_in)
    )

    buffer = ""
    new_session_id = session_id

    try:
        async for msg in query(prompt=text, options=opts):
            if keys.barge_in.is_set():
                break  # Stop consuming Claude's response

            # ... process StreamEvent, AssistantMessage, ResultMessage ...

    finally:
        # Flush or cancel
        if keys.barge_in.is_set():
            # Drain sentence_queue
            while not sentence_queue.empty():
                sentence_queue.get_nowait()
        else:
            if buffer.strip():
                await sentence_queue.put(buffer.strip())

        await sentence_queue.put(None)
        await gen_task
        await play_task
        keys.stop()

    if keys.barge_in.is_set():
        # Passer en mode écoute
        reset(bundle)
        print("\n[interrupted — listening...]", file=sys.stderr)
        new_text = listen_until_silence(bundle)
        if new_text:
            print(f"\n> {new_text}", flush=True)
            return await conversation_turn(new_text, new_session_id, language, bundle)

    return new_session_id
```

## Flow complet

```
1. User parle → STT → texte
2. Texte → Claude Code (streaming)
3. Claude streame → phrases extraites → gen_worker → audio files
4. play_worker joue fichier 1 (chunks 40ms, vérifie barge_in)
   ┌─ PENDANT CE TEMPS: key_monitor écoute stdin
   │
   ├─ Cas A: pas d'Espace → phrase 1 finit, phrase 2 joue, etc.
   │
   └─ Cas B: Espace pressé → barge_in.set()
      ├─ play_worker: stream.abort(), drain queue, stop
      ├─ gen_worker: stop, delete fichier en cours
      ├─ Claude streaming: break (sort de la boucle async for)
      ├─ Terminal restauré (tcsetattr)
      ├─ STT reset + listen_until_silence
      └─ Nouvelle conversation_turn avec le texte capturé
```

## UX terminal

Pendant la lecture TTS, afficher un indicateur :

```
> Bonjour, comment ça va ?

Claude: Je vais bien, merci de demander.                [Espace pour interrompre]
```

Quand interrompu :

```
[interrupted — listening...]
> Change de sujet
```

## Fichiers modifiés

### `src/jarvis/talk.py` — REFACTOR

- Nouveau : `KeyMonitor` class (thread daemon, cbreak stdin)
- `play_and_cleanup()` → `play_interruptible()` avec barge-in event
- `gen_worker` : vérifie barge-in avant chaque génération
- `play_worker` : passe le barge-in event, drain si interrompu
- `conversation_turn()` : crée KeyMonitor, gère l'annulation + récursion

### `src/jarvis/cli.py` — EDIT

- Option `--trigger` pour choisir la touche (défaut: Espace)

### Autres fichiers : AUCUN CHANGEMENT

## Risques et mitigations

| Risque | Mitigation |
|--------|-----------|
| Terminal cassé si crash avant tcsetattr restore | `try/finally` + `atexit` handler |
| Espace tapé accidentellement | Indicateur visuel clair, touche configurable |
| Fichiers temp orphelins après cancel | `drain_queue` supprime les fichiers, `finally` blocks |
| STT overflow après interruption | `reset(bundle)` recrée LmGen + StreamTokenizer |
| Claude query continue en background | Coût API marginal, la task se terminera |
| Récursion infinie (barge-in → barge-in → ...) | Profondeur limitée naturellement par l'interaction humaine |

## Vérification

1. `jah talk` → poser une question → Claude répond
2. Pendant la lecture, appuyer Espace → la lecture s'arrête immédiatement
3. Le terminal affiche "[interrupted — listening...]"
4. Parler → le texte est transcrit et envoyé à Claude
5. Vérifier pas de fichiers temp orphelins (`ls /tmp/jarvis_tts_*`)
6. Vérifier que le terminal est restauré après Ctrl+C
7. Vérifier que le STT fonctionne après interruption

## Évolutions futures

- **VAD micro** : détection automatique de parole (nécessite AEC ou casque)
- **VAD hardware** : `kAudioDevicePropertyVoiceActivityDetectionState` (Neural Engine M4)
- **`ClaudeSDKClient.interrupt()`** : interrompre proprement la réponse Claude
- **Double-tap Espace** : premier tap = pause, deuxième = reprendre ou interrompre
- **Touche Echap** : annuler la réponse sans passer en écoute
