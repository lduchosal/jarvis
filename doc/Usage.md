# Usage: q3tts

## Generate audio

```bash
# Basic (English, default voice)
uv run q3tts.py "Hello world"

# French
uv run q3tts.py -l French "On est reparti !"

# With voice instruction
uv run q3tts.py -i "deep masculine voice with warm cheerful intonation" "Hello"

# French + voice instruction
uv run q3tts.py -l French -i "deep masculine voice with warm cheerful intonation" "On est reparti !"

# Custom output filename
uv run q3tts.py -o greeting.wav "Hello world"

# Piped input
echo "Hello world" | uv run q3tts.py

# Verbose (shows progress)
uv run q3tts.py -v "Hello world"
```

Output files are auto-incremented: `output.wav`, `output-2.wav`, `output-3.wav`, ...

## Play audio

```bash
uv run play.py 1cestparti.wav
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `-o` | Output filename | `output.wav` |
| `-l` | Language (`English`, `French`, `Chinese`, ...) | `English` |
| `-i` | Voice instruction (e.g. `"deep low voice"`) | none |
| `-v` | Verbose output | off |

## Daemon mode (streaming, low latency)

Le daemon garde le modèle en mémoire — élimine ~5.4s de startup par requête. L'audio sort directement du speaker en streaming.

### Lancer le daemon

```bash
# Terminal 1 : démarrer le daemon (reste en foreground)
uv run src/q3tts_daemon.py
```

### Envoyer du texte

```bash
# Terminal 2 : générer et jouer sur le speaker
uv run src/q3tts_client.py "Hello world"

# French + voice instruction
uv run src/q3tts_client.py -l French -i "deep masculine voice" "Bonjour"

# Sauvegarder en fichier
uv run src/q3tts_client.py -o greeting.wav "Hello world"

# Piped input
echo "Hello world" | uv run src/q3tts_client.py
```

### Contrôle du daemon

```bash
# Vérifier si le daemon tourne
uv run src/q3tts_client.py status

# Arrêter le daemon proprement
uv run src/q3tts_client.py stop
```

### Hot-reload

Modifier `src/handlers.py` et envoyer une nouvelle requête — le daemon recharge automatiquement le code sans redémarrer. Pas besoin de relancer quoi que ce soit.
