# Usage: jah

## Daemon mode (streaming, low latency)

Le daemon garde le modèle en mémoire — élimine ~5.4s de startup par requête. L'audio sort directement du speaker en streaming.

### Lancer le daemon

```bash
# Terminal 1 : démarrer le daemon (reste en foreground)
jah serve
```

### Envoyer du texte

```bash
# Terminal 2 : générer et jouer sur le speaker
jah "Hello world"

# French + voice instruction
jah speak -l French -i "deep masculine voice" "Bonjour"

# Sauvegarder en fichier
jah speak -o greeting.wav "Hello world"

# Piped input
echo "Hello world" | jah speak
```

### Contrôle du daemon

```bash
# Vérifier si le daemon tourne
jah status

# Arrêter le daemon proprement
jah stop
```

### Stress test

```bash
# Lancer les tests de stabilité (silent = pas d'audio)
jah stress --silent

# Avec options
jah stress --silent --delay 0.2 --category short
jah stress --report results.json
```

### Hot-reload

Modifier `src/jarvis/handlers.py` et envoyer une nouvelle requête — le daemon recharge automatiquement le code sans redémarrer.

## Options (speak)

| Flag | Description | Default |
|------|-------------|---------|
| `-o` | Output filename | speakers |
| `-l` | Language (`English`, `French`, `Chinese`, ...) | `English` |
| `-i` | Voice instruction (e.g. `"deep masculine voice"`) | none |

## Scripts de développement

```bash
# Installer les dépendances
pdm install

# Lancer les tests
pdm run test

# Linting
pdm run lint

# Stress test
pdm run stress
```

## Standalone (legacy)

```bash
# Génération one-shot (recharge le modèle à chaque appel)
python q3tts.py "Hello world"
python q3tts.py -l French -i "deep masculine voice" "Bonjour"

# Jouer un fichier audio
python play.py output.wav
```
