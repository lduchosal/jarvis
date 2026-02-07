# Plan : Implémentation du daemon TTS (v1)

## Contexte

Actuellement, chaque appel `uv run q3tts.py` recharge Python (~2.6s) + le modèle (~2.7s). Le daemon garde tout en mémoire et élimine 5.4s de latence par requête. On implémente la spec définie dans `doc/Architecture.md` et `doc/Feat-Hotswap-Daemon.md`.

## Fichiers à créer

### 1. `src/q3tts_daemon.py` — Le shell stable

Le daemon long-lived. Responsabilités :
- Charger le modèle une fois au démarrage via `load_model("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")`
- Bind Unix socket `~/.q3tts.sock`
- Boucle : accept → lire requête (length-prefixed JSON, 4 bytes big-endian uint32) → `importlib.reload(handlers)` → `handlers.handle(model, request)` → répondre → close
- Signal handling SIGTERM/SIGINT : cleanup socket, exit propre
- Erreurs dans handlers → log + réponse erreur, daemon survit

### 2. `src/handlers.py` — Le cerveau hot-reloadé

Contrat : `handle(model, request: dict) -> dict`
- Parse la requête (text, language, instruct, output)
- Streaming audio via `sd.OutputStream` + `model.generate_voice_design()`
- Optionnel : sauvegarder en fichier si `output` est spécifié
- Retourne `{"status": "ok"}` ou erreur

### 3. `src/q3tts_client.py` — Le client CLI

Point d'entrée CLI (click) :
- `q3tts serve` → lance le daemon en foreground
- `q3tts "texte"` → envoie au daemon via socket
- `q3tts -o out.wav "texte"` → idem avec sauvegarde fichier
- `q3tts stop` → shutdown propre du daemon
- `q3tts status` → check si le daemon tourne
- Fallback : si daemon pas lancé et pas de `serve`, mode inline (comportement actuel)

## Protocole socket

Wire format : `[4 bytes uint32 big-endian length][UTF-8 JSON payload]`

Requête : `{"action": "generate", "text": "...", "language": "English", "instruct": null, "output": null}`
Réponse : `{"status": "ok"}` ou `{"status": "error", "message": "..."}`

## Dépendances

Mêmes que `q3tts.py` + `sounddevice` pour le streaming. Inline PEP 723 dans chaque fichier.

## Vérification

1. `uv run src/q3tts_client.py serve` → daemon démarre, affiche "model loaded" + "listening"
2. Dans un autre terminal : `uv run src/q3tts_client.py "Bonjour"` → audio sort du speaker
3. Modifier `src/handlers.py` → re-envoyer texte → nouveau code pris en compte
4. `uv run src/q3tts_client.py stop` → daemon s'arrête proprement, socket nettoyé
5. Erreur dans handlers.py → daemon survit, affiche l'erreur
