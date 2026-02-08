# Spécification : `jah talk` — Interface vocale pour Claude Code

## Objectif

Créer une commande `jah talk` qui transforme Claude Code en assistant vocal interactif. L'utilisateur parle dans le micro, Claude Code répond à voix haute — avec tous ses outils (Bash, Edit, Read, etc.), le contexte projet, et la persistance de session.

## Architecture

```
jah talk (foreground, Python asyncio)
  │
  ├── STT (Kyutai STT 1B, inline MLX)
  │     micro → texte en streaming
  │     détection fin de parole (silence ~1s)
  │
  ├── Claude Code SDK (claude-code-sdk 0.0.25)
  │     ClaudeSDKClient — session bidirectionnelle
  │     query(texte) → receive_response()
  │     outils, contexte, session persistante
  │
  └── TTS daemon (jah serve, process séparé)
        réponse texte → Unix socket → speakers
```

**Contrainte MLX** : STT (Kyutai) et TTS (Qwen3) utilisent MLX mais dans des processus séparés. `jah talk` charge le STT dans son process ; le TTS tourne dans le daemon `jah serve`.

## Dépendances

- `claude-code-sdk>=0.0.25` — SDK Python pour Claude Code (déjà installé)
- `claude` CLI >= 2.1.37 — Claude Code (déjà installé, `/Users/q/.nvm/versions/node/v22.7.0/bin/claude`)
- STT : Kyutai STT 1B via `moshi_mlx` (déjà fonctionnel, `src/jarvis/stt.py`)
- TTS : daemon `jah serve` (déjà fonctionnel, `src/jarvis/daemon.py`)

## SDK Claude Code — API utilisée

### ClaudeSDKClient (session interactive bidirectionnelle)

```python
from claude_code_sdk import ClaudeSDKClient, ClaudeCodeOptions

options = ClaudeCodeOptions(
    # Système
    append_system_prompt="Tu es un assistant vocal. Réponds de façon concise et naturelle, adaptée à la lecture à voix haute. Pas de markdown, pas de blocs de code longs.",
    cwd="/Users/q/Projects/jarvis",

    # Permissions — auto-approve les outils safe
    allowed_tools=["Read", "Glob", "Grep", "WebSearch", "WebFetch"],
    # Les outils dangereux (Bash, Edit, Write) demandent confirmation vocale

    # Session
    continue_conversation=True,  # reprend la dernière session du répertoire

    # Streaming
    include_partial_messages=True,  # StreamEvent avec text_delta
)

async with ClaudeSDKClient(options) as client:
    # Envoyer un message
    client.query("Explique le fichier daemon.py")

    # Recevoir la réponse (streaming)
    async for msg in client.receive_response():
        # msg est un de: AssistantMessage, SystemMessage, ResultMessage, StreamEvent
        ...
```

### Types de messages reçus

| Type | Contenu | Action |
|------|---------|--------|
| `StreamEvent` | `event.delta.type == "text_delta"` → `event.delta.text` | Buffer → TTS par phrases |
| `AssistantMessage` | `content: [TextBlock, ToolUseBlock, ...]` | Message complet (après streaming) |
| `ResultMessage` | `result: str`, `session_id`, `is_error`, `total_cost_usd` | Fin du tour, retour écoute |

### StreamEvent — extraction du texte

```python
if isinstance(msg, StreamEvent):
    event = msg.event
    if event.get("type") == "content_block_delta":
        delta = event.get("delta", {})
        if delta.get("type") == "text_delta":
            text_chunk = delta["text"]
            # → ajouter au buffer TTS
```

### Approbation des outils (can_use_tool)

```python
async def voice_approval(tool_name, tool_input, context):
    """Demande confirmation vocale pour les outils dangereux."""
    if tool_name == "Bash":
        command = tool_input.get("command", "")
        # Synthèse vocale : "Exécuter la commande: git status ?"
        await tts_speak(f"Exécuter: {command[:80]} ?")
        # Écoute réponse : "oui" / "non"
        response = await stt_listen_short()
        if "oui" in response.lower() or "yes" in response.lower():
            return PermissionResultAllow(updated_input=tool_input)
        return PermissionResultDeny(message="Refusé par l'utilisateur")
    # Les autres outils sont auto-approuvés via allowed_tools
    return PermissionResultAllow(updated_input=tool_input)

options = ClaudeCodeOptions(
    can_use_tool=voice_approval,
    allowed_tools=["Read", "Glob", "Grep"],
)
```

### Interruption

```python
# L'utilisateur parle pendant que Claude répond → interrompre
client.interrupt()
```

## Fichiers à créer/modifier

### 1. `src/jarvis/talk.py` — NEW — Orchestrateur vocal

Module principal. Boucle STT → Claude → TTS.

```python
"""jah talk — voice interface for Claude Code."""

import asyncio
from claude_code_sdk import ClaudeSDKClient, ClaudeCodeOptions, StreamEvent, ResultMessage

from jarvis.stt import load_model as load_stt, listen_until_silence
from jarvis.cli import send_request, daemon_is_running


SENTENCE_DELIMITERS = {'.', '!', '?', ':', ';', '\n'}


async def talk(system_prompt=None, continue_session=True):
    """Main voice loop: listen → Claude → speak → repeat."""

    # 1. Vérifier que le daemon TTS tourne
    if not daemon_is_running():
        print("Error: TTS daemon not running. Start with: jah serve", file=sys.stderr)
        return

    # 2. Charger STT
    print("Loading STT model...", file=sys.stderr)
    stt_bundle = load_stt()
    print("STT ready.", file=sys.stderr)

    # 3. Configurer Claude Code
    options = ClaudeCodeOptions(
        append_system_prompt=system_prompt or VOICE_SYSTEM_PROMPT,
        allowed_tools=["Read", "Glob", "Grep", "WebSearch", "WebFetch"],
        continue_conversation=continue_session,
        include_partial_messages=True,
    )

    async with ClaudeSDKClient(options) as client:
        print("Ready. Speak now.", file=sys.stderr)

        while True:
            # 4. Écoute (STT) jusqu'à silence
            user_text = await asyncio.to_thread(listen_until_silence, stt_bundle)

            if not user_text.strip():
                continue

            print(f"\n> {user_text}", file=sys.stderr)

            # 5. Envoyer à Claude
            client.query(user_text)

            # 6. Recevoir et lire la réponse
            sentence_buffer = ""
            async for msg in client.receive_response():
                if isinstance(msg, StreamEvent):
                    chunk = extract_text_delta(msg)
                    if chunk:
                        sentence_buffer += chunk
                        # Envoyer au TTS dès qu'on a une phrase complète
                        sentence_buffer = flush_sentences(sentence_buffer)

                elif isinstance(msg, ResultMessage):
                    # Flush le reste du buffer
                    if sentence_buffer.strip():
                        tts_speak(sentence_buffer.strip())
                    break

            # 7. Attendre que le TTS finisse, puis recommencer
```

**Fonctions clés :**

- `listen_until_silence(stt_bundle) -> str` : variante de `stt.listen()` qui retourne le texte quand le silence est détecté (~12 steps vides = 1s de silence)
- `flush_sentences(buffer) -> str` : découpe le buffer sur `.!?;\n`, envoie chaque phrase au TTS, retourne le reste
- `tts_speak(text)` : envoie le texte au daemon TTS via Unix socket (réutilise `cli.send_request`)
- `extract_text_delta(stream_event) -> str|None` : extrait le texte d'un StreamEvent

### 2. `src/jarvis/stt.py` — EDIT — Ajouter `listen_until_silence()`

Nouvelle fonction basée sur `listen()` existante, mais qui retourne le texte transcrit au lieu de le print, et s'arrête après détection de silence.

```python
def listen_until_silence(bundle, silence_threshold=12, max_duration=30):
    """Listen until silence detected. Returns transcribed text.

    Args:
        silence_threshold: nombre de steps consécutifs sans token texte (12 ≈ 1s)
        max_duration: durée max en secondes
    Returns:
        str: texte transcrit
    """
    # Même boucle que listen() mais :
    # - accumule le texte au lieu de print()
    # - compte les steps sans token texte
    # - retourne quand silence_count >= silence_threshold
    # - retourne aussi si max_duration atteint
```

### 3. `src/jarvis/cli.py` — EDIT — Ajouter commande `talk`

```python
@cli.command()
@click.option("--system-prompt", default=None, help="Custom system prompt for Claude")
def talk(system_prompt):
    """Voice conversation with Claude Code."""
    from jarvis.talk import talk as voice_talk
    asyncio.run(voice_talk(system_prompt=system_prompt))
```

Ajouter `"talk"` dans `SUBCOMMANDS`.

### 4. `pyproject.toml` — EDIT — Ajouter dépendance

```toml
"claude-code-sdk>=0.0.25",
```

## Flux détaillé d'un tour de conversation

```
┌─ ÉCOUTE ─────────────────────────────────────────┐
│                                                    │
│  sounddevice.InputStream (24kHz, 80ms chunks)      │
│       │                                            │
│       ▼                                            │
│  StreamTokenizer.encode() → get_encoded()          │
│       │                                            │
│       ▼                                            │
│  gen.step(audio_tokens, ct) → text_token           │
│       │                                            │
│       ▼                                            │
│  Accumule texte, compte silence                    │
│  silence >= 12 steps (1s) → FIN ÉCOUTE            │
│                                                    │
└─── user_text ─────────────────────────────────────┘
         │
         ▼
┌─ CLAUDE CODE ─────────────────────────────────────┐
│                                                    │
│  client.query(user_text)                           │
│       │                                            │
│       ▼                                            │
│  async for msg in client.receive_response():       │
│       │                                            │
│       ├── StreamEvent(text_delta) → buffer          │
│       │     phrase complète? → TTS                  │
│       │                                            │
│       ├── ToolUseBlock → "J'exécute..."            │
│       │     (optionnel: confirmation vocale)        │
│       │                                            │
│       └── ResultMessage → flush buffer → TTS        │
│                                                    │
└─── réponse complète ──────────────────────────────┘
         │
         ▼
┌─ TTS ─────────────────────────────────────────────┐
│                                                    │
│  send_request({                                    │
│      "action": "generate",                         │
│      "text": phrase,                               │
│      "language": "French"                          │
│  })                                                │
│       │                                            │
│       ▼                                            │
│  daemon TTS → sounddevice speakers                 │
│                                                    │
└───────────────────────────────────────────────────┘
         │
         ▼
    Retour à ÉCOUTE
```

## Détection de fin de parole

Le modèle STT émet des tokens texte quand il reconnaît de la parole. Pendant le silence, il émet des tokens 0 ou 3 (padding).

```python
silence_count = 0
SILENCE_THRESHOLD = 12  # 12 steps × 80ms = ~1 seconde

for each step:
    text_token = gen.step(audio_tokens, ct)[0].item()
    if text_token not in (0, 3):
        silence_count = 0   # reset
        accumulated_text += decode(text_token)
    else:
        silence_count += 1

    if silence_count >= SILENCE_THRESHOLD and accumulated_text:
        return accumulated_text  # l'utilisateur a fini de parler
```

## Envoi au TTS par phrases (streaming)

Pour réduire la latence, on n'attend pas la réponse complète de Claude. On découpe en phrases et on envoie au TTS au fur et à mesure :

```python
def flush_sentences(buffer):
    """Envoie les phrases complètes au TTS, retourne le reste."""
    last_delim = -1
    for i, c in enumerate(buffer):
        if c in '.!?;\n':
            last_delim = i

    if last_delim == -1:
        return buffer  # pas de phrase complète

    sentence = buffer[:last_delim + 1].strip()
    remainder = buffer[last_delim + 1:]

    if sentence:
        tts_speak(sentence)

    return remainder
```

## System prompt vocal

```python
VOICE_SYSTEM_PROMPT = """Tu es un assistant vocal. Règles :
- Réponds de façon concise et naturelle, adaptée à la lecture à voix haute
- Pas de markdown (pas de **, ##, ```, etc.)
- Pas de listes à puces sauf si demandé
- Pas de blocs de code longs — décris plutôt ce que tu fais
- Quand tu utilises un outil, dis brièvement ce que tu fais
- Réponds dans la langue de l'utilisateur
"""
```

## Gestion des interruptions

Si l'utilisateur parle pendant que Claude répond (le TTS joue), on pourrait :
1. Détecter l'audio entrant (VAD) pendant la lecture TTS
2. Appeler `client.interrupt()` pour arrêter Claude
3. Arrêter le TTS (envoyer `{"action": "stop"}` au daemon)
4. Reprendre l'écoute STT

**Phase 1** : pas d'interruption. L'utilisateur attend la fin de la réponse.
**Phase 2** : ajouter l'interruption.

## Options CLI

```
jah talk [OPTIONS]

Options:
  --system-prompt TEXT   Custom system prompt for Claude
  --language TEXT         Preferred language (default: auto)
  --continue / --new     Continue last session or start fresh (default: continue)
```

## Vérification

1. `jah serve` tourne dans un terminal
2. `jah talk` dans un autre terminal → charge STT, "Ready. Speak now."
3. Parler : "Quels fichiers sont dans le projet ?" → Claude utilise Glob/Read, répond à voix haute
4. Enchaîner : "Explique le daemon" → Claude utilise le contexte de la session
5. Ctrl+C → arrêt propre

## Risques et mitigations

| Risque | Mitigation |
|--------|-----------|
| Latence STT→Claude→TTS | Streaming par phrases, pas attendre la réponse complète |
| Claude répond trop long | `append_system_prompt` demande concision ; `max_turns` optionnel |
| Bruit micro déclenche Claude | `silence_threshold` + minimum de texte avant envoi |
| Outils dangereux | `allowed_tools` whitelist, `can_use_tool` pour confirmation vocale |
| Session perdue | `continue_conversation=True` par défaut |
| MLX conflict STT/TTS | Processus séparés (talk = STT, serve = TTS) |
