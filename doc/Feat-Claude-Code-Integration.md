# Spécification : Interface vocale pour Claude Code

## Objectif

Ajouter une interface vocale à Claude Code : l'utilisateur parle dans le micro, Claude Code répond à voix haute — avec tous ses outils (Bash, Edit, Read, etc.), le contexte projet, et la persistance de session.

## Contraintes

- **TUI** : ne pas perdre le TUI de Claude Code (couleurs, preview outils, interactivité)
- **MLX** : STT (Kyutai) et TTS (Qwen3) utilisent MLX mais doivent tourner dans des processus séparés (MLX n'est pas thread-safe)
- **Outils** : conserver l'accès à tous les outils Claude Code (Bash, Edit, Read, Glob, etc.)
- **Session** : conserver la persistance de session et le contexte projet
- **Dual input** : l'utilisateur peut parler OU taper — les deux doivent coexister

## Briques disponibles

| Brique | Status | Localisation |
|--------|--------|-------------|
| STT Kyutai 1B | Done | `src/jarvis/stt.py` — `load_model()`, `listen()`, `listen_until_silence()` |
| TTS Qwen3 daemon | Done | `src/jarvis/daemon.py` — Unix socket, streaming audio |
| CLI `jah speak` | Done | `src/jarvis/cli.py` — pipe stdin ou argument |
| CLI `jah listen` | Done | `src/jarvis/cli.py` — transcription streaming |
| CLI `jah echo` | Done | `src/jarvis/cli.py` — STT → TTS en boucle |
| Claude Code SDK | Installé | `claude-code-sdk 0.0.25` — `ClaudeSDKClient`, `query()` |
| Claude Code CLI | Installé | `claude 2.1.37` — TUI interactif |

## Dépendances

- `claude-code-sdk>=0.0.25` — SDK Python pour Claude Code (déjà installé)
- `claude` CLI >= 2.1.37 — Claude Code (déjà installé)
- STT : Kyutai STT 1B via `moshi_mlx` (déjà fonctionnel)
- TTS : daemon `jah serve` (déjà fonctionnel)

---

## Alternatives explorées

### A. SDK `ClaudeSDKClient` — REPL custom (`jah talk`)

Remplace le terminal par un REPL Python qui orchestre STT → Claude SDK → TTS.

```
jah talk (foreground, Python asyncio)
  ├── STT Kyutai (inline, listen_until_silence)
  ├── ClaudeSDKClient (session bidirectionnelle, streaming)
  └── TTS daemon (Unix socket)
```

**Flow :** micro → STT → `client.query(text)` → `receive_response()` → stream text_delta → TTS par phrases → retour écoute.

**SDK API clé :**
```python
from claude_code_sdk import ClaudeSDKClient, ClaudeCodeOptions

opts = ClaudeCodeOptions(
    append_system_prompt="Réponds brièvement, pour la voix.",
    allowed_tools=["Read", "Glob", "Grep", "Bash"],
    continue_conversation=True,
    include_partial_messages=True,
    can_use_tool=voice_approval,  # confirmation vocale pour outils dangereux
)

async with ClaudeSDKClient(opts) as client:
    client.query("Explique daemon.py")
    async for msg in client.receive_response():
        # StreamEvent → text_delta → TTS
        # ResultMessage → fin du tour
    client.interrupt()  # si l'utilisateur parle pendant la réponse
```

**Types de messages :**

| Type | Contenu | Action |
|------|---------|--------|
| `StreamEvent` | `event.delta.text` (text_delta) | Buffer → TTS par phrases |
| `AssistantMessage` | `content: [TextBlock, ToolUseBlock, ...]` | Message complet |
| `ResultMessage` | `result`, `session_id`, `is_error`, `cost` | Fin du tour |

### B. tmux + injection de touches (`jah dictate`)

Claude Code tourne normalement dans un pane tmux. Un process parallèle écoute le micro et injecte le texte transcrit via `tmux send-keys`.

```
┌──────────────────────────┐
│  claude (TUI normal)     │  pane 0
│  > _                     │
├──────────────────────────┤
│  jah dictate             │  pane 1
│  Listening...            │
│  > "refactore le daemon" │
│    → tmux send-keys      │
└──────────────────────────┘
```

**Flow :** micro → STT → `listen_until_silence()` → `tmux send-keys -t 0 "texte" Enter` → Claude Code reçoit l'input comme du clavier.

**Pour le TTS (sortie vocale) :** hook Claude Code `Notification` ou post-processing du terminal output.

### C. Serveur MCP — outils `listen` et `speak`

Un serveur MCP fournit des outils vocaux à Claude Code. Claude décide quand écouter et parler.

```
Claude Code (TUI intact)
  ├── outil MCP: listen(duration?, silence_threshold?)
  │     → active micro, STT, retourne {"text": "..."}
  ├── outil MCP: speak(text, language?)
  │     → envoie au daemon TTS, retourne {"status": "ok"}
  └── tous les autres outils intacts
```

**Flow :** l'utilisateur tape "écoute-moi" → Claude appelle `listen()` → micro s'active → STT transcrit → Claude reçoit le texte → traite → appelle `speak(réponse)` → TTS lit.

**Implémentation MCP :**
```python
from claude_code_sdk import create_sdk_mcp_server, tool

@tool("listen")
def listen(duration: float = 30, silence_threshold: int = 15):
    """Écoute le micro et retourne le texte transcrit."""
    bundle = get_stt_bundle()  # singleton, chargé une fois
    text = listen_until_silence(bundle, silence_threshold, duration)
    return {"text": text}

@tool("speak")
def speak(text: str, language: str = "French"):
    """Lit le texte à voix haute via le daemon TTS."""
    resp = send_request({"action": "generate", "text": text, "language": language})
    return resp
```

**Configuration Claude Code (`.claude/settings.local.json`) :**
```json
{
  "mcpServers": {
    "jarvis-voice": {
      "command": ".venv/bin/python",
      "args": ["-m", "jarvis.mcp_server"]
    }
  }
}
```

### D. Hooks Claude Code — TTS automatique en sortie

Utilise le système de hooks Claude Code pour lire automatiquement chaque réponse à voix haute. L'input reste au clavier.

```
Claude Code (TUI intact, clavier normal)
  └── hook post-response:
        extrait le texte → jah speak
```

**Configuration hooks :**
```json
{
  "hooks": {
    "Notification": [{
      "matcher": "",
      "command": "echo \"$CLAUDE_NOTIFICATION\" | jah speak -l French"
    }]
  }
}
```

Limité : seulement TTS en sortie, pas de STT en entrée. Mais peut se combiner avec B ou C.

### E. SDK `query()` one-shot avec session resume

Appels `claude -p` successifs avec `--resume SESSION_ID` pour maintenir le contexte.

```python
from claude_code_sdk import query, ClaudeCodeOptions

async for msg in query(
    prompt=text,
    options=ClaudeCodeOptions(resume=session_id)
):
    # extraire texte → TTS
```

**Limité :** pas de streaming bidirectionnel, pas d'interruption, pas de TUI.

### F. Combinaison MCP + tmux (C + B)

Le meilleur des deux mondes : Claude Code a des outils vocaux MCP ET on peut dicter via tmux.

```
Claude Code (TUI intact)
  ├── MCP: speak(text) → TTS (Claude décide quand parler)
  ├── MCP: listen() → STT (Claude décide quand écouter)
  └── tmux: jah dictate injecte du texte (l'utilisateur dicte quand il veut)
```

L'utilisateur peut :
- Taper au clavier normalement
- Dicter via `jah dictate` (injection tmux)
- Demander à Claude d'écouter : "utilise l'outil listen"
- Claude lit ses réponses à voix haute via `speak`

---

## Tableau comparatif

| Critère | A. SDK REPL | B. tmux | C. MCP | D. Hooks | E. query() | F. MCP+tmux |
|---------|:-----------:|:-------:|:------:|:--------:|:----------:|:-----------:|
| **TUI Claude Code** | Non | Oui | Oui | Oui | Non | Oui |
| **STT (entrée voix)** | Oui | Oui | Oui* | Non | Oui | Oui |
| **TTS (sortie voix)** | Oui | Partiel | Oui | Oui | Oui | Oui |
| **Outils Claude** | Oui | Oui | Oui | Oui | Oui | Oui |
| **Session persistante** | Oui | Oui | Oui | Oui | Partiel | Oui |
| **Dual input clavier+voix** | Difficile | Oui | Oui* | Clavier seul | Non | Oui |
| **Interruption vocale** | Oui | Non | Non | Non | Non | Non |
| **Complexité** | Moyenne | Faible | Moyenne | Faible | Faible | Moyenne |
| **Dépendance externe** | SDK | tmux | Aucune | Aucune | SDK | tmux |
| **Claude contrôle la voix** | Non | Non | Oui | Non | Non | Oui |
| **Latence** | Bonne | Bonne | Variable** | Bonne | Mauvaise | Bonne |

\* STT déclenché par Claude (pas par l'utilisateur directement)
\** Claude doit décider d'appeler `listen`, ajoute un round-trip

### Légende

- **A. SDK REPL** : puissant mais perd le TUI
- **B. tmux** : simple, garde le TUI, mais limité pour le TTS
- **C. MCP** : élégant, natif Claude Code, mais Claude contrôle le timing
- **D. Hooks** : TTS seul, se combine avec d'autres
- **E. query()** : trop limité, pas de session vraie
- **F. MCP+tmux** : le plus complet mais plus de pièces à assembler

---

## Détails techniques communs

### Détection de fin de parole (STT)

Le modèle STT émet des tokens texte quand il reconnaît de la parole. Pendant le silence, tokens 0 ou 3 (padding).

```python
silence_count = 0
SILENCE_THRESHOLD = 15  # 15 steps × 80ms ≈ 1.2 seconde

for each step:
    text_token = gen.step(audio_tokens, ct)[0].item()
    if text_token not in (0, 3):
        silence_count = 0
        accumulated_text += decode(text_token)
    else:
        if accumulated_text:
            silence_count += 1

    if silence_count >= SILENCE_THRESHOLD and accumulated_text:
        return accumulated_text
```

Déjà implémenté dans `src/jarvis/stt.py:listen_until_silence()`.

### Envoi TTS par phrases (streaming)

Pour réduire la latence, découper la réponse Claude en phrases et envoyer au TTS au fil de l'eau :

```python
def flush_sentences(buffer):
    """Envoie les phrases complètes au TTS, retourne le reste."""
    last_delim = max(buffer.rfind(c) for c in '.!?;\n')
    if last_delim == -1:
        return buffer
    sentence = buffer[:last_delim + 1].strip()
    if sentence:
        tts_speak(sentence)
    return buffer[last_delim + 1:]
```

### System prompt vocal

Pour les approches qui contrôlent le system prompt (A, E) :

```
Tu es un assistant vocal. Règles :
- Réponds de façon concise et naturelle, adaptée à la lecture à voix haute
- Pas de markdown (pas de **, ##, ```, etc.)
- Pas de listes à puces sauf si demandé
- Pas de blocs de code longs — décris ce que tu fais
- Quand tu utilises un outil, dis brièvement ce que tu fais
- Réponds dans la langue de l'utilisateur
```

Pour l'approche MCP (C, F), le prompt peut être ajouté via `.claude/CLAUDE.md` :

```markdown
## Voix
Tu as accès aux outils vocaux `listen` et `speak`.
- Utilise `speak` pour lire tes réponses importantes à voix haute
- Quand l'utilisateur dit "écoute" ou "voice", utilise `listen` pour capter sa voix
- Avec `speak`, sois concis et naturel — pas de markdown
```

---

## Risques et mitigations

| Risque | Mitigation |
|--------|-----------|
| Latence STT→Claude→TTS | Streaming par phrases (flush_sentences) |
| Claude répond trop long pour la voix | System prompt demande concision |
| Bruit micro déclenche faux positifs | `silence_threshold` + minimum de texte avant envoi |
| Outils dangereux exécutés sans contrôle | `allowed_tools` whitelist, `can_use_tool` (A) ou permissions Claude Code (B/C/F) |
| MLX conflict STT/TTS | Processus séparés (STT dans jah, TTS dans daemon serve) |
| Session perdue | `continue_conversation=True` (A/E), session native Claude Code (B/C/F) |
| tmux non disponible | Approche C (MCP) comme fallback |
| SDK instable (v0.0.25) | Approche B ou C comme fallback |

---

## Recherche approfondie : MCP Resources + Subscribe

### Question posée

Peut-on utiliser `resources/subscribe` et `notifications/resources/updated` du protocole MCP pour que le serveur STT **pousse** du texte transcrit vers Claude Code de façon asynchrone ?

### Résultat : NON SUPPORTÉ par Claude Code

Le protocole MCP définit bien :
- `resources/subscribe` — le client s'abonne aux changements d'une ressource
- `notifications/resources/updated` — le serveur notifie le client qu'une ressource a changé
- `notifications/resources/list_changed` — le serveur signale que la liste de ressources a changé

**Mais Claude Code n'implémente PAS `resources/subscribe` ni `notifications/resources/updated`.**

- GitHub issue [#7252](https://github.com/anthropics/claude-code/issues/7252) demandait cette feature
- **Fermée comme "Not Planned"** — pas prévu d'être implémenté
- Claude Code supporte uniquement :
  - `resources/list` — lister les ressources disponibles
  - `resources/read` — lire le contenu d'une ressource
  - `notifications/resources/list_changed` — rafraîchir le catalogue (pas le contenu)

### Conséquence

**Aucun mécanisme push natif** n'existe dans Claude Code pour qu'un service externe injecte des données dans la conversation en cours. Le modèle MCP pur (approche C) reste limité à des outils que Claude décide d'appeler.

### Workaround : outil MCP bloquant (long-poll)

Le pattern le plus viable est un outil MCP qui **bloque** en attendant du texte :

```python
@tool("wait_for_speech")
def wait_for_speech(timeout: float = 60):
    """Bloque jusqu'à ce que l'utilisateur parle."""
    bundle = get_stt_bundle()
    text = listen_until_silence(bundle, max_duration=timeout)
    return {"text": text}
```

Claude appellerait `wait_for_speech()` en boucle. Mais cela nécessite que Claude **décide** de rester en mode écoute — ajoutant un round-trip et une dépendance sur le comportement du modèle.

---

## Autres pistes explorées

### G. Claude Code Skills (custom slash commands)

Un skill `/voice` qui enchaîne listen → traitement → speak. Limité : les skills sont des prompts prédéfinis, pas de code custom. Ne résout pas le problème d'input continu.

### H. osascript / AppleScript injection

Injecter du texte dans le terminal via `osascript -e 'tell application "Terminal" to do script'`. Fragile, dépend de l'app Terminal, pas de iTerm2/tmux.

### I. PTY proxy (pseudo-terminal)

Un wrapper PTY entre l'utilisateur et Claude Code. Intercepte stdin/stdout, injecte du texte STT dans stdin, extrait le texte des réponses pour TTS. Puissant mais complexe (parser les escape codes ANSI).

### J. Clipboard monitoring

Écrire le texte STT dans le clipboard, simuler Cmd+V via osascript. Hacky, détruit le clipboard de l'utilisateur.

### K. iTerm2 Python API

iTerm2 expose une API Python pour contrôler les sessions. `session.async_send_text()` injecte du texte. Plus propre que tmux mais verrouille sur iTerm2.

### L. Hooks Claude Code pour input

Les hooks `pre-tool` et `post-tool` exécutent du code à chaque appel d'outil. Pourrait déclencher un `jah listen` mais ne résout pas l'injection de texte dans le prompt.

### M. File polling (CLAUDE.md dynamique)

Modifier `.claude/CLAUDE.md` dynamiquement avec les transcriptions STT. Claude le relit à chaque tour. Très indirect, non fiable pour du temps réel.

---

## Tableau comparatif (mise à jour)

| Critère | A. SDK REPL | B. tmux | C. MCP | D. Hooks | E. query() | F. MCP+tmux |
|---------|:-----------:|:-------:|:------:|:--------:|:----------:|:-----------:|
| **TUI Claude Code** | Non | Oui | Oui | Oui | Non | Oui |
| **STT (entrée voix)** | Oui | Oui | Oui* | Non | Oui | Oui |
| **TTS (sortie voix)** | Oui | Partiel | Oui | Oui | Oui | Oui |
| **Outils Claude** | Oui | Oui | Oui | Oui | Oui | Oui |
| **Session persistante** | Oui | Oui | Oui | Oui | Partiel | Oui |
| **Dual input clavier+voix** | Difficile | Oui | Non** | Clavier seul | Non | Oui |
| **Input voix à l'initiative user** | Oui | Oui | Non*** | Non | Oui | Oui |
| **Interruption vocale** | Oui | Non | Non | Non | Non | Non |
| **Complexité** | Moyenne | Faible | Moyenne | Faible | Faible | Moyenne |
| **Dépendance externe** | SDK | tmux | Aucune | Aucune | SDK | tmux |
| **Claude contrôle la voix** | Non | Non | Oui | Non | Non | Oui |
| **Latence** | Bonne | Bonne | Variable | Bonne | Mauvaise | Bonne |
| **Push async (STT→Claude)** | N/A | Oui | Non**** | Non | N/A | Oui |

\* STT déclenché par Claude (pas par l'utilisateur directement)
\** `resources/subscribe` non supporté par Claude Code — pas de push possible
\*** L'utilisateur doit demander à Claude d'appeler `listen`, pas d'écoute continue
\**** `notifications/resources/updated` non implémenté (issue #7252, "Not Planned")

### Légende mise à jour

- **A. SDK REPL** : puissant mais perd le TUI
- **B. tmux** : simple, garde le TUI, input voix naturel via injection
- **C. MCP seul** : élégant mais limité — pas de push, Claude contrôle le timing, pas de dual input réel
- **D. Hooks** : TTS sortie seul, se combine avec B
- **E. query()** : trop limité
- **F. MCP+tmux** : le plus complet — tmux pour l'input voix (push), MCP pour le speak (Claude contrôle)

---

## Décision

**À prendre.** Après exploration exhaustive, le classement final :

1. **B+D (tmux + hooks)** — le plus simple et le plus naturel
   - tmux injecte le texte STT comme du clavier (push, initiative user)
   - hooks lisent les réponses via TTS
   - Pas de dépendance SDK, pas de MCP, Claude Code vanilla
   - Complexité : faible

2. **F (MCP + tmux)** — le plus complet
   - tmux pour l'input voix (push, initiative user)
   - MCP `speak()` pour TTS (Claude contrôle quand parler)
   - MCP `listen()` optionnel (Claude peut aussi écouter)
   - Complexité : moyenne

3. **C (MCP seul)** — compromis
   - Pas de push (limitation confirmée : resources/subscribe non supporté)
   - Claude doit décider d'appeler `listen` — round-trip supplémentaire
   - Plus adapté si on veut que Claude contrôle tout le flow

**Recommandation : commencer par B+D** (le plus simple), puis ajouter MCP `speak` (→ F) si on veut que Claude contrôle le TTS.
