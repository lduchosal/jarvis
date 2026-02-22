"""jah panel — Multi-model debate: up to 6 AI participants."""

import asyncio
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

from claude_code_sdk import (
    query,
    ClaudeCodeOptions,
    ResultMessage,
)
from claude_code_sdk.types import StreamEvent

# Monkey-patch: claude-code-sdk crashes on unknown message types like
# "rate_limit_event" (informational, not an error). Patch parse_message
# to return None for unknown types instead of raising MessageParseError.
import claude_code_sdk._internal.message_parser as _mp
import claude_code_sdk._internal.client as _cl
_original_parse = _mp.parse_message

def _patched_parse(data):
    try:
        return _original_parse(data)
    except _mp.MessageParseError:
        return None

_cl.parse_message = _patched_parse

from openai_codex_sdk import (
    Codex,
    AgentMessageItem,
    ItemUpdatedEvent,
    ItemCompletedEvent,
)

PANELS_DIR = Path.home() / ".cache" / "jarvis" / "panels"
KEYS_FILE = Path.home() / ".config" / "jarvis" / "keys"


def _load_keys():
    """Load API keys from ~/.config/jarvis/keys into os.environ."""
    if not KEYS_FILE.exists():
        return
    for line in KEYS_FILE.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())
    # google-genai expects GOOGLE_API_KEY
    if not os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]


# ---------------------------------------------------------------------------
# Participants table
# ---------------------------------------------------------------------------

PARTICIPANTS = {
    "Opus":       {"type": "claude", "model": "claude-opus-4-6",            "label": "\033[1;35m[Opus]\033[0m"},
    "Sonnet":     {"type": "claude", "model": "claude-sonnet-4-5-20250929", "label": "\033[1;34m[Sonnet]\033[0m"},
    "Haiku":      {"type": "claude", "model": "claude-haiku-4-5-20251001",  "label": "\033[1;36m[Haiku]\033[0m"},
    "Codex":      {"type": "codex",                                         "label": "\033[1;32m[Codex]\033[0m"},
    "Gemini 2.5": {"type": "gemini", "model": "gemini-2.5-flash",          "label": "\033[1;33m[Gemini 2.5]\033[0m"},
    "Gemini 3.0": {"type": "gemini", "model": "gemini-3-flash-preview",    "label": "\033[1;31m[Gemini 3.0]\033[0m"},
}

DEFAULT_PARTICIPANTS = ["Opus", "Codex"]

# Aliases for CLI -p flag (lowercase -> canonical name)
ALIASES = {
    "opus": "Opus",
    "sonnet": "Sonnet",
    "haiku": "Haiku",
    "codex": "Codex",
    "gemini-2.5": "Gemini 2.5",
    "gemini-3.0": "Gemini 3.0",
    "gemini2.5": "Gemini 2.5",
    "gemini3.0": "Gemini 3.0",
    "gemini": "Gemini 2.5",  # bare "gemini" → default to 2.5
    "claude": "Opus",        # bare "claude" → default to Opus
}

PANEL_ROLES = {
    "Opus": (
        "Ton rôle : Avocat du Diable. Défends toujours la position la moins "
        "représentée dans la discussion. Si un consensus émerge, attaque-le. "
        "Si tout le monde est contre une idée, défends-la."
    ),
    "Sonnet": (
        "Ton rôle : Vérificateur Technique. Exige des preuves concrètes "
        "(fichier:ligne, doc officielle, output de commande) pour toute "
        "affirmation technique. Refuse les affirmations vagues. Marque "
        "explicitement les hypothèses non vérifiées comme telles."
    ),
    "Haiku": (
        "Ton rôle : Accélérateur d'Exécution. Ramène chaque discussion à "
        "l'action concrète et immédiate. Identifie la décision à prendre "
        "maintenant. Signale quand le débat devient trop théorique."
    ),
    "Codex": (
        "Ton rôle : Contradicteur Méthodologique. Cherche les failles "
        "logiques et les contre-exemples factuels. Pour chaque intervention, "
        "énonce : ta thèse, ton objection principale, et ce qui te ferait "
        "changer d'avis."
    ),
    "Gemini 2.5": (
        "Ton rôle : Synthétiseur. Identifie ce qui manque dans le débat, "
        "les angles morts que personne n'a couverts. Relie les positions "
        "entre elles et signale les non-dits."
    ),
    "Gemini 3.0": (
        "Ton rôle : Pragmatique. Évalue la faisabilité concrète, les coûts "
        "et les compromis réels. Transforme les oppositions en solutions "
        "constructives adaptées à l'intention de l'utilisateur."
    ),
}

PANEL_BASE = (
    "Tu participes à un panel de discussion avec d'autres modèles IA. "
    "L'utilisateur pose des questions, vous répondez chacun à tour de rôle. "
    "Tu peux commenter, compléter ou contredire les autres. "
    "Sois concis (3-5 phrases). Pas de markdown, pas de listes, pas de blocs de code. "
    "Réponds dans la langue de l'utilisateur. "
    "RÈGLE ANTI-CONVERGENCE : si les réponses précédentes convergent, "
    "tu DOIS explorer une position opposée ou un angle ignoré."
)


def panel_system_for(name: str, roles: bool = False) -> str:
    if roles:
        role = PANEL_ROLES.get(name, "")
        return f"{PANEL_BASE}\n\n{role}"
    return PANEL_BASE

LABEL_YOU = "\033[1;36mYou>\033[0m"

def _detect_speakers(question: str, active: dict) -> dict:
    """Detect which participants are directly addressed in the question.

    Only triggers when names appear in direct address position:
    before the first comma/colon at the start of the message.
    e.g. "opus, rédige un draft" → Opus only
         "Tout le monde dit merci à Haiku" → all (mention, not address)
    """
    # Extract the address prefix (text before first , or :)
    for sep in (",", ":"):
        idx = question.find(sep)
        if idx != -1:
            prefix = question[:idx].lower().strip()
            if len(prefix) > 60:
                # Too long to be a direct address
                continue

            # Check which active participants are named in the prefix
            addressed = {}
            for name, p in active.items():
                if name.lower() in prefix:
                    addressed[name] = p
                    continue
                for alias, canonical in ALIASES.items():
                    if canonical == name and alias in prefix:
                        addressed[name] = p
                        break

            if addressed:
                return addressed

    return active


def _is_rate_limited(e: Exception) -> bool:
    err = str(e).lower()
    return any(k in err for k in ("rate_limit", "rate limit", "resource_exhausted", "quota", "429"))


def _clean(text: str) -> str:
    """Strip surrogate characters that break UTF-8 encoding."""
    return text.encode("utf-8", errors="ignore").decode("utf-8")


def _resolve_participants(participants_filter: str | None) -> dict:
    """Return the active participants dict based on the CLI filter."""
    if not participants_filter:
        has_gemini_key = bool(os.environ.get("GOOGLE_API_KEY"))
        active = {}
        for name in DEFAULT_PARTICIPANTS:
            p = PARTICIPANTS[name]
            if p["type"] == "gemini" and not has_gemini_key:
                continue
            active[name] = p
        return active

    active = {}
    for token in participants_filter.split(","):
        token = token.strip()
        canonical = ALIASES.get(token.lower())
        if canonical and canonical in PARTICIPANTS:
            active[canonical] = PARTICIPANTS[canonical]
        else:
            print(f"Unknown participant: {token!r}", file=sys.stderr)
            print(f"Available: {', '.join(ALIASES.keys())}", file=sys.stderr)
            sys.exit(1)
    return active


# ---------------------------------------------------------------------------
# Conversation log — saves every turn to disk, survives crashes
# ---------------------------------------------------------------------------

class ConversationLog:
    """Append-only JSONL log of the panel conversation."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sessions: dict = {}  # name -> session_id
        self.turns: list[dict] = []

    def record_turn(self, question: str, responses: dict[str, str]):
        clean_responses = {k: _clean(v) for k, v in responses.items()}
        turn = {
            "ts": datetime.now().isoformat(),
            "question": _clean(question),
            "responses": clean_responses,
            "sessions": dict(self.sessions),
        }
        self.turns.append(turn)
        with open(self.path, "a") as f:
            f.write(json.dumps(turn, ensure_ascii=False) + "\n")

    def set_session(self, name: str, session_id: str | None):
        if session_id:
            self.sessions[name] = session_id

    @staticmethod
    def load(path: Path) -> "ConversationLog":
        log = ConversationLog(path)
        if path.exists():
            for line in path.read_text().splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    turn = json.loads(line)
                    log.turns.append(turn)
                    if "sessions" in turn:
                        log.sessions.update(turn["sessions"])
                except json.JSONDecodeError:
                    continue
        return log

    def build_resume_context(self) -> str:
        """Build a summary of previous turns for models that lost their session."""
        if not self.turns:
            return ""
        parts = ["Voici le résumé de la conversation précédente :\n"]
        for i, turn in enumerate(self.turns, 1):
            parts.append(f"Tour {i} — Question : {turn['question']}")
            for model, resp in turn.get("responses", {}).items():
                parts.append(f"  {model} : {resp}")
        return "\n".join(parts)

    @staticmethod
    def list_sessions() -> list[Path]:
        if not PANELS_DIR.exists():
            return []
        return sorted(PANELS_DIR.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)


# ---------------------------------------------------------------------------
# Claude streaming (via claude-code-sdk) — parameterized model
# ---------------------------------------------------------------------------

async def stream_claude(
    prompt: str,
    session_id: str | None,
    model: str,
    label: str,
    name: str = "",
    roles: bool = False,
) -> tuple[str, str | None]:
    opts = ClaudeCodeOptions(
        model=model,
        append_system_prompt=panel_system_for(name, roles),
        cwd=str(Path.cwd()),
        include_partial_messages=True,
        **({"resume": session_id} if session_id else {}),
    )

    print(f"\n{label}")
    full_text = ""
    new_session_id = session_id

    try:
        async for msg in query(prompt=_clean(prompt), options=opts):
            if msg is None:
                continue
            if isinstance(msg, StreamEvent):
                event = msg.event
                if event.get("type") == "content_block_delta":
                    delta = event.get("delta", {})
                    if delta.get("type") == "text_delta":
                        chunk = _clean(delta["text"])
                        print(chunk, end="", flush=True)
                        full_text += chunk
            elif isinstance(msg, ResultMessage):
                new_session_id = msg.session_id
    except Exception as e:
        if not _is_rate_limited(e):
            raise
        print(f"\n  [rate limited: {e}]", file=sys.stderr, flush=True)

    print(flush=True)
    return _clean(full_text), new_session_id


async def stream_claude_silent(
    prompt: str,
    session_id: str | None,
    model: str,
    name: str = "",
    roles: bool = False,
) -> tuple[str, str | None]:
    """Send prompt to Claude without printing — just to keep session in sync."""
    opts = ClaudeCodeOptions(
        model=model,
        append_system_prompt=panel_system_for(name, roles),
        cwd=str(Path.cwd()),
        **({"resume": session_id} if session_id else {}),
    )
    new_session_id = session_id
    try:
        async for msg in query(prompt=_clean(prompt), options=opts):
            if msg is None:
                continue
            if isinstance(msg, ResultMessage):
                new_session_id = msg.session_id
    except Exception:
        pass
    return "", new_session_id


# ---------------------------------------------------------------------------
# Codex streaming (via openai-codex-sdk) — parameterized label
# ---------------------------------------------------------------------------

async def stream_codex(
    prompt: str,
    thread,
    label: str,
) -> str:
    print(f"\n{label}")

    full_text = ""
    prev_len = 0
    streamed = await thread.run_streamed(_clean(prompt))

    async for event in streamed.events:
        if isinstance(event, ItemUpdatedEvent) and isinstance(event.item, AgentMessageItem):
            new_text = _clean(event.item.text)
            if len(new_text) > prev_len:
                print(new_text[prev_len:], end="", flush=True)
                prev_len = len(new_text)
                full_text = new_text
        elif isinstance(event, ItemCompletedEvent) and isinstance(event.item, AgentMessageItem):
            new_text = _clean(event.item.text)
            if len(new_text) > prev_len:
                print(new_text[prev_len:], end="", flush=True)
            full_text = new_text

    print(flush=True)
    return _clean(full_text)


# ---------------------------------------------------------------------------
# Gemini streaming (via google-genai)
# ---------------------------------------------------------------------------

async def stream_gemini(
    prompt: str,
    chat,
    label: str,
) -> str:
    print(f"\n{label}")
    full_text = ""
    try:
        async for chunk in await chat.send_message_stream(_clean(prompt)):
            if chunk.text:
                text = _clean(chunk.text)
                print(text, end="", flush=True)
                full_text += text
    except Exception as e:
        if not _is_rate_limited(e):
            raise
        print("  [rate limited]", file=sys.stderr, flush=True)

    print(flush=True)
    return _clean(full_text)


# ---------------------------------------------------------------------------
# TTS helper
# ---------------------------------------------------------------------------

def _tts_speak(text: str, language: str):
    try:
        from jarvis.cli import send_request
        send_request({
            "action": "generate",
            "text": text,
            "language": language,
        })
    except Exception as e:
        print(f"  TTS error: {e}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------

def _build_context(previous_responses: dict[str, str], current_question: str) -> str:
    parts = []
    for name, text in previous_responses.items():
        if text:
            parts.append(f"[{name} a dit : {text}]")
    if parts:
        return "\n".join(parts) + f"\n\nQuestion de l'utilisateur : {current_question}"
    return current_question


def _build_recap(question: str, turn_responses: dict[str, str]) -> str:
    """Build a recap message for models that didn't speak this turn."""
    parts = [f"[L'utilisateur a demandé : {question}]"]
    for name, text in turn_responses.items():
        if text:
            parts.append(f"[{name} a répondu : {text}]")
    return "\n".join(parts) + "\n\n(Tu n'avais pas la parole ce tour-ci. Retiens ce contexte.)"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_panel(
    language: str = "French",
    tts: bool = False,
    resume: str | None = None,
    participants_filter: str | None = None,
    roles: bool = False,
):
    """Main panel loop: user types questions, AI models debate."""
    _load_keys()
    active = _resolve_participants(participants_filter)
    names = ", ".join(active.keys())
    roles_tag = " \033[2m(roles)\033[0m" if roles else ""
    print(f"\n\033[1mPanel: {names}\033[0m{roles_tag}")

    # Load or create conversation log
    if resume == "latest":
        sessions = ConversationLog.list_sessions()
        if sessions:
            log = ConversationLog.load(sessions[0])
            print(f"Resumed: {sessions[0].name} ({len(log.turns)} turns)")
        else:
            print("No previous sessions found.", file=sys.stderr)
            return
    elif resume:
        path = PANELS_DIR / f"{resume}.jsonl"
        if path.exists():
            log = ConversationLog.load(path)
            print(f"Resumed: {path.name} ({len(log.turns)} turns)")
        else:
            print(f"Session not found: {path}", file=sys.stderr)
            return
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log = ConversationLog(PANELS_DIR / f"panel_{ts}.jsonl")
        print(f"Session: {log.path.name}")

    print("Type your questions. Ctrl+C to exit.\n")

    # --- Init sessions per type ---
    sessions: dict = {}  # name -> session state (session_id, chat, thread)

    # Codex thread
    codex_thread = None
    if "Codex" in active:
        codex = Codex()
        codex_thread = codex.start_thread({
            "working_directory": str(Path.cwd()),
            "skip_git_repo_check": True,
        })
        sessions["Codex"] = codex_thread

    # Gemini async chats
    gemini_names = [n for n in active if active[n]["type"] == "gemini"]
    if gemini_names:
        from google import genai
        from google.genai import types
        gclient = genai.Client()
        for name in gemini_names:
            sessions[name] = gclient.aio.chats.create(
                model=active[name]["model"],
                config=types.GenerateContentConfig(
                    system_instruction=panel_system_for(name, roles),
                ),
            )

    # Claude session IDs (from log if resuming)
    for name in active:
        if active[name]["type"] == "claude":
            sessions[name] = log.sessions.get(name)

    # Resume context for stateless models (Codex + Gemini)
    last_responses: dict[str, str] = {}
    if log.turns:
        last_turn = log.turns[-1]
        last_responses = last_turn.get("responses", {})
        resume_ctx = log.build_resume_context()
        if resume_ctx:
            resume_msg = (
                "Voici le contexte d'une conversation précédente. "
                f"Lis-le et confirme que tu es prêt à continuer.\n\n{_clean(resume_ctx)}"
            )
            # Codex
            if codex_thread:
                print("Reloading context into Codex...", file=sys.stderr, flush=True)
                await codex_thread.run(resume_msg)
            # Gemini chats
            for name in gemini_names:
                print(f"Reloading context into {name}...", file=sys.stderr, flush=True)
                async for _ in await sessions[name].send_message_stream(resume_msg):
                    pass
            if codex_thread or gemini_names:
                print("Ready.\n", file=sys.stderr, flush=True)

    while True:
        try:
            question = input(f"{LABEL_YOU} ")
            if not question.strip():
                continue
        except (KeyboardInterrupt, EOFError):
            break

        turn_responses: dict[str, str] = {}

        # Detect who is addressed; randomize order
        speakers = _detect_speakers(question, active)
        order = list(speakers.keys())
        random.shuffle(order)
        if len(speakers) < len(active):
            names = ", ".join(speakers.keys())
            print(f"  \033[2m→ {names}\033[0m", file=sys.stderr, flush=True)

        for i, name in enumerate(order):
            p = speakers[name]

            # First speaker sees other's previous response;
            # subsequent speakers see current turn responses
            if i == 0:
                context = {k: v for k, v in last_responses.items() if k != name}
            else:
                context = dict(turn_responses)

            try:
                if p["type"] == "claude":
                    prompt = _build_context(context, question)
                    response, new_sid = await stream_claude(
                        prompt, sessions[name], p["model"], p["label"], name, roles,
                    )
                    sessions[name] = new_sid
                    log.set_session(name, new_sid)

                elif p["type"] == "codex":
                    prompt = f"{panel_system_for(name, roles)}\n\n" + _build_context(context, question)
                    response = await stream_codex(prompt, codex_thread, p["label"])

                elif p["type"] == "gemini":
                    prompt = _build_context(context, question)
                    response = await stream_gemini(prompt, sessions[name], p["label"])

            except Exception as e:
                print(f"\n  {name} error: {e}", file=sys.stderr)
                response = ""

            turn_responses[name] = response

            if tts and response:
                await asyncio.to_thread(_tts_speak, response, language)

        # Broadcast recap to models that didn't speak
        silent = {n: p for n, p in active.items() if n not in speakers}
        if silent and turn_responses:
            recap = _build_recap(question, turn_responses)
            for name, p in silent.items():
                try:
                    if p["type"] == "claude":
                        # Inject recap into Claude session
                        _, new_sid = await stream_claude_silent(
                            recap, sessions[name], p["model"], name, roles,
                        )
                        sessions[name] = new_sid
                        log.set_session(name, new_sid)
                    elif p["type"] == "gemini":
                        # Inject recap into Gemini chat
                        async for _ in await sessions[name].send_message_stream(
                            _clean(recap)
                        ):
                            pass
                    elif p["type"] == "codex":
                        # Inject recap into Codex thread
                        await codex_thread.run(_clean(recap))
                except Exception as e:
                    print(f"  {name} sync error: {e}", file=sys.stderr)

        # Save turn to disk (crash-safe)
        log.record_turn(question, turn_responses)
        last_responses = dict(turn_responses)

        print()

    print(f"\nSaved: {log.path}", file=sys.stderr)
