"""jah panel — Multi-model debate: Claude 4.6 + Codex 5.3."""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

from claude_code_sdk import (
    query,
    ClaudeCodeOptions,
    ResultMessage,
)
from claude_code_sdk.types import StreamEvent

from openai_codex_sdk import (
    Codex,
    AgentMessageItem,
    ItemUpdatedEvent,
    ItemCompletedEvent,
)

PANELS_DIR = Path.home() / ".cache" / "jarvis" / "panels"

PANEL_SYSTEM_CLAUDE = (
    "Tu participes à un panel de discussion avec Codex 5.3 (OpenAI). "
    "L'utilisateur pose des questions, vous répondez chacun à tour de rôle. "
    "Tu peux commenter, compléter ou contredire Codex. "
    "Sois concis (3-5 phrases). Pas de markdown, pas de listes, pas de blocs de code. "
    "Réponds dans la langue de l'utilisateur."
)

PANEL_SYSTEM_CODEX = (
    "Tu participes à un panel de discussion avec Claude 4.6 (Anthropic). "
    "L'utilisateur pose des questions, vous répondez chacun à tour de rôle. "
    "Tu peux commenter, compléter ou contredire Claude. "
    "Sois concis (3-5 phrases). Pas de markdown, pas de listes, pas de blocs de code. "
    "Réponds dans la langue de l'utilisateur."
)

LABEL_CLAUDE = "\033[1;35m[Claude 4.6]\033[0m"
LABEL_CODEX = "\033[1;32m[Codex 5.3]\033[0m"
LABEL_YOU = "\033[1;36mYou>\033[0m"


def _clean(text: str) -> str:
    """Strip surrogate characters that break UTF-8 encoding."""
    return text.encode("utf-8", errors="ignore").decode("utf-8")


# ---------------------------------------------------------------------------
# Conversation log — saves every turn to disk, survives crashes
# ---------------------------------------------------------------------------

class ConversationLog:
    """Append-only JSONL log of the panel conversation."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.sessions: dict = {}  # model -> session_id
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

    def set_session(self, model: str, session_id: str | None):
        if session_id:
            self.sessions[model] = session_id

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
# Claude streaming (via claude-code-sdk)
# ---------------------------------------------------------------------------

async def stream_claude(
    prompt: str,
    session_id: str | None,
) -> tuple[str, str | None]:
    opts = ClaudeCodeOptions(
        append_system_prompt=PANEL_SYSTEM_CLAUDE,
        cwd=str(Path.cwd()),
        include_partial_messages=True,
        **({"resume": session_id} if session_id else {}),
    )

    print(f"\n{LABEL_CLAUDE}")
    full_text = ""
    new_session_id = session_id

    async for msg in query(prompt=_clean(prompt), options=opts):
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

    print(flush=True)
    return _clean(full_text), new_session_id


# ---------------------------------------------------------------------------
# Codex streaming (via openai-codex-sdk)
# ---------------------------------------------------------------------------

async def stream_codex(
    prompt: str,
    thread,
) -> str:
    print(f"\n{LABEL_CODEX}")

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


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

async def run_panel(
    language: str = "French",
    tts: bool = False,
    resume: str | None = None,
):
    """Main panel loop: user types questions, Claude and Codex debate."""
    print(f"\n\033[1mPanel: Claude 4.6 + Codex 5.3\033[0m")

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

    # Init Codex thread
    codex = Codex()
    codex_thread = codex.start_thread({
        "working_directory": str(Path.cwd()),
    })

    # Restore sessions from log
    claude_session_id = log.sessions.get("claude")

    # If resuming, rebuild context for Codex (no session resume)
    last_responses: dict[str, str] = {}
    if log.turns:
        last_turn = log.turns[-1]
        last_responses = last_turn.get("responses", {})
        resume_ctx = log.build_resume_context()
        if resume_ctx:
            print("Reloading context into Codex...", file=sys.stderr, flush=True)
            await codex_thread.run(
                f"Voici le contexte d'une conversation précédente. "
                f"Lis-le et confirme que tu es prêt à continuer.\n\n{_clean(resume_ctx)}"
            )
            print("Ready.\n", file=sys.stderr, flush=True)

    while True:
        try:
            question = input(f"{LABEL_YOU} ")
            if not question.strip():
                continue
        except (KeyboardInterrupt, EOFError):
            break

        turn_responses: dict[str, str] = {}

        # --- Claude (goes first, sees Codex's previous response) ---
        try:
            claude_prompt = _build_context(last_responses, question)
            claude_response, claude_session_id = await stream_claude(
                claude_prompt, claude_session_id,
            )
            log.set_session("claude", claude_session_id)
            turn_responses["Claude"] = claude_response
        except Exception as e:
            print(f"\n  Claude error: {e}", file=sys.stderr)
            turn_responses["Claude"] = ""

        if tts and turn_responses.get("Claude"):
            await asyncio.to_thread(_tts_speak, turn_responses["Claude"], language)

        # --- Codex (sees Claude's current response) ---
        try:
            codex_context = {"Claude": turn_responses.get("Claude", "")}
            codex_prompt = (
                f"{PANEL_SYSTEM_CODEX}\n\n"
                + _build_context(codex_context, question)
            )
            codex_response = await stream_codex(codex_prompt, codex_thread)
            turn_responses["Codex"] = codex_response
        except Exception as e:
            print(f"\n  Codex error: {e}", file=sys.stderr)
            turn_responses["Codex"] = ""

        if tts and turn_responses.get("Codex"):
            await asyncio.to_thread(_tts_speak, turn_responses["Codex"], language)

        # Save turn to disk (crash-safe)
        log.record_turn(question, turn_responses)
        last_responses = {"Codex": turn_responses.get("Codex", "")}

        print()

    print(f"\nSaved: {log.path}", file=sys.stderr)
