"""jah panel — Multi-model debate: Claude 4.6 + Codex 5.3."""

import asyncio
import sys
from pathlib import Path

from claude_code_sdk import (
    query,
    ClaudeCodeOptions,
    AssistantMessage,
    ResultMessage,
)
from claude_code_sdk.types import StreamEvent

from openai_codex_sdk import (
    Codex,
    AgentMessageItem,
    ItemUpdatedEvent,
    ItemCompletedEvent,
)

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


async def stream_claude(
    prompt: str,
    session_id: str | None,
) -> tuple[str, str | None]:
    """Send prompt to Claude, stream response. Returns (response_text, new_session_id)."""
    opts = ClaudeCodeOptions(
        append_system_prompt=PANEL_SYSTEM_CLAUDE,
        cwd=str(Path.cwd()),
        include_partial_messages=True,
        **({"resume": session_id} if session_id else {}),
    )

    print(f"\n{LABEL_CLAUDE}")
    full_text = ""
    new_session_id = session_id

    async for msg in query(prompt=prompt, options=opts):
        if isinstance(msg, StreamEvent):
            event = msg.event
            if event.get("type") == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    chunk = delta["text"]
                    print(chunk, end="", flush=True)
                    full_text += chunk
        elif isinstance(msg, ResultMessage):
            new_session_id = msg.session_id

    print(flush=True)
    return full_text, new_session_id


async def stream_codex(
    prompt: str,
    thread,
) -> str:
    """Send prompt to Codex, stream response. Returns response text."""
    print(f"\n{LABEL_CODEX}")

    full_text = ""
    prev_len = 0
    streamed = await thread.run_streamed(prompt)

    async for event in streamed.events:
        if isinstance(event, ItemUpdatedEvent) and isinstance(event.item, AgentMessageItem):
            new_text = event.item.text
            if len(new_text) > prev_len:
                print(new_text[prev_len:], end="", flush=True)
                prev_len = len(new_text)
                full_text = new_text
        elif isinstance(event, ItemCompletedEvent) and isinstance(event.item, AgentMessageItem):
            new_text = event.item.text
            if len(new_text) > prev_len:
                print(new_text[prev_len:], end="", flush=True)
            full_text = new_text

    print(flush=True)
    return full_text


def _tts_speak(text: str, language: str):
    """Send text to TTS daemon for playback (blocking)."""
    try:
        from jarvis.cli import send_request
        send_request({
            "action": "generate",
            "text": text,
            "language": language,
        })
    except Exception as e:
        print(f"  TTS error: {e}", file=sys.stderr)


async def run_panel(language: str = "French", tts: bool = False):
    """Main panel loop: user types questions, Claude and Codex debate."""
    print(f"\n\033[1mPanel: Claude 4.6 + Codex 5.3\033[0m")
    print("Type your questions. Ctrl+C to exit.\n")

    # Init Codex thread
    codex = Codex()
    codex_thread = codex.start_thread({
        "working_directory": str(Path.cwd()),
    })

    claude_session_id = None
    codex_last_response = ""

    while True:
        try:
            question = input(f"{LABEL_YOU} ")
            if not question.strip():
                continue
        except (KeyboardInterrupt, EOFError):
            break

        # Build Claude prompt with Codex's previous response as context
        claude_prompt = question
        if codex_last_response:
            claude_prompt = (
                f"[Codex a dit au tour précédent : {codex_last_response}]\n\n"
                f"{question}"
            )

        # Stream Claude response
        claude_response, claude_session_id = await stream_claude(
            claude_prompt, claude_session_id,
        )

        if tts and claude_response:
            await asyncio.to_thread(_tts_speak, claude_response, language)

        # Build Codex prompt with Claude's response as context
        codex_prompt = (
            f"{PANEL_SYSTEM_CODEX}\n\n"
            f"[Claude vient de dire : {claude_response}]\n\n"
            f"Question de l'utilisateur : {question}"
        )

        # Stream Codex response
        codex_last_response = await stream_codex(codex_prompt, codex_thread)

        if tts and codex_last_response:
            await asyncio.to_thread(_tts_speak, codex_last_response, language)

        print()

    print("\nDone.", file=sys.stderr)
