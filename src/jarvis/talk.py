"""jah talk — Voice conversation with Claude Code."""

import asyncio
import re
import sys
from pathlib import Path

from claude_code_sdk import (
    query,
    ClaudeCodeOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
)
from claude_code_sdk.types import StreamEvent

from jarvis.stt import load_model, listen_until_silence, reset
from jarvis.cli import send_request

VOICE_SYSTEM_PROMPT = (
    "Tu es un assistant vocal. Tes réponses seront lues par un synthétiseur vocal. "
    "RÈGLES STRICTES : "
    "- Maximum 2 à 3 phrases courtes. Jamais plus de 50 mots. "
    "- Pas de markdown, pas de formatage, pas de listes. "
    "- Pas de blocs de code. Décris ce que tu fais en une phrase. "
    "- Réponds dans la langue de l'utilisateur."
)


def speak(text: str, language: str = "French"):
    """Send text to TTS daemon (blocking)."""
    try:
        send_request({
            "action": "generate",
            "text": text,
            "language": language,
        })
    except Exception as e:
        print(f"TTS error: {e}", file=sys.stderr)


def extract_sentences(buffer: str) -> tuple[list[str], str]:
    """Extract complete sentences from buffer.

    Returns (sentences, remaining_buffer).
    Splits on sentence-ending punctuation followed by a space.
    """
    sentences = []
    while True:
        match = re.search(r"[.!?;:]\s", buffer)
        if not match:
            break
        end = match.end()
        sentence = buffer[:end].strip()
        if sentence:
            sentences.append(sentence)
        buffer = buffer[end:]
    return sentences, buffer


async def tts_worker(queue: asyncio.Queue, language: str):
    """Consume sentences from queue and speak them sequentially."""
    while True:
        sentence = await queue.get()
        if sentence is None:
            break
        await asyncio.to_thread(speak, sentence, language)


async def conversation_turn(
    text: str, session_id: str | None, language: str
) -> str | None:
    """Send text to Claude Code, stream sentences to TTS. Returns new session_id."""
    opts = ClaudeCodeOptions(
        append_system_prompt=VOICE_SYSTEM_PROMPT,
        cwd=str(Path.cwd()),
        resume=session_id,
        include_partial_messages=True,
    )

    sentence_queue = asyncio.Queue()
    worker = asyncio.create_task(tts_worker(sentence_queue, language))

    buffer = ""
    new_session_id = session_id

    async for msg in query(prompt=text, options=opts):
        if isinstance(msg, StreamEvent):
            event = msg.event
            # Extract text deltas from raw API stream events
            if event.get("type") == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    chunk = delta["text"]
                    print(chunk, end="", flush=True)
                    buffer += chunk
                    # Flush complete sentences to TTS
                    sentences, buffer = extract_sentences(buffer)
                    for s in sentences:
                        await sentence_queue.put(s)
        elif isinstance(msg, AssistantMessage):
            for block in msg.content:
                if isinstance(block, ToolUseBlock):
                    print(f"  [{block.name}]", file=sys.stderr, flush=True)
        elif isinstance(msg, ResultMessage):
            new_session_id = msg.session_id
            if msg.is_error:
                print(f"\nError: {msg.result}", file=sys.stderr)

    # Flush remaining buffer
    if buffer.strip():
        await sentence_queue.put(buffer.strip())

    # Signal worker to finish and wait
    await sentence_queue.put(None)
    await worker

    print(flush=True)
    return new_session_id


async def run_talk(language: str = "French"):
    """Main talk loop: listen -> Claude Code -> speak -> repeat."""
    print("Loading STT model...", file=sys.stderr, flush=True)
    bundle = load_model()
    print("Ready. Speak now.\n", file=sys.stderr, flush=True)

    session_id = None

    while True:
        try:
            text = listen_until_silence(bundle)
            if not text:
                continue

            print(f"\n> {text}", flush=True)
            session_id = await conversation_turn(text, session_id, language)
            reset(bundle)
        except KeyboardInterrupt:
            break

    print("\nDone.", file=sys.stderr)
