"""jah talk — Voice conversation with Claude Code."""

import asyncio
import os
import re
import sys
from pathlib import Path

import sounddevice as sd
import soundfile as sf

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


def generate_to_file(text: str, language: str, path: str) -> bool:
    """Ask daemon to generate audio to file (no playback). Returns success."""
    try:
        resp = send_request({
            "action": "generate",
            "text": text,
            "language": language,
            "output": path,
        })
        return resp.get("status") == "ok"
    except Exception as e:
        print(f"TTS gen error: {e}", file=sys.stderr)
        return False


def play_and_cleanup(path: str):
    """Play a WAV file via sounddevice, then delete it."""
    try:
        data, sr = sf.read(path)
        sd.play(data, sr)
        sd.wait()
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


async def gen_worker(
    sentence_queue: asyncio.Queue, audio_queue: asyncio.Queue, language: str
):
    """Read sentences, send to daemon for generation, put audio paths in queue."""
    i = 0
    while True:
        sentence = await sentence_queue.get()
        if sentence is None:
            await audio_queue.put(None)
            break
        path = f"/tmp/jarvis_tts_{os.getpid()}_{i:03d}.wav"
        i += 1
        ok = await asyncio.to_thread(generate_to_file, sentence, language, path)
        if ok:
            await audio_queue.put(path)
        else:
            print(f"  TTS failed: {sentence[:40]}", file=sys.stderr)


async def play_worker(audio_queue: asyncio.Queue):
    """Play audio files from queue sequentially."""
    while True:
        path = await audio_queue.get()
        if path is None:
            break
        await asyncio.to_thread(play_and_cleanup, path)


async def conversation_turn(
    text: str, session_id: str | None, language: str
) -> str | None:
    """Send text to Claude Code, pipeline sentences to TTS. Returns new session_id."""
    opts = ClaudeCodeOptions(
        append_system_prompt=VOICE_SYSTEM_PROMPT,
        cwd=str(Path.cwd()),
        resume=session_id,
        include_partial_messages=True,
    )

    sentence_queue = asyncio.Queue()
    audio_queue = asyncio.Queue()
    gen_task = asyncio.create_task(gen_worker(sentence_queue, audio_queue, language))
    play_task = asyncio.create_task(play_worker(audio_queue))

    buffer = ""
    new_session_id = session_id

    async for msg in query(prompt=text, options=opts):
        if isinstance(msg, StreamEvent):
            event = msg.event
            if event.get("type") == "content_block_delta":
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    chunk = delta["text"]
                    print(chunk, end="", flush=True)
                    buffer += chunk
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

    # Signal gen_worker to stop → it will signal play_worker
    await sentence_queue.put(None)
    await gen_task
    await play_task

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
