"""jah talk — Voice conversation with Claude Code."""

import asyncio
import atexit
import os
import re
import sys
import termios
import threading
import tty
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


# ---------------------------------------------------------------------------
# KeyMonitor — non-blocking keypress detection via cbreak stdin
# ---------------------------------------------------------------------------

class KeyMonitor:
    """Watch for a trigger keypress in a background thread."""

    def __init__(self, trigger_key=" "):
        self.trigger_key = trigger_key
        self.barge_in = threading.Event()
        self._thread = None
        self._stop = False
        self._old_settings = None

    def start(self):
        self.barge_in.clear()
        self._stop = False
        try:
            self._old_settings = termios.tcgetattr(sys.stdin)
        except termios.error:
            self._old_settings = None
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        # Safety net: restore terminal on process exit
        atexit.register(self._restore_terminal)

    def _watch(self):
        try:
            tty.setcbreak(sys.stdin.fileno())
            while not self._stop:
                ch = sys.stdin.read(1)
                if ch == self.trigger_key:
                    self.barge_in.set()
                    break
                if ch == "":
                    break  # EOF
        except (OSError, ValueError, termios.error):
            pass  # stdin closed or not a TTY

    def _restore_terminal(self):
        if self._old_settings:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            except termios.error:
                pass
            self._old_settings = None

    def stop(self):
        self._stop = True
        self._restore_terminal()
        atexit.unregister(self._restore_terminal)


# ---------------------------------------------------------------------------
# Sentence extraction
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# TTS generation & playback
# ---------------------------------------------------------------------------

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


def play_interruptible(path: str, barge_in: threading.Event, delete: bool = True) -> bool:
    """Play a WAV file, stoppable via barge_in event.

    Returns True if interrupted, False if completed normally.
    """
    try:
        data, sr = sf.read(path, dtype="float32")
    except Exception:
        return False
    finally:
        if delete:
            try:
                os.unlink(path)
            except OSError:
                pass

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
    stream.start()

    chunk_size = int(sr * 0.04)  # 40ms chunks → ~25 checks/sec
    offset = 0

    try:
        while offset < len(data):
            if barge_in.is_set():
                stream.abort()
                return True
            end = min(offset + chunk_size, len(data))
            stream.write(data[offset:end])
            offset = end
    finally:
        if stream.active:
            stream.stop()
        stream.close()

    return False


# ---------------------------------------------------------------------------
# Pipeline workers
# ---------------------------------------------------------------------------

async def gen_worker(
    sentence_queue: asyncio.Queue,
    audio_queue: asyncio.Queue,
    language: str,
    barge_in: threading.Event,
):
    """Read sentences, send to daemon for generation, put audio paths in queue."""
    i = 0
    while True:
        sentence = await sentence_queue.get()
        if sentence is None or barge_in.is_set():
            await audio_queue.put(None)
            break
        path = f"/tmp/jarvis_tts_{os.getpid()}_{i:03d}.wav"
        i += 1
        ok = await asyncio.to_thread(generate_to_file, sentence, language, path)
        if barge_in.is_set():
            try:
                os.unlink(path)
            except OSError:
                pass
            await audio_queue.put(None)
            break
        if ok:
            await audio_queue.put(path)
        else:
            print(f"  TTS failed: {sentence[:40]}", file=sys.stderr)


async def play_worker(
    audio_queue: asyncio.Queue,
    barge_in: threading.Event,
    filler_done: asyncio.Event | None = None,
):
    """Play audio files from queue sequentially, stoppable via barge_in."""
    # Wait for filler to finish before playing real audio
    if filler_done is not None:
        await filler_done.wait()

    while True:
        path = await audio_queue.get()
        if path is None:
            break
        interrupted = await asyncio.to_thread(play_interruptible, path, barge_in)
        if interrupted:
            # Drain and cleanup remaining queued files
            while not audio_queue.empty():
                remaining = audio_queue.get_nowait()
                if remaining is None:
                    break
                try:
                    os.unlink(remaining)
                except OSError:
                    pass
            break


# ---------------------------------------------------------------------------
# Conversation turn
# ---------------------------------------------------------------------------

async def conversation_turn(
    text: str, session_id: str | None, language: str, bundle: dict
) -> str | None:
    """Send text to Claude Code, pipeline sentences to TTS. Returns new session_id."""
    opts = ClaudeCodeOptions(
        append_system_prompt=VOICE_SYSTEM_PROMPT,
        cwd=str(Path.cwd()),
        resume=session_id,
        include_partial_messages=True,
    )

    sentence_queue: asyncio.Queue = asyncio.Queue()
    audio_queue: asyncio.Queue = asyncio.Queue()

    keys = KeyMonitor()
    keys.start()

    # Play a filler immediately while Claude thinks
    filler_done = asyncio.Event()

    async def _play_filler():
        try:
            resp = await asyncio.to_thread(
                send_request, {"action": "get_filler", "language": language}
            )
            filler_path = resp.get("path")
            if filler_path and not keys.barge_in.is_set():
                await asyncio.to_thread(
                    play_interruptible, filler_path, keys.barge_in, False
                )
        except Exception:
            pass
        finally:
            filler_done.set()

    filler_task = asyncio.create_task(_play_filler())

    gen_task = asyncio.create_task(
        gen_worker(sentence_queue, audio_queue, language, keys.barge_in)
    )
    play_task = asyncio.create_task(
        play_worker(audio_queue, keys.barge_in, filler_done)
    )

    buffer = ""
    new_session_id = session_id

    try:
        async for msg in query(prompt=text, options=opts):
            if keys.barge_in.is_set():
                break

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
    finally:
        if keys.barge_in.is_set():
            # Drain pending sentences
            while not sentence_queue.empty():
                sentence_queue.get_nowait()
        else:
            # Normal: flush remaining buffer
            if buffer.strip():
                await sentence_queue.put(buffer.strip())

        await sentence_queue.put(None)
        await filler_task
        await gen_task
        await play_task
        keys.stop()

    print(flush=True)

    if keys.barge_in.is_set():
        print("[interrupted — listening...]", file=sys.stderr, flush=True)
        reset(bundle)
        new_text = listen_until_silence(bundle)
        if new_text:
            print(f"\n> {new_text}", flush=True)
            return await conversation_turn(
                new_text, new_session_id, language, bundle
            )

    return new_session_id


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

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
            session_id = await conversation_turn(
                text, session_id, language, bundle
            )
            reset(bundle)
        except KeyboardInterrupt:
            break

    print("\nDone.", file=sys.stderr)
