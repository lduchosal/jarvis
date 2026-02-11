"""Jarvis TTS daemon — keeps the model loaded, accepts requests over Unix socket."""

import gc
import importlib
import json
import logging
import random
import resource
import signal
import socket
import struct
import sys
import time
import traceback
import warnings

warnings.filterwarnings("ignore", message="You are using a model of type")
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")

from pathlib import Path

import numpy as np
import soundfile as sf
import structlog

SOCKET_PATH = Path.home() / ".q3tts.sock"
LOG_DIR = Path(__file__).parent.parent.parent / "logs"
MAX_RETRIES = 2


def generation_timeout(text: str) -> int:
    """Scale timeout with text length: 10s base + 0.1s/char, max 60s."""
    return min(60, max(10, 10 + len(text) // 10))


def setup_logging():
    """Configure structlog with console + file output."""
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / "daemon.log"

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if sys.stderr.isatty() else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=open(log_file, "a")),
    )

    return structlog.get_logger()


def mem_mb() -> int:
    """Current RSS memory in MB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // (1024 * 1024)


def read_message(conn: socket.socket) -> dict:
    """Read a length-prefixed JSON message from the connection."""
    raw_len = b""
    while len(raw_len) < 4:
        chunk = conn.recv(4 - len(raw_len))
        if not chunk:
            raise ConnectionError("client disconnected")
        raw_len += chunk
    msg_len = struct.unpack("!I", raw_len)[0]

    data = b""
    while len(data) < msg_len:
        chunk = conn.recv(msg_len - len(data))
        if not chunk:
            raise ConnectionError("client disconnected")
        data += chunk

    return json.loads(data.decode("utf-8"))


def send_message(conn: socket.socket, msg: dict):
    """Send a length-prefixed JSON message to the connection."""
    payload = json.dumps(msg).encode("utf-8")
    conn.sendall(struct.pack("!I", len(payload)) + payload)


class GenerationTimeout(BaseException):
    """Inherits BaseException so it won't be caught by 'except Exception' in handlers."""
    pass


FILLERS = {
    "French": ["Hmm.", "Voyons.", "Alors.", "Bonne question.", "Voyons voir."],
    "English": ["Hmm.", "Let me think.", "Well.", "Good question.", "Let's see."],
}


def warm_fillers(model, log):
    """Pre-generate filler audio files at startup. Cached on disk."""
    cache_dir = Path.home() / ".cache" / "jarvis" / "fillers"
    cache_dir.mkdir(parents=True, exist_ok=True)

    filler_cache = {}
    for lang, phrases in FILLERS.items():
        prefix = lang[:2].lower()
        paths = []
        for i, phrase in enumerate(phrases):
            path = cache_dir / f"{prefix}_{i:02d}.wav"
            if path.exists():
                paths.append(str(path))
                continue
            log.info("generating filler", lang=lang, phrase=phrase)
            try:
                all_audio = []
                max_tokens = max(256, len(phrase) * 20)
                for result in model.generate_voice_design(
                    text=phrase, language=lang, instruct="",
                    verbose=False,
                    temperature=0.7, repetition_penalty=1.2,
                    max_tokens=max_tokens,
                ):
                    chunk = np.array(result.audio, dtype=np.float32)
                    all_audio.append(chunk)
                if all_audio:
                    audio = np.concatenate(all_audio)
                    # Trim trailing silence
                    flat = audio.flatten() if audio.ndim > 1 else audio
                    above = np.where(np.abs(flat) > 0.01)[0]
                    if len(above) > 0:
                        end = min(above[-1] + int(model.sample_rate * 0.3), len(flat))
                        audio = audio[:end]
                    sf.write(str(path), audio, model.sample_rate)
                    paths.append(str(path))
                    log.info("filler ready", path=str(path))
                else:
                    log.warning("filler empty", phrase=phrase)
            except Exception as e:
                log.error("filler generation failed", phrase=phrase, error=str(e))
        filler_cache[lang] = paths
    return filler_cache


def main():
    log = setup_logging()

    # Clean up stale socket
    if SOCKET_PATH.exists():
        log.warning("removing stale socket", path=str(SOCKET_PATH))
        SOCKET_PATH.unlink()

    # Load model once
    log.info("loading model")
    t0 = time.time()
    from mlx_audio.tts.utils import load_model
    model = load_model("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    load_time = time.time() - t0
    log.info("model loaded", params="1.7B", load_time=f"{load_time:.1f}s", mem_mb=mem_mb())

    # Pre-generate fillers
    filler_cache = warm_fillers(model, log)
    log.info("fillers ready", count=sum(len(v) for v in filler_cache.values()))

    # Import handlers (hot-reloaded on each request)
    from jarvis import handlers

    # Bind socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(SOCKET_PATH))
    sock.listen(5)
    log.info("listening", socket=str(SOCKET_PATH))

    # Signal handling for clean shutdown
    shutdown = False

    def handle_shutdown(signum, frame):
        nonlocal shutdown
        shutdown = True
        log.info("shutdown signal received", signal=signum)
        sock.close()

    def handle_alarm(signum, frame):
        raise GenerationTimeout("generation timed out")

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGALRM, handle_alarm)

    # Request loop
    request_count = 0
    while not shutdown:
        try:
            conn, _ = sock.accept()
        except OSError:
            break  # socket closed by signal handler

        try:
            request = read_message(conn)
            action = request.get("action", "generate")

            if action == "shutdown":
                send_message(conn, {"status": "ok"})
                conn.close()
                log.info("shutdown requested by client")
                break

            if action == "status":
                send_message(conn, {
                    "status": "ok",
                    "model": "loaded",
                    "requests_served": request_count,
                    "memory_mb": mem_mb(),
                })
                conn.close()
                continue

            if action == "get_filler":
                lang = request.get("language", "French")
                paths = filler_cache.get(lang, [])
                if paths:
                    path = random.choice(paths)
                    send_message(conn, {"status": "ok", "path": path})
                else:
                    send_message(conn, {"status": "error", "message": f"no fillers for {lang}"})
                conn.close()
                continue

            if action == "generate":
                request_count += 1
                t_start = time.time()
                text = request.get("text", "")
                output = request.get("output")
                dest = output if output else "speakers"

                log.debug("request received", req=request_count, text=text[:60], dest=dest)

                # Hot-reload handlers
                importlib.reload(handlers)

                # Run on main thread with alarm timeout + retry
                timeout = generation_timeout(text)
                for attempt in range(1, MAX_RETRIES + 1):
                    signal.alarm(timeout)
                    try:
                        result = handlers.handle(model, request)
                        break
                    except GenerationTimeout:
                        signal.alarm(0)
                        if attempt < MAX_RETRIES:
                            log.warning("timeout, retrying", req=request_count,
                                        attempt=attempt, timeout=timeout, text=text[:40])
                        else:
                            result = {"status": "error",
                                      "message": f"generation timed out after {MAX_RETRIES}x{timeout}s"}
                    finally:
                        signal.alarm(0)

                send_message(conn, result)

                elapsed = time.time() - t_start
                status = result.get("status", "?")

                if status == "ok":
                    log.info("request done", req=request_count, status=status,
                             elapsed=f"{elapsed:.1f}s", text=text[:40], dest=dest, mem_mb=mem_mb())
                else:
                    log.error("request failed", req=request_count, status=status,
                              elapsed=f"{elapsed:.1f}s", text=text[:40], dest=dest, mem_mb=mem_mb(),
                              error=result.get("message"))

                # GC after each request to prevent memory buildup
                gc.collect()

        except GenerationTimeout as e:
            log.error("timeout", error=str(e))
            try:
                send_message(conn, {"status": "error", "message": str(e)})
            except Exception:
                pass
        except SyntaxError as e:
            log.error("syntax error in handlers", error=str(e), traceback=traceback.format_exc())
            try:
                send_message(conn, {"status": "error", "message": f"syntax error: {e}"})
            except Exception:
                pass
        except Exception as e:
            log.error("unhandled error", error=str(e), traceback=traceback.format_exc())
            try:
                send_message(conn, {"status": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            signal.alarm(0)  # always cancel any pending alarm
            try:
                conn.close()
            except Exception:
                pass

    # Cleanup
    try:
        sock.close()
    except Exception:
        pass
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()
    log.info("shutdown complete", requests_served=request_count)


if __name__ == "__main__":
    main()
