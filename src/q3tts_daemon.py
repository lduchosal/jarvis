# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "transformers>=5.0.0rc1",
#     "mlx-audio==0.3.0rc1",
#     "numpy",
#     "sounddevice",
#     "soundfile",
# ]
# ///
"""q3tts daemon — keeps the model loaded, accepts requests over Unix socket."""

import gc
import importlib
import json
import os
import resource
import signal
import socket
import struct
import sys
import threading
import time
import traceback

from pathlib import Path

SOCKET_PATH = Path.home() / ".q3tts.sock"
GENERATION_TIMEOUT = 60  # seconds


def log(msg: str):
    print(msg, file=sys.stderr, flush=True)


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


def run_handler_with_timeout(handlers, model, request, timeout=GENERATION_TIMEOUT):
    """Run handlers.handle in a thread with a timeout. Returns result dict."""
    result = [None]
    error = [None]

    def target():
        try:
            result[0] = handlers.handle(model, request)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            error[0] = e

    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        log(f"[timeout] generation exceeded {timeout}s, aborting")
        # Thread is stuck — we can't kill it, but we respond to client
        return {"status": "error", "message": f"generation timed out after {timeout}s"}

    if error[0] is not None:
        return {"status": "error", "message": str(error[0])}

    return result[0] or {"status": "error", "message": "no result"}


def main():
    # Clean up stale socket
    if SOCKET_PATH.exists():
        log(f"removing stale socket {SOCKET_PATH}")
        SOCKET_PATH.unlink()

    # Load model once
    log("loading model...")
    t0 = time.time()
    from mlx_audio.tts.utils import load_model
    model = load_model("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
    log(f"model loaded (1.7B, {time.time() - t0:.1f}s)")
    log(f"[mem] {mem_mb()}MB")

    # Add src/ to sys.path so we can import handlers
    src_dir = str(Path(__file__).parent)
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    import handlers

    # Bind socket
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.bind(str(SOCKET_PATH))
    sock.listen(5)
    log(f"listening on {SOCKET_PATH}")

    # Signal handling for clean shutdown
    shutdown = False

    def handle_signal(signum, frame):
        nonlocal shutdown
        shutdown = True
        log("shutdown")
        sock.close()

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

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

            if action == "generate":
                request_count += 1
                t_start = time.time()

                # Hot-reload handlers
                importlib.reload(handlers)

                # Run with timeout
                result = run_handler_with_timeout(handlers, model, request)
                send_message(conn, result)

                elapsed = time.time() - t_start
                text_preview = request.get("text", "")[:40]
                output = request.get("output")
                dest = output if output else "speakers"
                status = result.get("status", "?")
                log(f'[{request_count}] {status} {elapsed:.1f}s "{text_preview}" -> {dest} [mem:{mem_mb()}MB]')

                # GC after each request to prevent memory buildup
                gc.collect()

        except SyntaxError as e:
            log(f"[error] syntax error: {e}")
            traceback.print_exc(file=sys.stderr)
            try:
                send_message(conn, {"status": "error", "message": f"syntax error: {e}"})
            except Exception:
                pass
        except Exception as e:
            log(f"[error] {e}")
            traceback.print_exc(file=sys.stderr)
            try:
                send_message(conn, {"status": "error", "message": str(e)})
            except Exception:
                pass
        finally:
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
    log(f"shutdown complete ({request_count} requests served)")


if __name__ == "__main__":
    main()
