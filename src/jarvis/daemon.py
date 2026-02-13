"""Jarvis TTS daemon — asyncio event loop + multiprocessing worker pool.

Architecture:
  Main process: asyncio server, handles status/filler/shutdown, dispatches generate to workers.
  N worker processes: each loads its own MLX model, handles generate requests.
  MLX is NOT thread-safe — multiprocessing is required (one model per process).
"""

import asyncio
import gc
import importlib
import json
import logging
import multiprocessing as mp
import queue
import random
import resource
import signal
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
DEFAULT_WORKERS = 3


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


class GenerationTimeout(BaseException):
    """Inherits BaseException so it won't be caught by 'except Exception' in handlers."""
    pass


FILLER_VARIANTS = 5  # number of vocal takes per phrase

FILLERS = {
    "French": [
        "Hmm.",
        "Voyons.",
        "Alors.",
        "Bonne question.",
        "Voyons voir.",
        "Attends, je réfléchis.",
        "Laisse-moi réfléchir un instant.",
        "Ah, intéressant.",
        "OK, voyons ça.",
        "Oui, alors.",
        "Attends voir.",
        "Eh bien.",
        "C'est une bonne question, ça.",
        "Hmm, laisse-moi vérifier.",
        "OK, deux secondes.",
        "Alors, comment dire.",
        "Ah oui, d'accord.",
        "Hmm, voyons un peu.",
        "Oui, je vois.",
        "Euh, attends.",
    ],
    "English": [
        "Hmm.",
        "Let me think.",
        "Well.",
        "Good question.",
        "Let's see.",
        "One moment.",
        "OK, let me check.",
        "Right, so.",
        "Interesting.",
        "Let me think about that.",
        "Hmm, good one.",
        "OK, hang on.",
        "Yeah, so.",
        "Let me see.",
        "Ah, right.",
    ],
}


def warm_fillers(model, log):
    """Pre-generate filler audio files at startup. Cached on disk.

    Generates FILLER_VARIANTS vocal takes per phrase for natural variety.
    """
    cache_dir = Path.home() / ".cache" / "jarvis" / "fillers"
    cache_dir.mkdir(parents=True, exist_ok=True)

    filler_cache = {}
    for lang, phrases in FILLERS.items():
        prefix = lang[:2].lower()
        paths = []
        for i, phrase in enumerate(phrases):
            for v in range(FILLER_VARIANTS):
                path = cache_dir / f"{prefix}_{i:02d}_v{v}.wav"
                if path.exists():
                    paths.append(str(path))
                    continue
                log.info("generating filler", lang=lang, phrase=phrase, variant=v)
                try:
                    all_audio = []
                    max_tokens = max(256, len(phrase) * 20)
                    is_custom_voice = getattr(model.config, "tts_model_type", "") == "custom_voice"
                    if is_custom_voice:
                        gen = model.generate_custom_voice(
                            text=phrase, language=lang,
                            speaker=model.supported_speakers[0],
                            verbose=False,
                            temperature=0.7, repetition_penalty=1.2,
                            max_tokens=max_tokens,
                        )
                    else:
                        gen = model.generate_voice_design(
                            text=phrase, language=lang, instruct="",
                            verbose=False,
                            temperature=0.7, repetition_penalty=1.2,
                            max_tokens=max_tokens,
                        )
                    for result in gen:
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
                        log.warning("filler empty", phrase=phrase, variant=v)
                except Exception as e:
                    log.error("filler generation failed", phrase=phrase, variant=v, error=str(e))
        filler_cache[lang] = paths
    return filler_cache


MODEL_ALIASES = {
    "1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
}
DEFAULT_MODEL = "1.7b"


# ---------------------------------------------------------------------------
# Worker process
# ---------------------------------------------------------------------------

class WorkerLog:
    """Lightweight structlog-compatible logger for worker processes."""

    def __init__(self, worker_id):
        self.prefix = f"[worker-{worker_id}]"

    def _log(self, level, msg, **kw):
        extra = " ".join(f"{k}={v}" for k, v in kw.items()) if kw else ""
        ts = time.strftime("%H:%M:%S")
        print(f"{ts} {self.prefix} [{level}] {msg} {extra}".rstrip(), file=sys.stderr, flush=True)

    def info(self, msg, **kw):
        self._log("info", msg, **kw)

    def warning(self, msg, **kw):
        self._log("warn", msg, **kw)

    def error(self, msg, **kw):
        self._log("error", msg, **kw)

    def debug(self, msg, **kw):
        pass


def worker_loop(task_queue, result_queue, model_id, worker_id, do_fillers):
    """Worker process entry point: load model, handle generate requests.

    Each worker is a separate OS process with its own MLX model instance.
    Communication with main process via multiprocessing.Queue.
    """
    # Ignore SIGINT/SIGTERM — main process handles shutdown via poison pill
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)

    # SIGALRM for generation timeouts (safe per-process)
    def alarm_handler(signum, frame):
        raise GenerationTimeout("generation timed out")

    signal.signal(signal.SIGALRM, alarm_handler)

    warnings.filterwarnings("ignore", message="You are using a model of type")
    warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")

    log = WorkerLog(worker_id)
    log.info("loading model", model=model_id)
    t0 = time.time()

    from mlx_audio.tts.utils import load_model
    model = load_model(model_id)
    log.info("model loaded", elapsed=f"{time.time() - t0:.1f}s")

    # Worker 0 generates fillers (cached on disk, fast on subsequent runs)
    filler_cache = None
    if do_fillers:
        log.info("warming fillers")
        filler_cache = warm_fillers(model, log)
        log.info("fillers done", count=sum(len(v) for v in filler_cache.values()))

    # Signal ready to main process
    result_queue.put(("ready", worker_id, filler_cache))

    # Import handlers (hot-reloaded on each request)
    from jarvis import handlers

    # Request loop
    while True:
        task = task_queue.get()
        if task is None:
            break  # poison pill → shutdown

        request = task
        text = request.get("text", "")
        timeout_s = generation_timeout(text)

        # Hot-reload handlers from disk
        try:
            importlib.reload(handlers)
        except SyntaxError as e:
            log.error("syntax error in handlers", error=str(e))
            result_queue.put({"status": "error", "message": f"syntax error: {e}"})
            continue

        # Generation with SIGALRM timeout + retry
        for attempt in range(1, MAX_RETRIES + 1):
            signal.alarm(timeout_s)
            try:
                result = handlers.handle(model, request)
                break
            except GenerationTimeout:
                signal.alarm(0)
                if attempt < MAX_RETRIES:
                    log.warning("timeout, retrying", attempt=attempt, timeout=timeout_s, text=text[:40])
                else:
                    result = {"status": "error",
                              "message": f"generation timed out after {MAX_RETRIES}x{timeout_s}s"}
            finally:
                signal.alarm(0)

        result_queue.put(result)
        gc.collect()

    log.info("exiting")


# ---------------------------------------------------------------------------
# Worker Pool
# ---------------------------------------------------------------------------

class WorkerPool:
    """Manages N worker processes, dispatches generate requests round-robin."""

    def __init__(self, n_workers, model_id, log):
        self._log = log
        self._n_workers = n_workers
        self._workers = []  # [(Process, task_q, result_q), ...]
        self._free = asyncio.Queue()  # worker_ids of available workers

        for i in range(n_workers):
            task_q = mp.Queue()
            result_q = mp.Queue()
            p = mp.Process(
                target=worker_loop,
                args=(task_q, result_q, model_id, i, i == 0),
                daemon=True,
            )
            p.start()
            self._workers.append((p, task_q, result_q))
            log.info("worker started", worker=i, pid=p.pid)

    async def wait_ready(self) -> dict:
        """Wait for all workers to load their models. Returns filler_cache from worker 0."""
        filler_cache = {}

        async def wait_one(i):
            nonlocal filler_cache
            p, _, result_q = self._workers[i]
            msg = await asyncio.to_thread(result_q.get)
            _, wid, fc = msg
            if fc is not None:
                filler_cache = fc
            self._free.put_nowait(wid)
            self._log.info("worker ready", worker=wid, pid=p.pid)

        await asyncio.gather(*[wait_one(i) for i in range(self._n_workers)])
        return filler_cache

    async def submit(self, request) -> dict:
        """Submit a generate request to the next available worker. Awaits if all busy."""
        worker_id = await self._free.get()
        p, task_q, result_q = self._workers[worker_id]

        await asyncio.to_thread(task_q.put, request)

        try:
            result = await asyncio.to_thread(result_q.get, True, 120)
        except queue.Empty:
            result = {"status": "error", "message": "worker timeout"}

        if p.is_alive():
            self._free.put_nowait(worker_id)
        else:
            self._log.error("worker died", worker=worker_id)

        return result

    def shutdown(self):
        """Send poison pills and join all worker processes."""
        for _, task_q, _ in self._workers:
            try:
                task_q.put(None)
            except Exception:
                pass

        for p, _, _ in self._workers:
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join(timeout=2)


# ---------------------------------------------------------------------------
# Async socket protocol (length-prefixed JSON, same as cli.py)
# ---------------------------------------------------------------------------

async def async_read_message(reader: asyncio.StreamReader) -> dict:
    """Read a length-prefixed JSON message from an async stream."""
    raw_len = await reader.readexactly(4)
    msg_len = struct.unpack("!I", raw_len)[0]
    data = await reader.readexactly(msg_len)
    return json.loads(data.decode("utf-8"))


async def async_send_message(writer: asyncio.StreamWriter, msg: dict):
    """Send a length-prefixed JSON message to an async stream."""
    payload = json.dumps(msg).encode("utf-8")
    writer.write(struct.pack("!I", len(payload)) + payload)
    await writer.drain()


# ---------------------------------------------------------------------------
# Main server
# ---------------------------------------------------------------------------

async def serve(model_id: str, n_workers: int, log):
    """Async main loop: start workers, accept connections, dispatch requests."""
    # Clean up stale socket
    if SOCKET_PATH.exists():
        log.warning("removing stale socket", path=str(SOCKET_PATH))
        SOCKET_PATH.unlink()

    # Start worker pool and wait for readiness
    log.info("starting workers", model=model_id, workers=n_workers)
    pool = WorkerPool(n_workers, model_id, log)
    filler_cache = await pool.wait_ready()
    log.info("all workers ready", fillers=sum(len(v) for v in filler_cache.values()))

    shutdown_event = asyncio.Event()
    request_count = 0

    async def handle_client(reader, writer):
        nonlocal request_count
        try:
            request = await async_read_message(reader)
            action = request.get("action", "generate")

            if action == "shutdown":
                await async_send_message(writer, {"status": "ok"})
                log.info("shutdown requested by client")
                shutdown_event.set()
                return

            if action == "status":
                await async_send_message(writer, {
                    "status": "ok",
                    "model": "loaded",
                    "workers": n_workers,
                    "requests_served": request_count,
                    "memory_mb": mem_mb(),
                })
                return

            if action == "get_filler":
                lang = request.get("language", "French")
                paths = filler_cache.get(lang, [])
                if paths:
                    await async_send_message(writer, {"status": "ok", "path": random.choice(paths)})
                else:
                    await async_send_message(writer, {"status": "error", "message": f"no fillers for {lang}"})
                return

            if action == "generate":
                request_count += 1
                req_num = request_count
                t_start = time.time()
                text = request.get("text", "")
                output = request.get("output")
                dest = output or "speakers"

                log.debug("request received", req=req_num, text=text[:60], dest=dest)
                result = await pool.submit(request)
                elapsed = time.time() - t_start

                status = result.get("status", "?")
                if status == "ok":
                    log.info("request done", req=req_num, elapsed=f"{elapsed:.1f}s",
                             text=text[:40], dest=dest)
                else:
                    log.error("request failed", req=req_num, elapsed=f"{elapsed:.1f}s",
                              text=text[:40], error=result.get("message"))

                await async_send_message(writer, result)
                return

            await async_send_message(writer, {"status": "error", "message": f"unknown action: {action}"})

        except asyncio.IncompleteReadError:
            pass  # client disconnected
        except ConnectionResetError:
            pass
        except Exception as e:
            log.error("client error", error=str(e), traceback=traceback.format_exc())
            try:
                await async_send_message(writer, {"status": "error", "message": str(e)})
            except Exception:
                pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    # Signal handlers → set shutdown event
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown_event.set)

    # Start accepting connections
    server = await asyncio.start_unix_server(handle_client, path=str(SOCKET_PATH))
    log.info("listening", socket=str(SOCKET_PATH))

    # Serve until shutdown
    await shutdown_event.wait()

    # Graceful shutdown
    server.close()
    try:
        await asyncio.wait_for(server.wait_closed(), timeout=5)
    except asyncio.TimeoutError:
        log.warning("forcing server close")

    pool.shutdown()
    if SOCKET_PATH.exists():
        SOCKET_PATH.unlink()
    log.info("shutdown complete", requests_served=request_count)


def main(model_name: str | None = None, n_workers: int = DEFAULT_WORKERS):
    log = setup_logging()

    # Resolve model name
    model_key = (model_name or DEFAULT_MODEL).lower()
    model_id = MODEL_ALIASES.get(model_key, model_key)

    asyncio.run(serve(model_id, n_workers, log))


if __name__ == "__main__":
    main()
