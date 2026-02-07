# Feature: Hot-Swap Daemon Reload

## Problem

Model loading takes ~5 seconds. Python imports take ~0.7s. During development, every code change requires restarting the process, re-importing everything, and reloading the model. This kills iteration speed.

## Goal

Edit `handlers.py`, send text to the daemon, hear the result. Under 100ms from save to audio. No restart, no model reload.

## Design

### Separation of Concerns

```
q3tts_daemon.py          handlers.py
 (stable shell)          (hot-swapped)
┌──────────────┐        ┌──────────────────┐
│ model         │        │ handle(model, req)│
│ socket        │──reload──►│ generate + play   │
│ signal handler│        │ parse options     │
│ request loop  │        │ audio streaming   │
└──────────────┘        └──────────────────┘
 restart: never           reload: every request
```

**Rule**: if it's expensive to initialize, it goes in `q3tts_daemon.py`. If you want to iterate on it, it goes in `handlers.py`.

### Reload Mechanism

```python
# In q3tts_daemon.py request loop
import importlib
import handlers

while True:
    conn, _ = sock.accept()
    request = read_request(conn)

    importlib.reload(handlers)  # re-reads handlers.py from disk

    try:
        result = handlers.handle(model, request)
        send_response(conn, {"status": "ok"})
    except Exception as e:
        send_response(conn, {"status": "error", "message": str(e)})
    finally:
        conn.close()
```

`importlib.reload()`:
- Re-executes the module's top-level code
- Replaces all function/class objects in the module
- Existing references to the old module's objects are NOT updated (but we always go through `handlers.handle`, so this is fine)
- Takes <1ms for a small module

### What Gets Reloaded

| Change in `handlers.py`              | Picked up? |
|--------------------------------------|------------|
| Edit function body                   | Yes        |
| Add/remove function                  | Yes        |
| Change top-level constants           | Yes        |
| Add new import                       | Yes        |
| Change import of already-loaded lib  | No*        |

*Sub-imports are cached in `sys.modules`. If `handlers.py` imports `utils.py` and you edit `utils.py`, you'd need to reload that too. Keep hot logic in one file to avoid this.

### handlers.py Contract

```python
def handle(model, request: dict) -> dict:
    """
    Called once per request. Must be self-contained.

    Args:
        model: loaded MLX TTS model (do not reload or modify)
        request: dict with keys: text, language, instruct, output

    Returns:
        dict with status info
    """
```

The function receives the model as an argument — it never imports or loads it. This is what makes hot-reload safe: the expensive state is owned by the daemon, not the handler.

## Dev Workflow

### Terminal Layout (tmux)

```
┌─────────────────────┬──────────────────────┐
│                     │                      │
│  editor:            │  daemon:             │
│  handlers.py        │  uv run q3tts_daemon │
│                     │  > Model loaded.     │
│                     │  > Listening...      │
│                     │  > [reload] handled  │
│                     │  > [reload] handled  │
│                     │                      │
├─────────────────────┴──────────────────────┤
│  client:                                   │
│  $ q3tts "test this change"                │
│                                            │
└────────────────────────────────────────────┘
```

### Iteration Loop

1. Edit `handlers.py` in your editor
2. Save
3. Run `q3tts "test"` in the client pane (or up-arrow + enter)
4. Hear the result immediately
5. Repeat

No step involves restarting anything. The daemon prints `[reload]` on each request so you can confirm the new code is running.

## Error Handling

Errors in `handlers.py` must NOT crash the daemon.

```python
try:
    importlib.reload(handlers)
    result = handlers.handle(model, request)
except SyntaxError as e:
    # Bad code in handlers.py — report, don't crash
    send_response(conn, {"status": "error", "message": f"syntax error: {e}"})
except Exception as e:
    send_response(conn, {"status": "error", "message": str(e)})
```

The daemon survives:
- Syntax errors in `handlers.py`
- Runtime exceptions in `handle()`
- Broken imports in `handlers.py`

The daemon logs the error and waits for the next request. Fix the code, send another request — no restart needed.

## Daemon Logging

Minimal, timestamp-free, to stderr:

```
model loaded (1.7B, 4.8s)
listening on ~/.q3tts.sock
[reload] "hello world" -> speakers (0.34s)
[reload] "test" -> out.wav (0.28s)
[reload] error: name 'foo' is not defined
shutdown
```

Each `[reload]` line confirms that `handlers.py` was re-read from disk.

## Optional: File Watcher Pre-reload

For even tighter iteration, the daemon can watch `handlers.py` with `watchdog` and pre-reload on save, before any request arrives. This shaves off the <1ms reload time from the request path.

Not worth the added dependency for the initial version. `importlib.reload` on each request is fast enough.

## Implementation Checklist

- [ ] `q3tts_daemon.py`: socket setup, model loading, request loop with `importlib.reload`
- [ ] `handlers.py`: `handle(model, request)` with streaming playback
- [ ] Client mode in `q3tts.py`: detect running daemon, send JSON over socket
- [ ] Signal handling: SIGTERM/SIGINT clean up socket file
- [ ] Error isolation: syntax/runtime errors in handlers don't crash daemon
- [ ] Logging: `[reload]` line per request
