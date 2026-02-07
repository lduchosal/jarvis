# Architecture: q3tts Streaming Daemon

## Overview

A two-file TTS daemon that keeps the Qwen3 model loaded in memory and accepts text over a Unix socket. Audio streams to speakers as it generates — no intermediate files, no reload delay.

## System Diagram

```
q3tts "hello"                q3tts_daemon.py
  (client)                     (long-lived)
     │                              │
     │── JSON request ──►  ~/.q3tts.sock
     │                              │
     │                    importlib.reload(handlers)
     │                              │
     │                    handlers.handle(model, req)
     │                              │
     │                    model.generate_voice_design()
     │                         │ (yields chunks)
     │                         ▼
     │                    sd.OutputStream.write(chunk)
     │                         │
     │                         ▼
     │                      speakers
     │                              │
     │◄── JSON response ───────────┘
```

## Two-File Structure

### `q3tts_daemon.py` — The Shell

Long-lived process. Owns the expensive state. Never needs restarting during dev.

Responsibilities:
- Load model once at startup (~5s)
- Open and manage Unix domain socket at `~/.q3tts.sock`
- Accept connections, read JSON requests
- `importlib.reload(handlers)` before each request
- Delegate to `handlers.handle(model, request)`
- Return JSON response to client
- Clean up socket file on shutdown

Does NOT contain any TTS generation logic or audio playback code.

### `handlers.py` — The Brain

Hot-reloaded on every request. Contains all the logic you iterate on.

Responsibilities:
- Parse request options (language, instruct, output file)
- Call `model.generate_voice_design()` with the right kwargs
- Stream audio chunks to `sd.OutputStream`
- Optionally save to file
- Return result status

## Socket Protocol

Unix domain socket at `~/.q3tts.sock`. One connection per request, synchronous.

### Request (client -> daemon)

```json
{
  "action": "generate",
  "text": "hello world",
  "language": "English",
  "instruct": "deep low voice",
  "output": null
}
```

| Field      | Type         | Default     | Description                              |
|------------|--------------|-------------|------------------------------------------|
| `action`   | string       | `generate`  | `generate`, `stop`, `status`, `shutdown` |
| `text`     | string       | required    | Text to synthesize                       |
| `language` | string       | `"English"` | TTS language                             |
| `instruct` | string/null  | `null`      | Voice style instruction                  |
| `output`   | string/null  | `null`      | Save to file path (null = speaker only)  |

### Response (daemon -> client)

```json
{"status": "ok"}
{"status": "error", "message": "model not loaded"}
```

### Wire Format

Length-prefixed JSON. Each message is:
```
[4 bytes: payload length as big-endian uint32][payload as UTF-8 JSON]
```

This avoids delimiter parsing issues with JSON content.

## CLI Modes

Single entry point, behavior determined by subcommand:

```
q3tts serve              # start daemon in foreground
q3tts serve --daemon     # start daemon in background (fork + detach)
q3tts "hello world"      # send text to running daemon
q3tts -o out.wav "hi"    # send text, save to file
q3tts stop               # graceful shutdown
q3tts status             # check if daemon is alive
```

**Client fallback**: if the daemon isn't running and no `serve` subcommand is given, fall back to inline mode (load model, generate, play, exit) — same as current `q3tts.py` behavior.

## Streaming Playback

Audio plays as it generates. No buffering the full result.

```python
with sd.OutputStream(samplerate=model.sample_rate, channels=1) as stream:
    for result in model.generate_voice_design(**kwargs):
        stream.write(np.array(result.audio))
```

`OutputStream.write()` in blocking mode handles backpressure naturally — it blocks when the audio buffer is full, which throttles the generator to real-time speed.

## Lifecycle

```
startup:
  1. Load model into memory (~5s)
  2. Open sd.OutputStream (keeps audio device ready)
  3. Bind Unix socket at ~/.q3tts.sock
  4. Enter request loop

request:
  1. Accept connection
  2. Read length-prefixed JSON
  3. importlib.reload(handlers)
  4. handlers.handle(model, request)
  5. Write response, close connection

shutdown (SIGTERM, SIGINT, or "stop" action):
  1. Close socket
  2. Remove ~/.q3tts.sock
  3. Release audio device
  4. Exit
```

## Dependencies

Same as current `q3tts.py`, plus `sounddevice` for streaming playback:

```
transformers>=5.0.0rc1
mlx-audio==0.3.0rc1
click
numpy
soundfile
sounddevice
```

No additional dependencies. `socket`, `json`, `importlib`, `struct`, `signal` are all stdlib.
