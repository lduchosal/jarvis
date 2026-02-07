# Feature: Latency Breakdown & Optimization

## Measured Pipeline (Qwen3-TTS-12Hz-1.7B, Apple Silicon)

Test: `"Bonjour à toute la voiture !"` — French, 1.44s audio output.

| Phase | Time | Eliminated by daemon? |
|-------|-----:|:---------------------:|
| Python imports | 2.635s | Yes |
| Model load | 2.745s | Yes |
| Generation (first chunk) | 1.123s | No |
| Generation (total) | 1.129s | No |
| Playback | 1.604s | No |

## Time to First Audio

| Mode | Latency |
|------|--------:|
| Current (`uv run q3tts.py`) | ~6.5s |
| Daemon (model hot) | ~1.2s |
| Daemon + streaming | ~1.2s (but audio starts at first chunk) |

## Where the Time Goes

- **Imports (2.6s)** — loading `transformers`, `mlx-audio`, `numpy` into memory. One-time cost in a daemon.
- **Model load (2.7s)** — reading 1.7B parameters from disk into GPU. One-time cost in a daemon.
- **Generation (1.1s)** — the actual inference work. Incompressible.
- **Playback (1.6s)** — real-time audio duration (1.44s of audio).

## Daemon Impact

The daemon eliminates 5.4s of startup overhead per request (~66% of total wall time). The remaining latency is generation + playback, which are bounded by model speed and audio duration.

With streaming playback, generation and playback overlap — audio starts playing as soon as the first chunk is ready instead of waiting for full generation to complete.
