# jarvis
Local text-to-speech CLI powered by Qwen3-TTS and MLX. Runs entirely on Apple Silicon with no cloud API. Daemon mode keeps the model hot in memory for sub-second latency. Supports voice styling, multiple languages, and streaming playback.

## Panel (`jah panel`)

Multi-model debate mode. Up to 6 AI participants discuss your questions in randomized order each round.

**Participants:** Opus, Sonnet, Haiku (Claude), Codex (OpenAI), Gemini 2.5, Gemini 3.0 (Google)

```
jah panel                              # all available models
jah panel -p opus,codex                # only Opus + Codex
jah panel -p gemini-2.5,haiku          # only Gemini 2.5 + Haiku
jah panel --tts                        # read responses aloud
jah panel -r latest                    # resume last session
```

**API keys:** store in `~/.config/jarvis/keys` (one per line, `KEY=value` format). Gemini requires `GEMINI_API_KEY`. Claude and Codex use their respective SDK credentials.
