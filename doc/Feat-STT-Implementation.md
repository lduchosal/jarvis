# Plan : `jah listen` — STT avec Kyutai

## Contexte

Le TTS fonctionne (96.3% success rate). On ajoute le STT pour la boucle conversationnelle : micro → texte → (futur: Claude → TTS). Kyutai STT 1B est installé et testé — transcription parfaite à 25.8 tokens/sec.

**Contrainte critique** : MLX n'est PAS thread-safe. Le STT est un processus standalone (pas dans le daemon TTS).

## Architecture

`jah listen` = commande foreground standalone qui :
1. Charge le modèle Kyutai STT 1B (`kyutai/stt-1b-en_fr-mlx`)
2. Ouvre le micro via `sounddevice.InputStream` (24kHz, mono, blocksize=1920 = 80ms)
3. Encode audio → tokens via `rustymimi.StreamTokenizer` (non-blocking, adapté au temps réel)
4. Passe les tokens dans `LmGen.step()` → tokens texte
5. Affiche le texte transcrit en streaming (stdout)
6. Ctrl+C pour arrêter proprement

## Détails techniques du modèle STT

- HF repo: `kyutai/stt-1b-en_fr-mlx`
- Config: `dep_q=0` (pas de génération audio), `n_q=32` codebooks, `dim=2048`, `text_card=8000`
- Sampling texte: `temp=0.0, top_k=50` (quasi-greedy)
- Sampling audio: `temp=0.8, top_k=250` (pas utilisé pour STT)
- `stt_config.audio_delay_seconds=0.5` — pad 0.5s de silence à droite
- `other_codebooks = 32` (tous sont "other" car `dep_q=0`)
- `gen.step(audio_tokens, ct)` → `(text_token, transformer_out)` — on ne prend que `text_token`
- Tokens texte 0 et 3 = padding/spéciaux → filtrer
- `rustymimi.StreamTokenizer` pour encode/decode non-bloquant (vs `Tokenizer` pour batch)
- Le modèle peut avoir un `condition_provider` avec condition `"description"` → `"very_good"`

## Fichiers modifiés

### 1. `src/jarvis/stt.py` — NEW — Chargement modèle + transcription streaming

- `load_model()` : charge config HF, build Lm, quantize si besoin, warmup, crée LmGen
- `listen()` : boucle micro → encode → step → print texte

Référence : `moshi_mlx/run_inference.py` (boucle step) et `moshi_mlx/local.py` (StreamTokenizer + sounddevice)

### 2. `src/jarvis/cli.py` — EDIT — Commande `listen`

- Ajouter `"listen"` dans `SUBCOMMANDS`
- `@cli.command() def listen(duration)` avec `--duration` optionnel

### 3. `pyproject.toml` — NO CHANGE

`rustymimi`, `sentencepiece`, `sphn` déjà dans les dépendances. `moshi-mlx` installé via pip --no-deps.

## Flux audio

```
sounddevice.InputStream callback (80ms @ 24kHz)
    │
    ▼
audio_tokenizer.encode(pcm)          # Non-blocking, push PCM
audio_tokenizer.get_encoded()         # Poll for encoded tokens
    │
    ▼
mx.array(encoded).transpose(1,0)[:, :32]  # Shape pour gen.step
    │
    ▼
gen.step(audio_tokens, ct) → (text_token, _)
    │
    ▼
if text_token not in (0, 3):
    text_tokenizer.id_to_piece(token)
    print(text, end="", flush=True)
```

## Vérification

1. `pdm run jah listen` → modèle charge, "Listening..." s'affiche
2. Parler dans le micro → texte apparaît en streaming
3. Ctrl+C → arrêt propre avec stats (steps, tok/s)
4. `pdm run jah listen --duration 10` → écoute 10s puis s'arrête
5. Vérifier que le daemon TTS (`jah serve`) peut tourner en parallèle (processus séparés)
