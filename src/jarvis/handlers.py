"""TTS request handler — hot-reloaded by the daemon on every request."""

import time
import traceback
import sys

import re

import numpy as np
import sounddevice as sd
import soundfile as sf


def sanitize_text(text: str) -> str:
    """Clean up punctuation that causes TTS artifacts."""
    text = re.sub(r'\.{2,}', '.', text)  # ... or .. -> .
    text = re.sub(r'[;:]', ',', text)    # ; : -> ,
    text = re.sub(r'[«»"„“”"]', '', text)  # remove quotes
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def trim_trailing_silence(audio: np.ndarray, sr: int, threshold: float = 0.02, min_silence_s: float = 0.15) -> np.ndarray:
    """Remove trailing silence and low-energy artifacts from audio."""
    flat = audio.flatten() if audio.ndim > 1 else audio
    win = int(sr * 0.02)  # 20ms windows
    n_wins = len(flat) // win
    if n_wins == 0:
        return audio[:0]
    trimmed = flat[:n_wins * win].reshape(n_wins, win)
    rms = np.sqrt(np.mean(trimmed ** 2, axis=1))
    above = np.where(rms > threshold)[0]
    if len(above) == 0:
        return audio[:0]
    last_loud = (above[-1] + 1) * win
    tail = int(sr * min_silence_s)
    end = min(last_loud + tail, len(flat))
    return audio[:end]


MAX_RETRIES = 3
BAD_RMS_THRESHOLD = 0.1


def _generate_audio(gen_method, gen_kwargs, max_chunks, sr):
    """Run generation and return (chunks, avg_rms)."""
    all_audio = []
    rms_values = []
    silent_streak = 0

    for i, result in enumerate(gen_method(**gen_kwargs)):
        if i >= max_chunks:
            print(f"[TTS] chunk limit reached ({max_chunks})", file=sys.stderr)
            break
        chunk = np.array(result.audio, dtype=np.float32)
        if chunk.ndim == 1:
            chunk = chunk.reshape(-1, 1)
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        dur_ms = len(chunk) / sr * 1000
        print(f"[TTS] chunk {i}: {dur_ms:.0f}ms rms={rms:.4f}", file=sys.stderr)
        rms_values.append(rms)
        if rms < 0.02:
            silent_streak += 1
            if silent_streak >= 3:
                print(f"[TTS] stopping: {silent_streak} silent chunks in a row", file=sys.stderr)
                break
            continue
        else:
            silent_streak = 0
        all_audio.append(chunk)

    avg_rms = float(np.mean(rms_values)) if rms_values else 0.0
    return all_audio, avg_rms


def handle(model, request: dict) -> dict:
    """
    Generate audio from text and stream to speakers.

    Args:
        model: loaded MLX TTS model (owned by daemon, do not reload)
        request: dict with keys: text, language, instruct, output

    Returns:
        dict with status info
    """
    text = request.get("text")
    if not text or not text.strip():
        return {"status": "error", "message": "no text provided"}

    text = sanitize_text(text)
    language = request.get("language", "English")
    instruct = request.get("instruct") or ""
    output_path = request.get("output")
    play_audio = output_path is None

    max_tokens = max(256, min(4096, len(text) * 20))

    gen_kwargs = {
        "text": text,
        "language": language,
        "verbose": False,
        "temperature": 0.7,
        "repetition_penalty": 1.2,
        "max_tokens": max_tokens,
    }

    is_custom_voice = getattr(model.config, "tts_model_type", "") == "custom_voice"
    if is_custom_voice:
        gen_kwargs["speaker"] = request.get("speaker") or model.supported_speakers[0]
        gen_method = model.generate_custom_voice
    else:
        gen_kwargs["instruct"] = instruct
        gen_method = model.generate_voice_design

    max_chunks = max(50, len(text) * 5)
    sr = model.sample_rate

    log_kwargs = {k: v for k, v in gen_kwargs.items() if k != "text"}
    print(f"[TTS] text={text!r} params={log_kwargs}", file=sys.stderr)

    try:
        t0 = time.monotonic()
        best_audio = None
        best_rms = float("inf")

        for attempt in range(MAX_RETRIES):
            all_audio, avg_rms = _generate_audio(gen_method, gen_kwargs, max_chunks, sr)
            print(f"[TTS] attempt {attempt + 1}/{MAX_RETRIES}: avg_rms={avg_rms:.4f}", file=sys.stderr)

            if avg_rms < BAD_RMS_THRESHOLD:
                best_audio = all_audio
                best_rms = avg_rms
                break

            # Keep the best attempt so far
            if avg_rms < best_rms:
                best_audio = all_audio
                best_rms = avg_rms

            print(f"[TTS] bad quality (rms={avg_rms:.4f} >= {BAD_RMS_THRESHOLD}), retrying...", file=sys.stderr)

        all_audio = best_audio or []
        elapsed = time.monotonic() - t0
        total_samples = sum(len(c) for c in all_audio)
        total_dur = total_samples / sr if all_audio else 0
        print(f"[TTS] done: {len(all_audio)} chunks, {total_dur:.1f}s audio, {elapsed:.1f}s wall, rms={best_rms:.4f}", file=sys.stderr)

        # Play audio
        if play_audio and all_audio:
            stream = sd.OutputStream(samplerate=sr, channels=1, dtype="float32")
            stream.start()
            try:
                for chunk in all_audio:
                    stream.write(chunk)
            finally:
                time.sleep(0.1)
                stream.stop()
                stream.close()

        # Save to file if requested
        if output_path and output_path != "/dev/null" and all_audio:
            full_audio = np.concatenate(all_audio, axis=0)
            full_audio = trim_trailing_silence(full_audio, sr)
            sf.write(output_path, full_audio, sr)

        return {"status": "ok"}

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return {"status": "error", "message": str(e)}
