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
    text = text.replace('?', '.')
    text = re.sub(r'\.{2,}', '.', text)  # ... or .. -> .
    text = re.sub(r'[;:]', ',', text)    # ; : -> ,
    text = re.sub(r'[«»"„“”"]', '', text)  # remove quotes
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def trim_trailing_silence(audio: np.ndarray, sr: int, threshold: float = 0.01, min_silence_s: float = 0.3) -> np.ndarray:
    """Remove trailing silence from audio array."""
    flat = audio.flatten() if audio.ndim > 1 else audio
    above = np.where(np.abs(flat) > threshold)[0]
    if len(above) == 0:
        return audio[:0]
    last_loud = above[-1]
    tail = int(sr * min_silence_s)
    end = min(last_loud + tail, len(flat))
    return audio[:end]


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
    play_audio = output_path is None  # play on speaker only if no output file

    # Scale max_tokens with text length (short text = fewer tokens needed)
    max_tokens = max(256, min(4096, len(text) * 20))

    gen_kwargs = {
        "text": text,
        "language": language,
        "verbose": False,
        "instruct": instruct,
        "temperature": 0.7,
        "repetition_penalty": 1.2,
        "max_tokens": max_tokens,
    }

    # Cap chunks to prevent infinite generation loops
    max_chunks = max(50, len(text) * 5)

    try:
        all_audio = []
        stream = None

        if play_audio:
            stream = sd.OutputStream(samplerate=model.sample_rate, channels=1, dtype="float32")
            stream.start()

        try:
            for i, result in enumerate(model.generate_voice_design(**gen_kwargs)):
                if i >= max_chunks:
                    break
                chunk = np.array(result.audio, dtype=np.float32)
                if chunk.ndim == 1:
                    chunk = chunk.reshape(-1, 1)
                if stream is not None:
                    stream.write(chunk)
                all_audio.append(chunk)
        finally:
            if stream is not None:
                time.sleep(0.1)
                stream.stop()
                stream.close()

        # Save to file if requested
        if output_path and output_path != "/dev/null" and all_audio:
            full_audio = np.concatenate(all_audio, axis=0)
            full_audio = trim_trailing_silence(full_audio, model.sample_rate)
            sf.write(output_path, full_audio, model.sample_rate)

        return {"status": "ok"}

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return {"status": "error", "message": str(e)}
