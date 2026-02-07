"""TTS request handler — hot-reloaded by the daemon on every request."""

import time
import traceback
import sys

import numpy as np
import sounddevice as sd
import soundfile as sf


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

    language = request.get("language", "English")
    instruct = request.get("instruct") or ""
    output_path = request.get("output")
    play_audio = output_path is None  # play on speaker only if no output file

    gen_kwargs = {
        "text": text,
        "language": language,
        "verbose": False,
        "instruct": instruct,
    }

    try:
        all_audio = []
        stream = None

        if play_audio:
            stream = sd.OutputStream(samplerate=model.sample_rate, channels=1, dtype="float32")
            stream.start()

        try:
            for result in model.generate_voice_design(**gen_kwargs):
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
            sf.write(output_path, full_audio, model.sample_rate)

        return {"status": "ok"}

    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        return {"status": "error", "message": str(e)}
