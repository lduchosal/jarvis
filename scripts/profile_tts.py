# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "transformers>=5.0.0rc1",
#     "mlx-audio==0.3.0rc1",
#     "click",
#     "numpy",
#     "soundfile",
#     "sounddevice",
# ]
# ///
import time

t0 = time.perf_counter()
import numpy as np
import soundfile as sf
import sounddevice as sd
from mlx_audio.tts.utils import load_model
t_import = time.perf_counter()

t1 = time.perf_counter()
model = load_model("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign")
t_model = time.perf_counter()

gen_kwargs = {
    "text": "Bonjour à toute la voiture !",
    "language": "French",
    "verbose": False,
    "instruct": "warm masculine voice, joyful and enthusiastic",
}

t2 = time.perf_counter()
first_chunk_time = None
chunks = []
for result in model.generate_voice_design(**gen_kwargs):
    if first_chunk_time is None:
        first_chunk_time = time.perf_counter()
    chunks.append(np.array(result.audio))
t_gen = time.perf_counter()

audio = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]

t3 = time.perf_counter()
sd.play(audio, model.sample_rate)
sd.wait()
t_play = time.perf_counter()

print(f"imports:          {t_import - t0:.3f}s")
print(f"model load:       {t_model - t1:.3f}s")
print(f"first chunk:      {first_chunk_time - t2:.3f}s")
print(f"full generation:  {t_gen - t2:.3f}s")
print(f"playback:         {t_play - t3:.3f}s")
print(f"")
print(f"time to first audio (current):  {t_import - t0 + t_model - t1 + t_gen - t2 + 0.06:.3f}s")
print(f"time to first audio (daemon):   {first_chunk_time - t2 + 0.06:.3f}s")
print(f"total:            {t_play - t0:.3f}s")
print(f"audio duration:   {len(audio) / model.sample_rate:.2f}s")
