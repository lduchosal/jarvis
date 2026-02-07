# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "sounddevice",
#     "soundfile",
# ]
# ///
import time

t0 = time.perf_counter()
import sys
import soundfile as sf
import sounddevice as sd
t_import = time.perf_counter()

t1 = time.perf_counter()
data, sr = sf.read(sys.argv[1])
t_read = time.perf_counter()

t2 = time.perf_counter()
sd.play(data, sr)
t_play = time.perf_counter()

sd.wait()
t_done = time.perf_counter()

print(f"imports:  {t_import - t0:.3f}s")
print(f"read:     {t_read - t1:.3f}s")
print(f"sd.play:  {t_play - t2:.3f}s")
print(f"playback: {t_done - t2:.3f}s")
print(f"total:    {t_done - t0:.3f}s")
