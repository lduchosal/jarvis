# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "sounddevice",
#     "soundfile",
# ]
# ///
import sys
import soundfile as sf
import sounddevice as sd

data, sr = sf.read(sys.argv[1])
sd.play(data, sr)
sd.wait()
