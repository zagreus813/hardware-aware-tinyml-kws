from pathlib import Path
import wave

import numpy as np


DATASET_ROOT = Path(
    "~/projects/tinyml-esp32/data/speech_commands_v0.02"
).expanduser()

FILE = (
    DATASET_ROOT
    / "yes"
    / "004ae714_nohash_0.wav"
)


with wave.open(str(FILE), "rb") as wav:

    sample_rate = wav.getframerate()
    channels = wav.getnchannels()
    sample_width = wav.getsampwidth()
    num_frames = wav.getnframes()

    raw_data = wav.readframes(num_frames)


print("Sample rate:", sample_rate)
print("Channels:", channels)
print("Sample width:", sample_width)
print("Frames:", num_frames)
print("Raw bytes:", len(raw_data))


audio = np.frombuffer(
    raw_data,
    dtype=np.int16
)


audio = audio.astype(
    np.float32
) / 32768.0


print("\nProcessed audio:")
print("Shape:", audio.shape)
print("dtype:", audio.dtype)
print("Minimum:", audio.min())
print("Maximum:", audio.max())
print("Mean:", audio.mean())
print("Std:", audio.std())
