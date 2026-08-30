from pathlib import Path

import numpy as np

from audio_utils import (
    load_wav,
    pad_or_trim,
)

from feature_extraction import (
    extract_log_mel,
)


DATASET_ROOT = (
    Path(
        "~/projects/tinyml-esp32/"
        "data/speech_commands_v0.02"
    ).expanduser()
)

FILE = (
    DATASET_ROOT
    / "yes"
    / "37dca74f_nohash_1.wav"
)


audio, sample_rate = load_wav(
    FILE
)

audio = pad_or_trim(
    audio
)

log_mel = extract_log_mel(
    audio
)

print("Sample rate:", sample_rate)
print("Audio shape:", audio.shape)

print(
    "Log-Mel shape:",
    log_mel.shape
)

print(
    "dtype:",
    log_mel.dtype
)

print(
    "min:",
    float(log_mel.min())
)

print(
    "max:",
    float(log_mel.max())
)

print(
    "mean:",
    float(log_mel.mean())
)

print(
    "std:",
    float(log_mel.std())
)

print(
    "Finite:",
    np.isfinite(log_mel).all()
)
