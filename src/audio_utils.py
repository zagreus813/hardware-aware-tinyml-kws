from pathlib import Path
import wave

import numpy as np


TARGET_SAMPLE_RATE = 16000
TARGET_LENGTH = 16000


def load_wav(path):
    """
    Load a mono 16-bit PCM WAV file.

    Returns:
        audio: float32 array in approximately [-1, 1]
        sample_rate: integer sample rate
    """

    path = Path(path)

    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        num_frames = wav.getnframes()

        raw_data = wav.readframes(num_frames)

    if sample_rate != TARGET_SAMPLE_RATE:
        raise ValueError(
            f"Expected {TARGET_SAMPLE_RATE} Hz, "
            f"got {sample_rate} Hz: {path}"
        )

    if channels != 1:
        raise ValueError(
            f"Expected mono audio, "
            f"got {channels} channels: {path}"
        )

    if sample_width != 2:
        raise ValueError(
            f"Expected 16-bit PCM, "
            f"got {sample_width * 8}-bit: {path}"
        )

    audio = np.frombuffer(
        raw_data,
        dtype=np.int16,
    ).astype(np.float32)

    audio /= 32768.0

    return audio, sample_rate


def pad_or_trim(audio, target_length=TARGET_LENGTH):
    """
    Convert an audio waveform to a fixed length.

    Short signals are zero-padded.
    Long signals are truncated.
    """

    audio = np.asarray(
        audio,
        dtype=np.float32,
    )

    if len(audio) < target_length:

        return np.pad(
            audio,
            (0, target_length - len(audio)),
            mode="constant",
        )

    if len(audio) > target_length:

        return audio[:target_length]

    return audio.copy()
