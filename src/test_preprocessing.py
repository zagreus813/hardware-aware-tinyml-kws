from pathlib import Path

import numpy as np

from audio_utils import load_wav, pad_or_trim
from feature_extraction import extract_mfcc


DATASET_ROOT = Path(
    "~/projects/tinyml-esp32/data/speech_commands_v0.02"
).expanduser()


FILES = [
    DATASET_ROOT
    / "yes"
    / "37dca74f_nohash_1.wav",

    DATASET_ROOT
    / "up"
    / "37dca74f_nohash_1.wav",

    DATASET_ROOT
    / "down"
    / "37dca74f_nohash_1.wav",
]


def test_file(path):
    print("=" * 60)
    print("Testing:", path)

    audio, sample_rate = load_wav(path)

    print(
        "Original:",
        audio.shape,
        sample_rate,
    )

    audio = pad_or_trim(
        audio
    )

    print(
        "Fixed length:",
        audio.shape,
    )

    mfcc = extract_mfcc(
        audio
    )

    print(
        "MFCC shape:",
        mfcc.shape,
    )

    print(
        "MFCC dtype:",
        mfcc.dtype,
    )

    print(
        "Finite values:",
        np.isfinite(mfcc).all(),
    )

    assert sample_rate == 16000
    assert audio.shape == (16000,)
    assert mfcc.shape == (13, 98)
    assert mfcc.dtype == np.float32
    assert np.isfinite(mfcc).all()


for path in FILES:

    if not path.exists():
        print(
            "WARNING: file not found:",
            path,
        )
        continue

    test_file(path)


print("\nAll preprocessing tests passed.")
