from pathlib import Path
import json
import time

import numpy as np

from dataset import (
    TARGET_CLASSES,
    get_split_files,
    get_label,
)

from audio_utils import (
    load_wav,
    pad_or_trim,
)

from feature_extraction import (
    extract_log_mel,
)


# ============================================================
# Configuration
# ============================================================

CACHE_DIR = (
    Path("~/projects/tinyml-esp32/cache_log_mel")
    .expanduser()
)

EXPERIMENTS_DIR = (
    Path("~/projects/tinyml-esp32/experiments")
    .expanduser()
)

NORMALIZATION_FILE = (
    EXPERIMENTS_DIR
    / "log_mel_normalization.json"
)

FEATURE_SHAPE = (40, 98, 1)


# ============================================================
# Compute train normalization statistics
# ============================================================

def compute_train_statistics():

    train_files = get_split_files("train")

    print("=" * 60)
    print("Computing Log-Mel normalization statistics")
    print("=" * 60)

    print("Training samples:", len(train_files))

    sum_values = np.zeros(
        40,
        dtype=np.float64,
    )

    sum_squared = np.zeros(
        40,
        dtype=np.float64,
    )

    total_frames = 0

    start_time = time.time()

    for index, path in enumerate(
        train_files,
        start=1,
    ):

        audio, sample_rate = load_wav(
            path
        )

        audio = pad_or_trim(
            audio
        )

        log_mel = extract_log_mel(
            audio
        )

        sum_values += np.sum(
            log_mel,
            axis=1,
            dtype=np.float64,
        )

        sum_squared += np.sum(
            log_mel ** 2,
            axis=1,
            dtype=np.float64,
        )

        total_frames += log_mel.shape[1]

        if index % 1000 == 0:

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"[{index}/{len(train_files)}] "
                f"{index / elapsed:.2f} files/sec"
            )

    mean = (
        sum_values
        / total_frames
    )

    variance = (
        sum_squared
        / total_frames
        - mean ** 2
    )

    variance = np.maximum(
        variance,
        0.0,
    )

    std = np.sqrt(
        variance
    )

    std = np.maximum(
        std,
        1e-6,
    )

    stats = {
        "mean": mean.reshape(
            40,
            1,
        ).tolist(),

        "std": std.reshape(
            40,
            1,
        ).tolist(),

        "num_training_files": len(
            train_files
        ),
    }

    EXPERIMENTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        NORMALIZATION_FILE,
        "w",
    ) as file:

        json.dump(
            stats,
            file,
            indent=2,
        )

    print("\nSaved:")
    print(NORMALIZATION_FILE)

    return (
        mean.reshape(40, 1).astype(np.float32),
        std.reshape(40, 1).astype(np.float32),
    )


# ============================================================
# Cache one split
# ============================================================

def cache_split(
    split,
    mean,
    std,
):

    paths = get_split_files(split)

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_path = (
        CACHE_DIR
        / f"{split}_features.npy"
    )

    label_path = (
        CACHE_DIR
        / f"{split}_labels.npy"
    )

    features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            len(paths),
            *FEATURE_SHAPE,
        ),
    )

    labels = np.lib.format.open_memmap(
        label_path,
        mode="w+",
        dtype=np.int32,
        shape=(len(paths),),
    )

    start_time = time.time()

    for index, path in enumerate(
        paths
    ):

        audio, sample_rate = load_wav(
            path
        )

        audio = pad_or_trim(
            audio
        )

        log_mel = extract_log_mel(
            audio
        )

        log_mel = (
            log_mel - mean
        ) / (
            std + 1e-8
        )

        if log_mel.shape != (
            40,
            98,
        ):
            raise RuntimeError(
                f"Unexpected Log-Mel shape: "
                f"{log_mel.shape}"
            )

        features[index] = (
            log_mel[..., np.newaxis]
        )

        labels[index] = get_label(
            path
        )

        if (
            (index + 1) % 1000 == 0
            or index + 1 == len(paths)
        ):

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"[{index + 1}/{len(paths)}] "
                f"{(index + 1) / elapsed:.2f} files/sec"
            )

    features.flush()
    labels.flush()

    del features
    del labels

    print(
        "Saved:",
        feature_path,
    )

    print(
        "Saved:",
        label_path,
    )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    mean, std = compute_train_statistics()

    print("\n" + "=" * 60)
    print("Caching Log-Mel features")
    print("=" * 60)

    for split in [
        "train",
        "validation",
        "test",
    ]:

        print(
            f"\nProcessing {split}..."
        )

        cache_split(
            split,
            mean,
            std,
        )

    print("\n" + "=" * 60)
    print("Log-Mel caching complete")
    print("=" * 60)
