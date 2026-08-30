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
    extract_mfcc,
)


# ============================================================
# Configuration
# ============================================================

CACHE_DIR = (
    Path("~/projects/tinyml-esp32/cache")
    .expanduser()
)

NORMALIZATION_FILE = (
    Path(
        "~/projects/tinyml-esp32/experiments/"
        "mfcc_normalization.json"
    ).expanduser()
)

MFCC_SHAPE = (13, 98)
FEATURE_SHAPE = (13, 98, 1)


# ============================================================
# Load normalization statistics
# ============================================================

def load_stats():
    with open(NORMALIZATION_FILE, "r") as file:
        stats = json.load(file)

    mean = np.asarray(
        stats["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        stats["std"],
        dtype=np.float32,
    )

    return mean, std


# ============================================================
# Process one split
# ============================================================

def cache_split(split, mean, std):
    paths = get_split_files(split)

    num_samples = len(paths)

    feature_path = (
        CACHE_DIR
        / f"{split}_features.npy"
    )

    label_path = (
        CACHE_DIR
        / f"{split}_labels.npy"
    )

    print("\n" + "=" * 60)
    print(f"Caching split: {split}")
    print("=" * 60)

    print("Samples:", num_samples)

    # Memory-mapped arrays.
    # The full cache does not need to stay in RAM.
    features = np.lib.format.open_memmap(
        feature_path,
        mode="w+",
        dtype=np.float32,
        shape=(
            num_samples,
            *FEATURE_SHAPE,
        ),
    )

    labels = np.lib.format.open_memmap(
        label_path,
        mode="w+",
        dtype=np.int32,
        shape=(num_samples,),
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

        mfcc = extract_mfcc(
            audio
        )

        # Normalize using TRAIN statistics.
        mfcc = (
            mfcc - mean
        ) / (
            std + 1e-8
        )

        if mfcc.shape != MFCC_SHAPE:
            raise RuntimeError(
                f"Unexpected MFCC shape "
                f"{mfcc.shape} for {path}"
        )

        features[index] = mfcc[..., np.newaxis]
        labels[index] = get_label(path)


        if (
            (index + 1) % 500 == 0
            or index + 1 == num_samples
        ):

            elapsed = (
                time.time()
                - start_time
            )

            rate = (
                (index + 1)
                / elapsed
            )

            print(
                f"[{index + 1}/{num_samples}] "
                f"{rate:.2f} files/sec"
            )

    features.flush()
    labels.flush()

    metadata = {
        "split": split,
        "num_samples": num_samples,
        "feature_shape": list(
            features.shape
        ),
        "dtype": "float32",
        "normalized": True,
        "normalization_source": (
            "training_split"
        ),
        "classes": TARGET_CLASSES,
    }

    metadata_path = (
        CACHE_DIR
        / f"{split}_metadata.json"
    )

    with open(
        metadata_path,
        "w",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )

    del features
    del labels

    print("\nSaved:")
    print(" ", feature_path)
    print(" ", label_path)
    print(" ", metadata_path)


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    mean, std = load_stats()

    print(
        "MFCC mean shape:",
        mean.shape,
    )

    print(
        "MFCC std shape:",
        std.shape,
    )

    for split in [
        "train",
        "validation",
        "test",
    ]:

        cache_split(
            split,
            mean,
            std,
        )

    print("\n" + "=" * 60)
    print("Feature caching complete")
    print("=" * 60)
