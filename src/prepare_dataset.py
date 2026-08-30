from pathlib import Path
import json
import time

import numpy as np

from dataset import (
    NORMALIZATION_FILE,
    TARGET_CLASSES,
    get_split_files,
)
from audio_utils import load_wav, pad_or_trim
from feature_extraction import extract_mfcc


EXPERIMENTS_DIR = (
    Path("~/projects/tinyml-esp32/experiments")
    .expanduser()
)

REPORT_FILE = (
    EXPERIMENTS_DIR
    / "dataset_report.json"
)


def compute_training_statistics():

    train_files = get_split_files("train")

    print("=" * 60)
    print("Computing MFCC normalization statistics")
    print("=" * 60)

    print(
        "Training samples:",
        len(train_files),
    )

    print(
        "Target classes:",
        ", ".join(TARGET_CLASSES),
    )

    # We calculate statistics independently
    # for each MFCC coefficient.
    #
    # Shape:
    #     mean = (13, 1)
    #     std  = (13, 1)

    sum_values = np.zeros(
        13,
        dtype=np.float64,
    )

    sum_squared = np.zeros(
        13,
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

        mfcc = extract_mfcc(
            audio
        )

        sum_values += np.sum(
            mfcc,
            axis=1,
            dtype=np.float64,
        )

        sum_squared += np.sum(
            mfcc ** 2,
            axis=1,
            dtype=np.float64,
        )

        total_frames += mfcc.shape[1]

        if index % 500 == 0:

            elapsed = (
                time.time()
                - start_time
            )

            rate = index / elapsed

            print(
                f"[{index}/{len(train_files)}] "
                f"{rate:.2f} files/sec"
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

    # Floating point errors can create tiny
    # negative values.
    variance = np.maximum(
        variance,
        0.0,
    )

    std = np.sqrt(
        variance
    )

    # Prevent division by zero.
    std = np.maximum(
        std,
        1e-6,
    )

    stats = {
        "mean": mean.reshape(
            13,
            1,
        ).tolist(),

        "std": std.reshape(
            13,
            1,
        ).tolist(),

        "num_training_files": len(
            train_files
        ),

        "total_mfcc_frames": int(
            total_frames
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

    print("\n" + "=" * 60)
    print("Normalization statistics")
    print("=" * 60)

    for index in range(13):

        print(
            f"MFCC[{index:02d}]  "
            f"mean={mean[index]: .6f}  "
            f"std={std[index]: .6f}"
        )

    print(
        "\nSaved:",
        NORMALIZATION_FILE,
    )

    return stats


def build_dataset_report():

    report = {}

    for split in [
        "train",
        "validation",
        "test",
    ]:

        files = get_split_files(
            split
        )

        class_counts = {
            class_name: 0
            for class_name in TARGET_CLASSES
        }

        for path in files:

            class_counts[
                path.parent.name
            ] += 1

        report[split] = {
            "total": len(files),
            "classes": class_counts,
        }

    with open(
        REPORT_FILE,
        "w",
    ) as file:

        json.dump(
            report,
            file,
            indent=2,
        )

    print(
        "\nSaved dataset report:",
        REPORT_FILE,
    )


if __name__ == "__main__":

    build_dataset_report()

    compute_training_statistics()
