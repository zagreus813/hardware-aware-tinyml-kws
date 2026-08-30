from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from audio_utils import load_wav, pad_or_trim
from feature_extraction import extract_mfcc


# ============================================================
# Configuration
# ============================================================

DATASET_ROOT = Path(
    "~/projects/tinyml-esp32/data/speech_commands_v0.02"
).expanduser()

TARGET_CLASSES = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
]

CLASS_TO_INDEX = {
    name: index
    for index, name in enumerate(TARGET_CLASSES)
}

NUM_CLASSES = len(TARGET_CLASSES)

FEATURE_SHAPE = (13, 98, 1)

VALIDATION_FILE = DATASET_ROOT / "validation_list.txt"
TEST_FILE = DATASET_ROOT / "testing_list.txt"

NORMALIZATION_FILE = (
    Path("~/projects/tinyml-esp32/experiments")
    .expanduser()
    / "mfcc_normalization.json"
)


# ============================================================
# Split utilities
# ============================================================

def read_split_file(path: Path) -> set:
    """
    Read an official Speech Commands split file.

    Returns:
        Set of relative file paths.
    """

    with open(path, "r") as file:
        return {
            line.strip()
            for line in file
            if line.strip()
        }


def get_relative_path(path: Path) -> str:
    """
    Convert an absolute WAV path into a path relative
    to the dataset root.
    """

    return str(
        path.relative_to(DATASET_ROOT)
    )


def get_class_from_relative_path(
    relative_path: str,
) -> str:
    """
    Extract class name from:

        yes/abcdef_nohash_0.wav
    """

    return Path(relative_path).parts[0]


def get_split_files(split: str) -> List[Path]:
    """
    Return WAV paths for the requested split.

    Supported:
        train
        validation
        test
    """

    if split == "validation":
        split_files = read_split_file(
            VALIDATION_FILE
        )

        return [
            DATASET_ROOT / relative_path
            for relative_path in split_files
            if get_class_from_relative_path(
                relative_path
            ) in CLASS_TO_INDEX
        ]

    if split == "test":
        split_files = read_split_file(
            TEST_FILE
        )

        return [
            DATASET_ROOT / relative_path
            for relative_path in split_files
            if get_class_from_relative_path(
                relative_path
            ) in CLASS_TO_INDEX
        ]

    if split == "train":
        validation_files = read_split_file(
            VALIDATION_FILE
        )

        test_files = read_split_file(
            TEST_FILE
        )

        excluded = (
            validation_files
            | test_files
        )

        files = []

        for class_name in TARGET_CLASSES:

            class_dir = (
                DATASET_ROOT / class_name
            )

            for path in sorted(
                class_dir.glob("*.wav")
            ):

                relative_path = (
                    get_relative_path(path)
                )

                if relative_path not in excluded:
                    files.append(path)

        return files

    raise ValueError(
        f"Unknown split: {split}"
    )


# ============================================================
# Label utilities
# ============================================================

def get_label(path: Path) -> int:
    """
    Convert WAV path into integer class label.
    """

    class_name = path.parent.name

    if class_name not in CLASS_TO_INDEX:
        raise ValueError(
            f"Unknown class: {class_name}"
        )

    return CLASS_TO_INDEX[class_name]


# ============================================================
# Normalization
# ============================================================

def load_normalization_stats(
    path: Path = NORMALIZATION_FILE,
) -> Tuple[np.ndarray, np.ndarray]:

    with open(path, "r") as file:
        data = json.load(file)

    mean = np.asarray(
        data["mean"],
        dtype=np.float32,
    )

    std = np.asarray(
        data["std"],
        dtype=np.float32,
    )

    return mean, std


def normalize_mfcc(
    mfcc: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Normalize MFCC coefficients using statistics computed
    from the training split only.

    mean/std shape:
        (13, 1)

    mfcc shape:
        (13, 98)
    """

    return (
        mfcc - mean
    ) / (
        std + 1e-8
    )


# ============================================================
# Sample processing
# ============================================================

def process_file(
    path: Path,
    normalize: bool = False,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, int]:
    """
    Load WAV -> fixed length -> MFCC -> normalization.

    Returns:
        features: shape (13, 98, 1)
        label: integer
    """

    audio, sample_rate = load_wav(path)

    if sample_rate != 16000:
        raise ValueError(
            f"Unexpected sample rate: {sample_rate}"
        )

    audio = pad_or_trim(audio)

    mfcc = extract_mfcc(audio)

    if mfcc.shape != (13, 98):
        raise ValueError(
            f"Unexpected MFCC shape: {mfcc.shape}"
        )

    if normalize:

        if mean is None or std is None:
            raise ValueError(
                "Normalization statistics are required."
            )

        mfcc = normalize_mfcc(
            mfcc,
            mean,
            std,
        )

    # Add channel dimension for CNN:
    # (13, 98) -> (13, 98, 1)
    mfcc = mfcc[..., np.newaxis]

    return (
        mfcc.astype(np.float32),
        get_label(path),
    )


# ============================================================
# TensorFlow Dataset
# ============================================================

def create_tf_dataset(
    split: str,
    batch_size: int = 64,
    shuffle: bool = False,
    normalize: bool = True,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None,
) -> tf.data.Dataset:
    """
    Construct a tf.data.Dataset.

    Processing is lazy:
    WAV -> MFCC is performed when examples are consumed.
    """

    paths = get_split_files(split)

    def generator():

        ordered_paths = paths.copy()

        if shuffle:
            np.random.shuffle(
                ordered_paths
            )

        for path in ordered_paths:

            features, label = process_file(
                path,
                normalize=normalize,
                mean=mean,
                std=std,
            )

            yield features, label

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(
                shape=FEATURE_SHAPE,
                dtype=tf.float32,
            ),
            tf.TensorSpec(
                shape=(),
                dtype=tf.int32,
            ),
        ),
    )

    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=min(
                len(paths),
                2048,
            ),
            reshuffle_each_iteration=True,
        )

    dataset = dataset.batch(
        batch_size
    )

    dataset = dataset.prefetch(
        tf.data.experimental.AUTOTUNE
    )

    return dataset


def get_class_distribution(
    split: str,
) -> Dict[str, int]:
    """
    Count samples for each class.
    """

    counts = {
        class_name: 0
        for class_name in TARGET_CLASSES
    }

    for path in get_split_files(split):

        class_name = path.parent.name

        counts[class_name] += 1

    return counts
