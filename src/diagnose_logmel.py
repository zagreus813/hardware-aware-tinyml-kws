from pathlib import Path
from collections import Counter

import numpy as np
import tensorflow as tf


CACHE_DIR = (
    Path("~/projects/tinyml-esp32/cache_log_mel")
    .expanduser()
)

MODEL_PATH = (
    Path(
        "~/projects/tinyml-esp32/experiments/"
        "cnn_logmel.h5"
    ).expanduser()
)

CLASS_NAMES = [
    "yes",
    "no",
    "up",
    "down",
    "left",
    "right",
]


# ------------------------------------------------------------
# Load test data
# ------------------------------------------------------------

x = np.load(
    CACHE_DIR / "test_features.npy",
    mmap_mode="r",
)

y = np.load(
    CACHE_DIR / "test_labels.npy",
    mmap_mode="r",
)


# ------------------------------------------------------------
# Load best checkpoint
# ------------------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ------------------------------------------------------------
# Predictions
# ------------------------------------------------------------

probs = model.predict(
    x,
    batch_size=64,
    verbose=0,
)

pred = np.argmax(
    probs,
    axis=1,
)


# ------------------------------------------------------------
# Global statistics
# ------------------------------------------------------------

print("=" * 60)
print("Log-Mel CNN Diagnostic")
print("=" * 60)

print(
    "\nTrue distribution:"
)

print(
    Counter(y.tolist())
)

print(
    "\nPredicted distribution:"
)

print(
    Counter(pred.tolist())
)

print(
    "\nAccuracy:",
    np.mean(pred == y)
)

print(
    "Mean maximum probability:",
    np.max(probs, axis=1).mean()
)


# ------------------------------------------------------------
# Per-class recall
# ------------------------------------------------------------

print("\nPer-class recall:")

for class_id, class_name in enumerate(
    CLASS_NAMES
):

    mask = (
        y == class_id
    )

    recall = np.mean(
        pred[mask] == class_id
    )

    print(
        f"{class_name:>6}: "
        f"{recall * 100:.2f}% "
        f"({mask.sum()} samples)"
    )


# ------------------------------------------------------------
# First predictions
# ------------------------------------------------------------

print("\nFirst 30 predictions:")

print(
    "True:",
    y[:30].tolist()
)

print(
    "Pred:",
    pred[:30].tolist()
)
