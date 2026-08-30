from pathlib import Path
from collections import Counter

import numpy as np
import tensorflow as tf


CACHE_DIR = (
    Path("~/projects/tinyml-esp32/cache")
    .expanduser()
)

MODEL_PATH = (
    Path("~/projects/tinyml-esp32/experiments/small_cnn.h5")
    .expanduser()
)


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
# Load model
# ------------------------------------------------------------

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ------------------------------------------------------------
# Prediction
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
# Basic statistics
# ------------------------------------------------------------

print("=" * 60)
print("CNN Diagnostic")
print("=" * 60)

print(
    "True distribution:",
    Counter(y.tolist()),
)

print(
    "Predicted distribution:",
    Counter(pred.tolist()),
)

print(
    "Accuracy:",
    np.mean(pred == y),
)

print(
    "Mean maximum probability:",
    np.max(probs, axis=1).mean(),
)


# ------------------------------------------------------------
# Per-class accuracy
# ------------------------------------------------------------

print("\nPer-class accuracy:")

for class_id in range(6):

    mask = (
        y == class_id
    )

    count = int(
        mask.sum()
    )

    if count == 0:
        continue

    class_accuracy = np.mean(
        pred[mask] == class_id
    )

    print(
        f"Class {class_id}: "
        f"{class_accuracy * 100:.2f}% "
        f"({count} samples)"
    )


# ------------------------------------------------------------
# First predictions
# ------------------------------------------------------------

print("\nFirst 30 samples:")

print(
    "True:",
    y[:30].tolist(),
)

print(
    "Pred:",
    pred[:30].tolist(),
)
