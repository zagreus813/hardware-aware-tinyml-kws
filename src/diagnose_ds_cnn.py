from pathlib import Path
from collections import Counter

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report


CACHE_DIR = (
    Path("~/projects/tinyml-esp32/cache")
    .expanduser()
)

MODEL_PATH = (
    Path(
        "~/projects/tinyml-esp32/"
        "experiments/ds_cnn_v1.h5"
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


x = np.load(
    CACHE_DIR / "test_features.npy",
    mmap_mode="r",
)

y = np.load(
    CACHE_DIR / "test_labels.npy",
    mmap_mode="r",
)


model = tf.keras.models.load_model(
    MODEL_PATH
)


probs = model.predict(
    x,
    batch_size=64,
    verbose=0,
)

pred = np.argmax(
    probs,
    axis=1,
)


print("=" * 60)
print("DS-CNN v1 Diagnostic")
print("=" * 60)


print("\nConfusion Matrix:")
print(
    confusion_matrix(
        y,
        pred,
    )
)


print("\nClassification Report:")
print(
    classification_report(
        y,
        pred,
        target_names=CLASS_NAMES,
        digits=4,
    )
)


print(
    "\nTrue distribution:"
)

print(
    Counter(
        y.tolist()
    )
)


print(
    "\nPredicted distribution:"
)

print(
    Counter(
        pred.tolist()
    )
)


print(
    "\nAccuracy:",
    np.mean(pred == y),
)


print(
    "Mean maximum probability:",
    np.max(
        probs,
        axis=1
    ).mean(),
)
