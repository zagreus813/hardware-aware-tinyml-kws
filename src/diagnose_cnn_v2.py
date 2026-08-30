from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report


CACHE_DIR = (
    Path("~/projects/tinyml-esp32/cache")
    .expanduser()
)

MODEL_PATH = (
    Path("~/projects/tinyml-esp32/experiments/small_cnn_v2.h5")
    .expanduser()
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
print("CNN v2 Diagnostic")
print("=" * 60)

print(
    "\nConfusion Matrix:\n"
)

cm = confusion_matrix(
    y,
    pred,
)

print(cm)


print(
    "\nClassification Report:\n"
)

print(
    classification_report(
        y,
        pred,
        target_names=CLASS_NAMES,
        digits=4,
    )
)


print(
    "Mean max probability:",
    np.max(probs, axis=1).mean(),
)
