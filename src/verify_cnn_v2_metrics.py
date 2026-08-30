from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import accuracy_score


CACHE_DIR = (
    Path("~/projects/tinyml-esp32/cache")
    .expanduser()
)

MODEL_PATH = (
    Path("~/projects/tinyml-esp32/experiments/small_cnn_v2.h5")
    .expanduser()
)

BATCH_SIZE = 64


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


print("=" * 60)
print("CNN v2 Evaluation Verification")
print("=" * 60)

print(
    "Model:",
    MODEL_PATH,
)

print(
    "Input shape:",
    x.shape,
)

print(
    "Labels:",
    y.shape,
)


# ------------------------------------------------------------
# Evaluation through Keras
# ------------------------------------------------------------

test_ds = (
    tf.data.Dataset
    .from_tensor_slices(
        (x, y)
    )
    .batch(BATCH_SIZE)
)

loss, keras_accuracy = model.evaluate(
    test_ds,
    verbose=0,
)


# ------------------------------------------------------------
# Evaluation through prediction
# ------------------------------------------------------------

probs = model.predict(
    x,
    batch_size=BATCH_SIZE,
    verbose=0,
)

pred = np.argmax(
    probs,
    axis=1,
)

manual_accuracy = accuracy_score(
    y,
    pred,
)


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------

print("\nKeras evaluate:")
print(
    f"  loss     = {loss:.8f}"
)

print(
    f"  accuracy = {keras_accuracy:.8f}"
)

print("\nManual:")
print(
    f"  accuracy = {manual_accuracy:.8f}"
)

print("\nDifference:")
print(
    f"  {abs(keras_accuracy - manual_accuracy):.10f}"
)

print(
    "\nPredicted classes:",
    np.unique(pred),
)
