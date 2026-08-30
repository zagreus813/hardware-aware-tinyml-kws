from pathlib import Path

import numpy as np
import tensorflow as tf


CACHE_DIR = (
    Path("~/projects/tinyml-esp32/cache")
    .expanduser()
)

MODEL_PATH = (
    Path(
        "~/projects/tinyml-esp32/experiments/"
        "baseline_dense.h5"
    ).expanduser()
)

BATCH_SIZE = 64


print("=" * 60)
print("Verifying original baseline on cached test set")
print("=" * 60)


# ------------------------------------------------------------
# Load cached test data
# ------------------------------------------------------------

test_x = np.load(
    CACHE_DIR / "test_features.npy",
    mmap_mode="r",
)

test_y = np.load(
    CACHE_DIR / "test_labels.npy",
    mmap_mode="r",
)

print("Test X:", test_x.shape)
print("Test y:", test_y.shape)


# ------------------------------------------------------------
# Build TensorFlow dataset
# ------------------------------------------------------------

test_ds = tf.data.Dataset.from_tensor_slices(
    (
        test_x,
        test_y,
    )
)

test_ds = test_ds.batch(
    BATCH_SIZE
)


# ------------------------------------------------------------
# Load ORIGINAL trained baseline
# ------------------------------------------------------------

print("\nLoading:", MODEL_PATH)

model = tf.keras.models.load_model(
    MODEL_PATH
)


print("\nModel:")
model.summary()


# ------------------------------------------------------------
# Evaluate
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("Evaluation")
print("=" * 60)

loss, accuracy = model.evaluate(
    test_ds,
    verbose=1,
)


print("\n" + "=" * 60)
print("Result")
print("=" * 60)

print(
    f"Loss: {loss:.6f}"
)

print(
    f"Accuracy: {accuracy:.6f}"
)

print(
    f"Accuracy (%): {accuracy * 100:.2f}%"
)
