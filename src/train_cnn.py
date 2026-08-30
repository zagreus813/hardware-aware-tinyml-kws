from pathlib import Path
import json
import time

import numpy as np
import tensorflow as tf


# ============================================================
# Configuration
# ============================================================

SEED = 42

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-3

NUM_CLASSES = 6
INPUT_SHAPE = (13, 98, 1)

CACHE_DIR = (
    Path("~/projects/tinyml-esp32/cache")
    .expanduser()
)

EXPERIMENTS_DIR = (
    Path("~/projects/tinyml-esp32/experiments")
    .expanduser()
)

MODEL_PATH = (
    EXPERIMENTS_DIR
    / "small_cnn.h5"
)

HISTORY_PATH = (
    EXPERIMENTS_DIR
    / "small_cnn_history.json"
)


# ============================================================
# Reproducibility
# ============================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Load cached data
# ============================================================

print("=" * 60)
print("Loading cached MFCC features")
print("=" * 60)

train_x = np.load(
    CACHE_DIR / "train_features.npy",
    mmap_mode="r",
)

train_y = np.load(
    CACHE_DIR / "train_labels.npy",
    mmap_mode="r",
)

val_x = np.load(
    CACHE_DIR / "validation_features.npy",
    mmap_mode="r",
)

val_y = np.load(
    CACHE_DIR / "validation_labels.npy",
    mmap_mode="r",
)

test_x = np.load(
    CACHE_DIR / "test_features.npy",
    mmap_mode="r",
)

test_y = np.load(
    CACHE_DIR / "test_labels.npy",
    mmap_mode="r",
)


print("Train:", train_x.shape)
print("Validation:", val_x.shape)
print("Test:", test_x.shape)


# ============================================================
# TensorFlow datasets
# ============================================================

train_ds = (
    tf.data.Dataset
    .from_tensor_slices(
        (train_x, train_y)
    )
    .shuffle(
        2048,
        seed=SEED,
        reshuffle_each_iteration=True,
    )
    .batch(BATCH_SIZE)
    .prefetch(
        tf.data.experimental.AUTOTUNE
    )
)


val_ds = (
    tf.data.Dataset
    .from_tensor_slices(
        (val_x, val_y)
    )
    .batch(BATCH_SIZE)
    .prefetch(
        tf.data.experimental.AUTOTUNE
    )
)


test_ds = (
    tf.data.Dataset
    .from_tensor_slices(
        (test_x, test_y)
    )
    .batch(BATCH_SIZE)
)


# ============================================================
# Build Small CNN
# ============================================================

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(
            shape=INPUT_SHAPE
        ),

        tf.keras.layers.Conv2D(
            filters=16,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        ),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        tf.keras.layers.Conv2D(
            filters=24,
            kernel_size=(3, 3),
            padding="same",
            activation="relu",
        ),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        tf.keras.layers.GlobalAveragePooling2D(),

        tf.keras.layers.Dense(
            NUM_CLASSES,
            activation="softmax",
        ),
    ],
    name="small_cnn",
)


# ============================================================
# Model summary
# ============================================================

print("\n")
model.summary()


# ============================================================
# Compile
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=LEARNING_RATE
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


# ============================================================
# Callbacks
# ============================================================

EXPERIMENTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=4,
        mode="max",
        restore_best_weights=True,
    ),

    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(MODEL_PATH),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
    ),
]


# ============================================================
# Training
# ============================================================

print("\n" + "=" * 60)
print("Training Small CNN")
print("=" * 60)

start_time = time.time()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
)

training_time = (
    time.time() - start_time
)


# ============================================================
# Test evaluation
# ============================================================

print("\n" + "=" * 60)
print("Test Evaluation")
print("=" * 60)

test_loss, test_accuracy = (
    model.evaluate(
        test_ds,
        verbose=1,
    )
)


# ============================================================
# Save history
# ============================================================

history_data = {
    key: [
        float(value)
        for value in values
    ]
    for key, values in history.history.items()
}

history_data[
    "training_time_seconds"
] = float(training_time)

history_data[
    "test_loss"
] = float(test_loss)

history_data[
    "test_accuracy"
] = float(test_accuracy)

with open(
    HISTORY_PATH,
    "w",
) as file:

    json.dump(
        history_data,
        file,
        indent=2,
    )


# ============================================================
# Report
# ============================================================

print("\n" + "=" * 60)
print("Small CNN Result")
print("=" * 60)

print(
    f"Test accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print(
    f"Training time: "
    f"{training_time:.2f} seconds"
)

print(
    "Model:",
    MODEL_PATH,
)

print(
    "History:",
    HISTORY_PATH,
)
