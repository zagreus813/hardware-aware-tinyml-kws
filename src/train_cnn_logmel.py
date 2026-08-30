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
EPOCHS = 25

LEARNING_RATE = 3e-4

NUM_CLASSES = 6
INPUT_SHAPE = (40, 98, 1)

CACHE_DIR = (
    Path("~/projects/tinyml-esp32/cache_log_mel")
    .expanduser()
)

EXPERIMENTS_DIR = (
    Path("~/projects/tinyml-esp32/experiments")
    .expanduser()
)

MODEL_PATH = (
    EXPERIMENTS_DIR
    / "cnn_logmel.h5"
)

HISTORY_PATH = (
    EXPERIMENTS_DIR
    / "cnn_logmel_history.json"
)


# ============================================================
# Reproducibility
# ============================================================

np.random.seed(SEED)
tf.random.set_seed(SEED)


# ============================================================
# Load cached Log-Mel features
# ============================================================

print("=" * 60)
print("Loading cached Log-Mel features")
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
        4096,
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
# Same CNN architecture as CNN v2
# ============================================================

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(
            shape=INPUT_SHAPE
        ),

        # Block 1
        tf.keras.layers.Conv2D(
            16,
            (3, 3),
            padding="same",
            use_bias=False,
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.ReLU(),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Block 2
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            padding="same",
            use_bias=False,
        ),

        tf.keras.layers.BatchNormalization(),

        tf.keras.layers.ReLU(),

        tf.keras.layers.MaxPooling2D(
            pool_size=(2, 2)
        ),

        # Block 3
        tf.keras.layers.Conv2D(
            32,
            (3, 3),
            padding="same",
        ),

        tf.keras.layers.ReLU(),

        # Global representation
        tf.keras.layers.GlobalAveragePooling2D(),

        # Small classifier
        tf.keras.layers.Dense(
            32,
            activation="relu",
        ),

        tf.keras.layers.Dense(
            NUM_CLASSES,
            activation="softmax",
        ),
    ],
    name="cnn_logmel",
)


# ============================================================
# Summary
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
        patience=6,
        mode="max",
        restore_best_weights=True,
    ),

    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-6,
        verbose=1,
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
print("Training CNN on Log-Mel")
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
# Evaluate SAVED/BEST model
# ============================================================

print("\n" + "=" * 60)
print("Best checkpoint evaluation")
print("=" * 60)

best_model = tf.keras.models.load_model(
    MODEL_PATH
)

test_loss, test_accuracy = (
    best_model.evaluate(
        test_ds,
        verbose=1,
    )
)


# ============================================================
# Save history
# ============================================================

history_data = {
    key: [
        float(v)
        for v in values
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
# Final report
# ============================================================

print("\n" + "=" * 60)
print("CNN Log-Mel Result")
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
    "Best model:",
    MODEL_PATH,
)

print(
    "History:",
    HISTORY_PATH,
)
